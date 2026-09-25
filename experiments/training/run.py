import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any

from tqdm import tqdm
import hydra
from omegaconf import OmegaConf
import jax
import torch
import tempfile
import logging
import os

import fwh_core
from fwh_core.generative_processes.torch_generator import generate_data_batch, generate_data_batch_with_full_history
from fwh_core.structured_configs.mlflow import MLFlowConfig
from fwh_core.structured_configs.logging import LoggingConfig
from fwh_core.structured_configs.generative_process import GenerativeProcessConfig
from fwh_core.structured_configs.persistence import PersistenceConfig
from fwh_core.structured_configs.predictive_model import PredictiveModelConfig
from fwh_core.structured_configs.optimizer import OptimizerConfig
from fwh_core.structured_configs.activation_tracker import ActivationTrackerConfig
from fwh_core.structured_configs.metric_tracker import MetricTrackerConfig
from fwh_core.logging.mlflow_logger import MLFlowLogger

from optimal_loss import compute_position_optimal_loss
from utils import expand_state_by_batch_size

# Bespoke visualization imports
from visualization import plot_cev_over_training, plot_belief_regression, plot_belief_regression_grid
from visualization import plot_orthogonality_spectrum, plot_orthogonality_heatmap, plot_orthogonality_matrix
from visualization._types import CEVHistory, OrthogonalityHistory
from visualization.cev import update_cev_history, write_cev_html
from visualization.belief_regression import write_belief_regression_html
from visualization.orthogonality import update_orthogonality_history_from_scalars, write_orthogonality_html
from visualization.configs import VisualizationConfig
import numpy as np


CONFIG_DIR = str(Path(__file__).parent / "configs")
CONFIG_NAME = "train_small.yaml"

# Suppress Databricks SDK logging
logging.getLogger("databricks.sdk").setLevel(logging.WARNING)

# Set XLA_PYTHON_CLIENT_MEM_FRACTION to 0.2 to limit memory usage
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.2"


# Define training config to quiet type checkers (Optional)
@dataclass
class TrainingConfig:
    num_steps: int
    batch_size: int
    log_cheap_every: int
    log_expensive_every: int
    log_activations_every: int
    evaluate_every: int
    checkpoint_every: int
    checkpoint_schedule: list | None = None  # Adaptive schedule: list of [start, end, interval]
    analysis_batch_size: int | None = None  # Max samples for activation analysis (None = use batch_size)

# Define run config to quiet type checkers (Optional)
@dataclass
class TrainingRunConfig:
    mlflow: MLFlowConfig
    logging: LoggingConfig
    generative_process: GenerativeProcessConfig
    persistence: PersistenceConfig
    predictive_model: PredictiveModelConfig
    optimizer: OptimizerConfig
    activation_tracker: ActivationTrackerConfig
    training: TrainingConfig
    metric_tracker: MetricTrackerConfig

    experiment_name: str
    run_name: str
    seed: int
    weight_init_seed: int
    device: str
    tags: Dict[str, str]
    experiment_tags: Dict[str, str]

def should_checkpoint_at_step(step: int, schedule: list[list] | None) -> bool:
    """Determine if we should save a checkpoint at this step.

    Args:
        step: Current training step
        schedule: List of [start, end, interval] tuples. If None, returns False.
                  end=None means infinity.

    Returns:
        True if checkpoint should be saved at this step.
    """
    if schedule is None:
        return False

    for start, end, interval in schedule:
        end = end if end is not None else float('inf')
        if start <= step < end:
            return step % interval == 0

    return False


# Register a resolver named 'mult' that takes two arguments
OmegaConf.register_new_resolver("mult", lambda x, y: x * y)
OmegaConf.register_new_resolver("div", lambda x, y: x // y)

@hydra.main(config_path=CONFIG_DIR, config_name=CONFIG_NAME, version_base="1.3")
@fwh_core.managed_run(strict=False, verbose=True)
def main(cfg: TrainingRunConfig, components: fwh_core.Components):

    generative_process = components.get_generative_process()
    assert generative_process is not None
    predictive_model = components.get_predictive_model()
    assert predictive_model is not None
    logger = components.get_logger()
    assert logger is not None
    persister = components.get_persister()
    assert persister is not None
    optimizer = components.get_optimizer()
    assert optimizer is not None
    lr_scheduler = components.get_learning_rate_scheduler()  # May be None if not configured
    activation_tracker = components.get_activation_tracker()
    assert activation_tracker is not None
    training_metric_tracker = components.get_metric_tracker("training_metric_tracker")
    assert training_metric_tracker is not None
    eval_metric_tracker = components.get_metric_tracker("eval_metric_tracker")
    assert eval_metric_tracker is not None

    # Compute Bayes-optimal loss before training starts
    # This enables meaningful progress_to_optimal tracking
    bos_token = cfg.generative_process.bos_token
    eos_token = cfg.generative_process.eos_token
    context_len = cfg.predictive_model.instance.cfg.n_ctx
    sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)

    # Use a large fixed sample count for the Bayes-optimal estimate, decoupled
    # from `analysis_batch_size` (sized for the activation tracker, not MC of
    # -log P). At N=1024 the optimum estimator carries ~0.005-nat bias on the
    # feedback-transducer family — comparable to the model's distance from
    # optimum, making `progress_to_optimal` misleading. N=32768 brings the
    # bias well under 1e-3 nats (verified for telephone_fractal2 and
    # telephone_mess3 across three seeds).
    optimal_loss_num_samples = int(getattr(cfg.training, 'optimal_loss_num_samples', 32768))
    print(f"Computing Bayes-optimal loss for sequence_len={sequence_len} "
          f"with N={optimal_loss_num_samples} MC samples...")
    optimal_loss_info = compute_position_optimal_loss(
        generative_process=generative_process,
        num_samples=optimal_loss_num_samples,
        sequence_len=sequence_len,
        seed=cfg.seed + 99999,  # Use a different seed than training
    )
    optimal_loss = optimal_loss_info["average_loss"]
    print(f"  Optimal loss (position-averaged): {optimal_loss:.6f} nats")
    print(f"  Asymptotic estimate (final position): {optimal_loss_info['asymptotic_estimate']:.6f} nats")
    print(f"  Position 1 loss: {optimal_loss_info['position_1_loss']:.6f} nats")

    # Log optimal loss values as parameters for tracking
    logger.log_params({
        "optimal_loss/average": optimal_loss,
        "optimal_loss/asymptotic": optimal_loss_info["asymptotic_estimate"],
    })

    # Update metric trackers with the computed optimal loss
    # This enables meaningful progress_to_optimal tracking
    for tracker in [training_metric_tracker, eval_metric_tracker]:
        if "loss" in tracker._metrics:
            tracker._metrics["loss"].optimal_loss = optimal_loss

    # Add experiment tags if they exist (MLflow only)
    if hasattr(cfg, "experiment_tags") and isinstance(logger, MLFlowLogger):
        experiment_id = logger.experiment_id
        for k, v in cfg.experiment_tags.items():
            logger.client.set_experiment_tag(experiment_id, k, v)

    # Grab device for data generation
    # JAX doesn't support MPS and will use CPU while PyTorch model is on MPS
    model_device = next(predictive_model.parameters()).device
    device_arg = model_device if model_device.type == "mps" else None

    # Get a batch of initial states
    initial_states = expand_state_by_batch_size(generative_process.initial_state, cfg.training.batch_size)

    # Parse visualization config (with defaults if not in cfg)
    viz_cfg = VisualizationConfig.from_dict(cfg.get("visualization") if hasattr(cfg, "get") else None)

    # History storage for bespoke visualizations
    cev_history: CEVHistory = {}
    belief_regression_viz_data: dict[str, dict] = {}  # Updated by activations_analysis_step
    orthogonality_history: OrthogonalityHistory = {}  # Updated by activations_analysis_step

    # Define helper function to add a prefix to the keys of a dictionary
    # This lets us log separate sets of equivalent metric types (e.g. loss associated with training vs. evaluation)
    def add_key_prefix(d: dict[str, Any], prefix: str) -> dict[str, Any]:
        return {f"{prefix}/{k}": v for k, v in d.items()}

    # Data stream (`seed_data_stream`, top-level flag). Off (default): the
    # step-keyed stream, batch key PRNGKey(step), so the sampled data stream is
    # IDENTICAL across runs regardless of cfg.seed and seeds vary only in
    # weight init (the seed-77 canonical runs). On: the per-seed stream,
    # fold_in(PRNGKey(cfg.seed), step), making cfg.seed the process-RNG seed
    # for the entire data stream (train, eval, and end-of-run analysis
    # batches) while staying exactly reproducible given (seed, step).
    seed_data_stream = OmegaConf.select(cfg, "seed_data_stream", default=False)

    def data_key(index: int) -> jax.Array:
        if seed_data_stream:
            return jax.random.fold_in(jax.random.PRNGKey(cfg.seed), index)
        return jax.random.PRNGKey(index)

    # Define helper function to get the next batch of data
    def get_next_batch(step: int) -> tuple[jax.Array | tuple[jax.Array, ...], torch.Tensor, torch.Tensor]:
        key = data_key(step)
        bos_token = cfg.generative_process.bos_token
        eos_token = cfg.generative_process.eos_token
        context_len = cfg.predictive_model.instance.cfg.n_ctx
        sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)
        beliefs, inputs, labels = generate_data_batch(
            initial_states,
            generative_process,
            cfg.training.batch_size,
            sequence_len,
            key,
            bos_token=bos_token,
            eos_token=eos_token,
            device=device_arg,
        )
        # Move tensors to model device
        inputs = inputs.to(model_device)
        labels = labels.to(model_device)
        return beliefs, inputs, labels

    # Define helper function to compute the loss
    loss_fn = torch.nn.CrossEntropyLoss()
    def get_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return loss_fn(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1).long())

    # Define helper function to evaluate model on a fixed batch of data
    _, eval_inputs, eval_labels = get_next_batch(cfg.training.num_steps) # TODO: How should I be getting the next batch for evaluation?
    def evaluate_model() -> float:
        predictive_model.eval()
        with torch.no_grad(): # Don't build a computational graph for the evaluation
            logits = predictive_model(eval_inputs)
            loss = get_loss(logits, eval_labels)
        return loss.item()
    
    # Define helper function to run a single evaluation step
    def eval_step(step: int) -> None:
        eval_loss = evaluate_model()
        eval_metric_tracker.step(loss=eval_loss)
        eval_metrics = eval_metric_tracker.get_metrics()
        eval_metrics = add_key_prefix(eval_metrics, "eval")
        logger.log_metrics(step, eval_metrics)

    # Define helper function to run a single training step
    def train_step(step: int) -> None:
        predictive_model.train()
        _, inputs, labels = get_next_batch(step)
        logits = predictive_model(inputs)
        loss = get_loss(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step(loss.item())
        training_metric_tracker.step(tokens=inputs, loss=loss)
    
    # Define helper function to log the training metrics for a given metrics group
    def log_step(step: int, group: str) -> None:
        metrics = training_metric_tracker.get_metrics(group)
        logger.log_metrics(step, metrics)
    
    # Define helper function to run the activations analysis for a given step
    def activations_analysis_step(step: int) -> dict[str, dict]:
        """Run activation analysis and return belief regression viz data."""
        belief_regression_data: dict[str, dict] = {}
        bos_token = cfg.generative_process.bos_token
        eos_token = cfg.generative_process.eos_token
        context_len = cfg.predictive_model.instance.cfg.n_ctx
        sequence_len = context_len - int(bos_token is not None) - int(eos_token is not None)
        # Use analysis_batch_size if set, otherwise fall back to batch_size
        analysis_batch = cfg.training.analysis_batch_size or cfg.training.batch_size
        analysis_initial_states = expand_state_by_batch_size(generative_process.initial_state, analysis_batch)
        outs = generate_data_batch_with_full_history(
                analysis_initial_states,
                generative_process,
                analysis_batch,
                sequence_len,
                data_key(cfg.training.num_steps + 1),  # Fixed key index outside training range
                bos_token=bos_token,
                eos_token=eos_token,
        )
        belief_states = outs["belief_states"]
        inputs = outs["inputs"]
        assert isinstance(inputs, torch.Tensor)
        prefix_probs = outs["prefix_probabilities"]
        assert isinstance(prefix_probs, jax.Array)
        predictive_model.eval()
        with torch.no_grad():
            _, cache = predictive_model.run_with_cache(inputs, return_type="logits")
        if cache is not None:
            layer_indices = [int(name.split(".")[1]) for name in cache.keys() if name.startswith("blocks.") and name.split(".")[1].isdigit()]
            if layer_indices and any("resid" in name for name in cache.keys()):
                max_layer_idx = max(layer_indices)
                activations = {name: acts for name, acts in cache.items() if ("resid" in name and "post" in name and f"blocks.{max_layer_idx}." in name) or ("ln" in name and "normalized" in name and "final" in name)}
            else:                
                activations = {name: acts for name, acts in cache.items() if "hidden" in name}
            # # Add concatenated activations across all layers
            # ln_acts = [acts for name, acts in sorted(activations.items()) if "ln" in name and "normalized" in name]
            # resid_post_acts = [acts for name, acts in sorted(activations.items()) if "resid_post" in name]
            # if ln_acts:
            #     activations["all_ln"] = torch.cat(ln_acts, dim=-1)
            # if resid_post_acts:
            #     activations["all_resid_post"] = torch.cat(resid_post_acts, dim=-1)
                
            scalars, projections, _ = activation_tracker.analyze(
                inputs=inputs,
                beliefs=belief_states,
                probs=prefix_probs,
                activations=activations,
            )

            # Update CEV history for bespoke visualization
            update_cev_history(
                cev_history,
                projections,
                step=step,
                analysis_name="pca",
                max_components=viz_cfg.cev.max_components,
            )

            # Update orthogonality history for bespoke visualization
            if viz_cfg.orthogonality.enabled:
                update_orthogonality_history_from_scalars(
                    orthogonality_history,
                    scalars,
                    arrays=projections,  # Contains singular_values arrays
                    step=step,
                )

            # Extract belief regression data from projections for bespoke visualization
            if viz_cfg.belief_regression.enabled and projections:

                # Determine factor structure from belief_states
                if isinstance(belief_states, tuple):
                    num_factors = len(belief_states)
                    factor_dims = [np.array(b).shape[-1] for b in belief_states]
                else:
                    num_factors = 1
                    factor_dims = [np.array(belief_states).shape[-1]]

                # Get unique layer names from projection keys
                # Factored keys: "reg/projected/L0.resid.pre-F0" (format: reg/projected/{layer}-F{factor})
                # Non-factored keys: "reg/projected/L0.resid.pre" (format: reg/projected/{layer})
                import re
                layer_names_in_projections = set()
                is_factored = any(k.startswith("reg/projected/") and "-F" in k for k in projections.keys())

                if is_factored:
                    proj_pattern = re.compile(r"^reg/projected/(.+)-F(\d+)$")
                    for key in projections.keys():
                        match = proj_pattern.match(key)
                        if match:
                            layer_names_in_projections.add(match.group(1))
                else:
                    proj_pattern = re.compile(r"^reg/projected/(.+)$")
                    for key in projections.keys():
                        match = proj_pattern.match(key)
                        if match:
                            layer_names_in_projections.add(match.group(1))

                for layer_name in layer_names_in_projections:
                    # Collect predictions and targets directly from projections
                    y_pred_factors = []
                    y_true_factors = []
                    for factor_idx in range(num_factors):
                        if is_factored:
                            proj_key = f"reg/projected/{layer_name}-F{factor_idx}"
                            target_key = f"reg/targets/{layer_name}-F{factor_idx}"
                        else:
                            proj_key = f"reg/projected/{layer_name}"
                            target_key = f"reg/targets/{layer_name}"
                        if proj_key in projections and target_key in projections:
                            y_pred_factors.append(np.array(projections[proj_key]))
                            y_true_factors.append(np.array(projections[target_key]))

                    if len(y_pred_factors) != num_factors:
                        continue

                    y_pred = np.concatenate(y_pred_factors, axis=-1)
                    y_true = np.concatenate(y_true_factors, axis=-1)

                    # Compute RMSE metrics
                    factor_rmse_scores = []
                    offset = 0
                    for factor_idx in range(num_factors):
                        end = offset + factor_dims[factor_idx]
                        factor_rmse = float(np.sqrt(np.mean((y_pred[:, offset:end] - y_true[:, offset:end]) ** 2)))
                        factor_rmse_scores.append(factor_rmse)
                        offset = end

                    # Overall RMSE
                    overall_rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))

                    # Overall R²
                    ss_res = np.sum((y_true - y_pred) ** 2)
                    ss_tot = np.sum((y_true - np.mean(y_true, axis=0)) ** 2)
                    overall_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

                    # Store for visualization
                    safe_layer = layer_name.replace(".", "_")
                    belief_regression_data[safe_layer] = {
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "num_factors": num_factors,
                        "factor_dims": factor_dims,
                        "overall_rmse": overall_rmse,
                        "overall_r2": overall_r2,
                        "factor_rmse_scores": factor_rmse_scores,
                    }

                    # Log regression metrics
                    bespoke_metrics = {
                        f"belief_regression/{safe_layer}/overall_rmse": overall_rmse,
                        f"belief_regression/{safe_layer}/overall_r2": overall_r2,
                    }
                    for i, rmse in enumerate(factor_rmse_scores):
                        bespoke_metrics[f"belief_regression/{safe_layer}/factor_{i}_rmse"] = rmse
                    logger.log_metrics(step, bespoke_metrics)

            scalars = add_key_prefix(dict(scalars), "acts")
            logger.log_metrics(step, scalars)

        return belief_regression_data

    # Create a temporary directory to save the activation visualizations to in order to enable accumulation of visualizations across steps
    with tempfile.TemporaryDirectory() as visualization_path:

        def run_visualizations(step: int, belief_regression_viz_data: dict[str, dict]) -> None:
            """Generate bespoke visualizations (cev / belief_regression / orthogonality)
            for the given step. Factored out so the post-training final-step block can
            call it without duplicating the body."""
            # CEV visualization
            if viz_cfg.cev.enabled and cev_history:
                for layer_name in cev_history:
                    fig = plot_cev_over_training(cev_history, layer_name, config=viz_cfg.cev)
                    safe_name = layer_name.replace(".", "_")
                    cev_path = Path(visualization_path) / f"cev_{safe_name}_step_{step}.html"
                    write_cev_html(fig, str(cev_path))
                    logger.log_artifact(str(cev_path), artifact_path=f"figures/cev/{safe_name}")

            # Belief regression visualization
            if viz_cfg.belief_regression.enabled and belief_regression_viz_data:
                for layer_name, data in belief_regression_viz_data.items():
                    safe_name = layer_name.replace(".", "_")
                    try:
                        # 3D/per-factor view
                        fig = plot_belief_regression(
                            y_true=data["y_true"],
                            y_pred=data["y_pred"],
                            step=step,
                            layer_name=layer_name,
                            num_factors=data["num_factors"],
                            factor_dims=data["factor_dims"],
                            overall_rmse=data["overall_rmse"],
                            factor_rmse_scores=data["factor_rmse_scores"],
                            config=viz_cfg.belief_regression,
                        )
                        belief_path = Path(visualization_path) / f"belief_regression_{safe_name}_step_{step}.html"
                        write_belief_regression_html(fig, str(belief_path))
                        logger.log_artifact(str(belief_path), artifact_path=f"figures/belief_regression/{safe_name}")

                        # 2D grid view
                        fig_grid = plot_belief_regression_grid(
                            y_true=data["y_true"],
                            y_pred=data["y_pred"],
                            step=step,
                            layer_name=layer_name,
                            num_factors=data["num_factors"],
                            factor_dims=data["factor_dims"],
                            overall_rmse=data["overall_rmse"],
                            factor_rmse_scores=data["factor_rmse_scores"],
                            config=viz_cfg.belief_regression,
                        )
                        grid_path = Path(visualization_path) / f"belief_regression_grid_{safe_name}_step_{step}.html"
                        write_belief_regression_html(fig_grid, str(grid_path))
                        logger.log_artifact(str(grid_path), artifact_path=f"figures/belief_regression_grid/{safe_name}")
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Error generating belief regression for {layer_name}: {e}")

            # Orthogonality visualization
            if viz_cfg.orthogonality.enabled and orthogonality_history:
                # Extract unique layer names from history keys (format: "{layer}/F{i},F{j}")
                layer_names = set()
                for key in orthogonality_history.keys():
                    if "/" in key:
                        layer_name = key.rsplit("/", 1)[0]
                        layer_names.add(layer_name)

                for layer_name in sorted(layer_names):
                    safe_name = layer_name.replace(".", "_")
                    # Filter history to only include this layer's pairs
                    layer_history = {k: v for k, v in orthogonality_history.items() if k.startswith(f"{layer_name}/")}

                    if not layer_history:
                        continue

                    try:
                        # Plot 1: Principal angle spectrum
                        fig_spectrum = plot_orthogonality_spectrum(
                            layer_history,
                            config=viz_cfg.orthogonality,
                        )
                        spectrum_path = Path(visualization_path) / f"orthogonality_spectrum_{safe_name}_step_{step}.html"
                        write_orthogonality_html(fig_spectrum, str(spectrum_path))
                        logger.log_artifact(str(spectrum_path), artifact_path=f"figures/orthogonality/spectrum/{safe_name}")

                        # Plot 2: Timeline heatmap
                        fig_heatmap = plot_orthogonality_heatmap(
                            layer_history,
                            config=viz_cfg.orthogonality,
                        )
                        heatmap_path = Path(visualization_path) / f"orthogonality_heatmap_{safe_name}_step_{step}.html"
                        write_orthogonality_html(fig_heatmap, str(heatmap_path))
                        logger.log_artifact(str(heatmap_path), artifact_path=f"figures/orthogonality/heatmap/{safe_name}")

                        # Plot 3: Matrix snapshot (current step)
                        fig_matrix = plot_orthogonality_matrix(
                            layer_history,
                            step=step,
                            config=viz_cfg.orthogonality,
                        )
                        matrix_path = Path(visualization_path) / f"orthogonality_matrix_{safe_name}_step_{step}.html"
                        write_orthogonality_html(fig_matrix, str(matrix_path))
                        logger.log_artifact(str(matrix_path), artifact_path=f"figures/orthogonality/matrix/{safe_name}")
                    except Exception as e:
                        import warnings
                        warnings.warn(f"Error generating orthogonality visualization for {layer_name}: {e}")

        for step in tqdm(range(cfg.training.num_steps), desc="Training"):
            if step == 0:
                # Compute the initial loss and log it
                initial_loss = evaluate_model()
                training_metric_tracker.context.loss = initial_loss
                eval_metric_tracker.context.loss = initial_loss
            else:
                train_step(step)

            if step % cfg.training.log_cheap_every == 0:
                # log the cheap metrics
                log_step(step, "cheap")
                # Log learning rate if scheduler is active
                if lr_scheduler is not None:
                    current_lr = optimizer.param_groups[0]["lr"]
                    logger.log_metrics(step, {"train/learning_rate": current_lr})

            if step % cfg.training.log_expensive_every == 0:
                # log the expensive metrics
                log_step(step, "expensive")

            if step % cfg.training.log_activations_every == 0:
                belief_regression_viz_data = activations_analysis_step(step)

            # Generate bespoke visualizations
            if step % viz_cfg.every == 0:
                run_visualizations(step, belief_regression_viz_data)

            if step % cfg.training.evaluate_every == 0:
                eval_step(step)

            # Use adaptive schedule if configured, otherwise fall back to fixed interval
            if cfg.training.checkpoint_schedule:
                should_save = should_checkpoint_at_step(step, cfg.training.checkpoint_schedule)
            else:
                should_save = step % cfg.training.checkpoint_every == 0

            if should_save:
                persister.save_weights(predictive_model, step)

        # Final logging block: fire every periodic action unconditionally at
        # step == num_steps so the highest-period metrics (R^2, PCA, eval, viz,
        # checkpoint) ride out to the actual end of training. The loop above
        # iterates over range(num_steps), i.e. step 0..num_steps-1, so any
        # `step % N == 0` trigger naturally misses step == num_steps.
        final_step = cfg.training.num_steps
        log_step(final_step, "cheap")
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            logger.log_metrics(final_step, {"train/learning_rate": current_lr})
        log_step(final_step, "expensive")
        belief_regression_viz_data = activations_analysis_step(final_step)
        run_visualizations(final_step, belief_regression_viz_data)
        eval_step(final_step)
        persister.save_weights(predictive_model, final_step)

if __name__ == "__main__":
    main()
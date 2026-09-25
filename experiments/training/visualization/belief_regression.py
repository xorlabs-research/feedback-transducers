"""Belief regression visualizations for training experiments.

Visualizes the quality of linear probes predicting belief states from model activations.
Shows side-by-side scatter plots of true vs predicted belief states, colored by belief values.
"""

from __future__ import annotations

import re
import warnings
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from omegaconf import DictConfig

from visualization._types import BeliefRegressionHistory
from visualization.configs import BeliefRegressionVizConfig
from visualization.styles import (
    TITLE_FONT,
    PAPER_BG,
    PLOT_BG,
    DEFAULT_FONT,
    DEFAULT_MARGIN,
    XAXIS_DEFAULTS,
    YAXIS_DEFAULTS,
)


def _empty_figure(title: str, reason: str) -> go.Figure:
    """Create an empty figure with an explanation message."""
    fig = go.Figure()
    fig.add_annotation(
        text=reason,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="gray"),
    )
    fig.update_layout(title=title)
    return fig


def _coerce_config(
    config: BeliefRegressionVizConfig | Mapping[str, Any] | DictConfig | None,
) -> BeliefRegressionVizConfig:
    """Coerce config to BeliefRegressionVizConfig."""
    if config is None:
        return BeliefRegressionVizConfig()
    if isinstance(config, BeliefRegressionVizConfig):
        return config
    return BeliefRegressionVizConfig.from_dict(config)


def _compute_rgb_colors(
    belief_values: np.ndarray,
    dim: int,
) -> list[str]:
    """Compute RGB colors from belief state values.

    Maps first 3 dimensions of belief states to RGB channels for continuous color variation.
    """
    n_samples = len(belief_values)

    if dim >= 3:
        # Use first 3 dimensions directly for RGB
        rgb_values = belief_values[:, :3].copy()
    elif dim == 2:
        # For 2D: use the two values for R and G, set B to inverse of R
        rgb_values = np.zeros((n_samples, 3))
        rgb_values[:, 0] = belief_values[:, 0]
        rgb_values[:, 1] = belief_values[:, 1]
        rgb_values[:, 2] = 1.0 - belief_values[:, 0]
    else:
        # For 1D: create gradient from value
        rgb_values = np.zeros((n_samples, 3))
        rgb_values[:, 0] = belief_values[:, 0]
        rgb_values[:, 1] = 0.5 * belief_values[:, 0]
        rgb_values[:, 2] = 1.0 - belief_values[:, 0]

    # Normalize each channel to [0, 1]
    for i in range(3):
        min_val = rgb_values[:, i].min()
        max_val = rgb_values[:, i].max()
        if max_val > min_val:
            rgb_values[:, i] = (rgb_values[:, i] - min_val) / (max_val - min_val)
        else:
            rgb_values[:, i] = 0.5

    # Convert to RGB strings
    colors = [
        f"rgb({int(rgb_values[i, 0] * 255)},{int(rgb_values[i, 1] * 255)},{int(rgb_values[i, 2] * 255)})"
        for i in range(n_samples)
    ]
    return colors


def plot_belief_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    step: int,
    layer_name: str,
    num_factors: int,
    overall_rmse: float,
    factor_rmse_scores: list[float],
    factor_dims: list[int] | None = None,
    config: BeliefRegressionVizConfig | Mapping[str, Any] | DictConfig | None = None,
    title: str | None = None,
    title_suffix: str = "",
    sample_indices: np.ndarray | None = None,
) -> go.Figure:
    """Plot belief regression results as side-by-side true vs predicted scatter plots.

    Creates a visualization showing how well a linear probe can predict belief states
    from model activations. Each factor gets a row with true (left) and predicted (right)
    belief state scatter plots.

    Args:
        y_true: Ground truth belief states, shape [n_samples, belief_dim]
        y_pred: Predicted belief states, shape [n_samples, belief_dim]
        step: Training step number
        layer_name: Name of the layer/target being visualized
        num_factors: Number of factors in the belief state
        overall_rmse: Overall RMSE across all dimensions
        factor_rmse_scores: Per-factor RMSE scores
        factor_dims: Per-factor dimensions (if None, assumes equal dims)
        config: Visualization configuration
        title: Optional custom title
        title_suffix: Optional suffix to append to title
        sample_indices: Optional indices to subsample for visualization

    Returns:
        Plotly Figure with the visualization
    """
    cfg = _coerce_config(config)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    n_samples, belief_dim = y_true.shape

    # Handle factor dimensions - either provided or computed assuming equal dims
    if factor_dims is None:
        # Fallback: assume equal dimensions per factor
        dim_per_factor = belief_dim // num_factors
        factor_dims = [dim_per_factor] * num_factors

    # Subsample if needed
    if sample_indices is not None:
        y_true = y_true[sample_indices]
        y_pred = y_pred[sample_indices]
        n_samples = len(sample_indices)
    elif n_samples > cfg.max_samples:
        indices = np.random.choice(n_samples, cfg.max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        n_samples = cfg.max_samples

    # Build title
    if title is None:
        title = f"Belief State Regression - {layer_name} - Step {step}"
        per_factor_rmse_text = ", ".join([f"F{i}: {rmse:.3f}" for i, rmse in enumerate(factor_rmse_scores)])
        title += f"<br><sup>Overall RMSE: {overall_rmse:.4f} | Per-factor RMSE: {per_factor_rmse_text}</sup>"
        if title_suffix:
            title += f"<br><sup>{title_suffix}</sup>"

    # Determine subplot configuration based on per-factor dimensions
    subplot_specs = []
    subplot_titles = []

    for factor_idx in range(num_factors):
        factor_dim = factor_dims[factor_idx]
        if factor_dim == 2:
            subplot_specs.append([{"type": "scatter"}, {"type": "scatter"}])
            subplot_titles.extend([f"Factor {factor_idx} - True", f"Factor {factor_idx} - Predicted"])
        elif factor_dim == 3:
            subplot_specs.append([{"type": "scatter3d"}, {"type": "scatter3d"}])
            subplot_titles.extend([f"Factor {factor_idx} - True", f"Factor {factor_idx} - Predicted"])
        else:
            # Use PCA to reduce to 3D for visualization (handles both 1D and >3D)
            subplot_specs.append([{"type": "scatter3d"}, {"type": "scatter3d"}])
            subplot_titles.extend([f"Factor {factor_idx} - True (PCA)", f"Factor {factor_idx} - Pred (PCA)"])

    # Create subplots with 2 columns - minimize spacing between columns
    fig = make_subplots(
        rows=num_factors,
        cols=2,
        subplot_titles=subplot_titles,
        specs=subplot_specs,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
        row_heights=[1] * num_factors,
        column_widths=[0.48, 0.48],
    )

    offset = 0
    for factor_idx in range(num_factors):
        factor_dim = factor_dims[factor_idx]
        start_idx = offset
        end_idx = offset + factor_dim
        offset = end_idx

        # Extract true and predicted beliefs for this factor
        true_factor = y_true[:, start_idx:end_idx]
        pred_factor = y_pred[:, start_idx:end_idx]

        # Apply PCA if needed for high-dimensional beliefs
        if factor_dim > 3 and cfg.use_pca_for_high_dim:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=cfg.pca_components)
            # Fit on combined true and predicted to use same projection
            combined = np.vstack([true_factor, pred_factor])
            pca.fit(combined)
            true_factor = pca.transform(true_factor)
            pred_factor = pca.transform(pred_factor)
            dim = cfg.pca_components
        else:
            dim = factor_dim

        # Compute RGB colors from true belief state values
        colors = _compute_rgb_colors(true_factor, dim)

        if dim == 2:
            # 2D scatter plots
            fig.add_trace(
                go.Scatter(
                    x=true_factor[:, 0],
                    y=true_factor[:, 1],
                    mode="markers",
                    marker=dict(
                        size=cfg.marker_size,
                        color=colors,
                        opacity=cfg.marker_opacity,
                    ),
                    showlegend=False,
                    hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
                ),
                row=factor_idx + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter(
                    x=pred_factor[:, 0],
                    y=pred_factor[:, 1],
                    mode="markers",
                    marker=dict(
                        size=cfg.marker_size,
                        color=colors,  # Same colors for predicted
                        opacity=cfg.marker_opacity,
                    ),
                    showlegend=False,
                    hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
                ),
                row=factor_idx + 1,
                col=2,
            )

            # Update axes to have same range
            all_vals = np.concatenate([true_factor, pred_factor])
            x_range = [float(all_vals[:, 0].min() - 0.1), float(all_vals[:, 0].max() + 0.1)]
            y_range = [float(all_vals[:, 1].min() - 0.1), float(all_vals[:, 1].max() + 0.1)]

            fig.update_xaxes(range=x_range, row=factor_idx + 1, col=1)
            fig.update_xaxes(range=x_range, row=factor_idx + 1, col=2)
            fig.update_yaxes(range=y_range, row=factor_idx + 1, col=1)
            fig.update_yaxes(range=y_range, row=factor_idx + 1, col=2)

        else:  # dim >= 3
            # 3D scatter plots
            fig.add_trace(
                go.Scatter3d(
                    x=true_factor[:, 0],
                    y=true_factor[:, 1],
                    z=true_factor[:, 2],
                    mode="markers",
                    marker=dict(
                        size=cfg.marker_size,
                        color=colors,
                        opacity=cfg.marker_opacity,
                    ),
                    showlegend=False,
                    hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
                ),
                row=factor_idx + 1,
                col=1,
            )

            fig.add_trace(
                go.Scatter3d(
                    x=pred_factor[:, 0],
                    y=pred_factor[:, 1],
                    z=pred_factor[:, 2],
                    mode="markers",
                    marker=dict(
                        size=cfg.marker_size,
                        color=colors,
                        opacity=cfg.marker_opacity,
                    ),
                    showlegend=False,
                    hovertemplate="(%{x:.3f}, %{y:.3f}, %{z:.3f})<extra></extra>",
                ),
                row=factor_idx + 1,
                col=2,
            )

            # Update 3D axes to have same range
            all_vals = np.concatenate([true_factor, pred_factor])
            x_range = [float(all_vals[:, 0].min() - 0.1), float(all_vals[:, 0].max() + 0.1)]
            y_range = [float(all_vals[:, 1].min() - 0.1), float(all_vals[:, 1].max() + 0.1)]
            z_range = [float(all_vals[:, 2].min() - 0.1), float(all_vals[:, 2].max() + 0.1)]

            # Scene indices for 3D subplots
            scene_idx = factor_idx * 2 + 1
            for col_offset in range(2):
                scene_name = f"scene{scene_idx + col_offset}" if (scene_idx + col_offset) > 1 else "scene"
                fig.update_layout(
                    **{
                        scene_name: dict(
                            xaxis=dict(range=x_range),
                            yaxis=dict(range=y_range),
                            zaxis=dict(range=z_range),
                            aspectmode="cube",
                        )
                    }
                )

    # Calculate total height based on number of factors
    total_height = cfg.height_per_factor * num_factors

    # Update layout with archival styling
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        height=total_height,
        width=cfg.width,
        showlegend=False,
        margin=dict(l=20, r=20, t=120, b=20),
        font=DEFAULT_FONT,
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
    )

    return fig


def plot_belief_regression_grid(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    step: int,
    layer_name: str,
    num_factors: int,
    overall_rmse: float,
    factor_rmse_scores: list[float],
    factor_dims: list[int] | None = None,
    factor_names: list[str] | None = None,
    config: BeliefRegressionVizConfig | Mapping[str, Any] | DictConfig | None = None,
    title: str | None = None,
    title_suffix: str = "",
    sample_indices: np.ndarray | None = None,
) -> go.Figure:
    """Plot belief regression as a 2-row grid: ground truth (top) vs predictions (bottom).

    Creates a compact visualization with:
    - Row 1: Ground truth belief states for each factor
    - Row 2: Predicted belief states for each factor
    - Columns: One per factor
    - Factor names as column headers (top row only)
    - Row labels ("True", "Predicted") on the right

    All plots are 2D scatter plots. For beliefs with >2 dimensions, PCA is used
    to project to 2D.

    Args:
        y_true: Ground truth belief states, shape [n_samples, belief_dim]
        y_pred: Predicted belief states, shape [n_samples, belief_dim]
        step: Training step number
        layer_name: Name of the layer/target being visualized
        num_factors: Number of factors in the belief state
        overall_rmse: Overall RMSE across all dimensions
        factor_rmse_scores: Per-factor RMSE scores
        factor_dims: Per-factor dimensions (if None, assumes equal dims)
        factor_names: Names for each factor (if None, uses "Factor 0", "Factor 1", etc.)
        config: Visualization configuration
        title: Optional custom title
        title_suffix: Optional suffix to append to title
        sample_indices: Optional indices to subsample for visualization

    Returns:
        Plotly Figure with the 2-row grid visualization
    """
    cfg = _coerce_config(config)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape, got {y_true.shape} vs {y_pred.shape}")

    n_samples, belief_dim = y_true.shape

    # Handle factor dimensions
    if factor_dims is None:
        dim_per_factor = belief_dim // num_factors
        factor_dims = [dim_per_factor] * num_factors

    # Handle factor names
    if factor_names is None:
        factor_names = [f"Factor {i}" for i in range(num_factors)]

    # Subsample if needed
    if sample_indices is not None:
        y_true = y_true[sample_indices]
        y_pred = y_pred[sample_indices]
        n_samples = len(sample_indices)
    elif n_samples > cfg.max_samples:
        indices = np.random.choice(n_samples, cfg.max_samples, replace=False)
        y_true = y_true[indices]
        y_pred = y_pred[indices]
        n_samples = cfg.max_samples

    # Build title
    if title is None:
        title = f"Belief State Regression - {layer_name} - Step {step}"
        per_factor_rmse_text = ", ".join([f"F{i}: {rmse:.3f}" for i, rmse in enumerate(factor_rmse_scores)])
        title += f"<br><sup>Overall RMSE: {overall_rmse:.4f} | Per-factor RMSE: {per_factor_rmse_text}</sup>"
        if title_suffix:
            title += f"<br><sup>{title_suffix}</sup>"

    # Create 2-row, num_factors-column grid of 2D scatter plots (no subplot titles)
    fig = make_subplots(
        rows=2,
        cols=num_factors,
        horizontal_spacing=0.02,
        vertical_spacing=0.08,
    )

    # Process each factor
    offset = 0
    for factor_idx in range(num_factors):
        factor_dim = factor_dims[factor_idx]
        start_idx = offset
        end_idx = offset + factor_dim
        offset = end_idx

        # Extract true and predicted beliefs for this factor
        true_factor = y_true[:, start_idx:end_idx]
        pred_factor = y_pred[:, start_idx:end_idx]

        # Project to 2D if needed
        if factor_dim > 2:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            combined = np.vstack([true_factor, pred_factor])
            pca.fit(combined)
            true_factor_2d = pca.transform(true_factor)
            pred_factor_2d = pca.transform(pred_factor)
        elif factor_dim == 1:
            # For 1D, use the value as x and zeros as y
            true_factor_2d = np.column_stack([true_factor, np.zeros(n_samples)])
            pred_factor_2d = np.column_stack([pred_factor, np.zeros(n_samples)])
        else:
            true_factor_2d = true_factor
            pred_factor_2d = pred_factor

        # Compute RGB colors from true belief state values (use 2D projected values)
        colors = _compute_rgb_colors(true_factor_2d, 2)

        # Compute axis ranges to keep both plots aligned
        all_x = np.concatenate([true_factor_2d[:, 0], pred_factor_2d[:, 0]])
        all_y = np.concatenate([true_factor_2d[:, 1], pred_factor_2d[:, 1]])
        x_range = [float(all_x.min() - 0.05), float(all_x.max() + 0.05)]
        y_range = [float(all_y.min() - 0.05), float(all_y.max() + 0.05)]

        # Row 1: Ground truth
        fig.add_trace(
            go.Scatter(
                x=true_factor_2d[:, 0],
                y=true_factor_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=cfg.marker_size,
                    color=colors,
                    opacity=cfg.marker_opacity,
                ),
                showlegend=False,
                hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
            ),
            row=1,
            col=factor_idx + 1,
        )

        # Row 2: Predictions (same colors as ground truth for correspondence)
        fig.add_trace(
            go.Scatter(
                x=pred_factor_2d[:, 0],
                y=pred_factor_2d[:, 1],
                mode="markers",
                marker=dict(
                    size=cfg.marker_size,
                    color=colors,
                    opacity=cfg.marker_opacity,
                ),
                showlegend=False,
                hovertemplate="(%{x:.3f}, %{y:.3f})<extra></extra>",
            ),
            row=2,
            col=factor_idx + 1,
        )

        # Set same axis ranges for both rows, hide grid and ticks
        for row in [1, 2]:
            fig.update_xaxes(
                range=x_range,
                row=row,
                col=factor_idx + 1,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            )
            fig.update_yaxes(
                range=y_range,
                row=row,
                col=factor_idx + 1,
                showgrid=False,
                showticklabels=False,
                zeroline=False,
            )

    # Calculate dimensions
    width_per_factor = cfg.width // max(num_factors, 1)
    total_width = max(width_per_factor * num_factors, 600)
    total_height = cfg.height_per_factor  # Two rows share this height

    # Apply styling - extra top margin for title spacing
    fig.update_layout(
        title=dict(text=title, font=TITLE_FONT),
        height=total_height,
        width=total_width,
        showlegend=False,
        margin=dict(l=80, r=40, t=140, b=40),
        font=DEFAULT_FONT,
        plot_bgcolor=PAPER_BG,  # Match background for cleaner look
        paper_bgcolor=PAPER_BG,
    )

    # Add factor names as column headers above first row (format: "F0 - Name")
    for factor_idx, name in enumerate(factor_names):
        # Calculate x position for each column (centered)
        x_pos = (factor_idx + 0.5) / num_factors
        fig.add_annotation(
            text=f"F{factor_idx} - {name}",
            xref="paper",
            yref="paper",
            x=x_pos,
            y=1.03,
            showarrow=False,
            font=dict(size=14, color="#333"),
            xanchor="center",
            yanchor="bottom",
        )

    # Add row labels on the left side (vertical, larger font)
    fig.add_annotation(
        text="Ground Truth",
        xref="paper",
        yref="paper",
        x=-0.02,
        y=0.75,
        showarrow=False,
        font=dict(size=16, color="#555"),
        xanchor="center",
        yanchor="middle",
        textangle=-90,
    )
    fig.add_annotation(
        text="Prediction",
        xref="paper",
        yref="paper",
        x=-0.02,
        y=0.25,
        showarrow=False,
        font=dict(size=16, color="#555"),
        xanchor="center",
        yanchor="middle",
        textangle=-90,
    )

    return fig


_SYNC_SUBPLOTS_JS = """
<!-- subplot-sync-injected -->
<script>
// Synchronize camera (3D) and axis range (2D) between paired left/right
// subplots in a Plotly figure produced by `make_subplots(cols=2, ...)`.
// Subplot numbering pairs columns as (1,2), (3,4) — so scene↔scene2,
// scene3↔scene4, xaxis↔xaxis2, yaxis↔yaxis2, etc.
(function() {
  function axIndex(k) { var m = k.match(/(\\d+)$/); return m ? parseInt(m[1], 10) : 1; }
  function pairKeys(layout, prefix) {
    var keys = Object.keys(layout).filter(function(k) { return new RegExp('^' + prefix + '\\\\d*$').test(k); });
    keys.sort(function(a, b) { return axIndex(a) - axIndex(b); });
    var pairs = [];
    for (var i = 0; i + 1 < keys.length; i += 2) pairs.push([keys[i], keys[i + 1]]);
    return pairs;
  }
  function setupSync(gd) {
    if (!gd.layout || gd.__syncInstalled) return;
    gd.__syncInstalled = true;
    var scenePairs = pairKeys(gd.layout, 'scene');
    var xPairs = pairKeys(gd.layout, 'xaxis');
    var yPairs = pairKeys(gd.layout, 'yaxis');
    var syncing = false;
    gd.on('plotly_relayout', function(ed) {
      if (syncing) return;
      var update = {}, changed = false;
      scenePairs.forEach(function(pair) {
        var L = pair[0], R = pair[1];
        if (ed[L + '.camera']) { update[R + '.camera'] = ed[L + '.camera']; changed = true; }
        else if (ed[R + '.camera']) { update[L + '.camera'] = ed[R + '.camera']; changed = true; }
      });
      function syncRange(L, R) {
        var Lhit = ed[L + '.range[0]'] !== undefined || ed[L + '.range[1]'] !== undefined || ed[L + '.range'] !== undefined;
        var Rhit = ed[R + '.range[0]'] !== undefined || ed[R + '.range[1]'] !== undefined || ed[R + '.range'] !== undefined;
        if (Lhit && !Rhit && gd.layout[L].range) { update[R + '.range'] = gd.layout[L].range.slice(); changed = true; }
        else if (Rhit && !Lhit && gd.layout[R].range) { update[L + '.range'] = gd.layout[R].range.slice(); changed = true; }
      }
      xPairs.forEach(function(p) { syncRange(p[0], p[1]); });
      yPairs.forEach(function(p) { syncRange(p[0], p[1]); });
      if (changed) {
        syncing = true;
        Plotly.relayout(gd, update).then(function() { syncing = false; }, function() { syncing = false; });
      }
    });
  }
  function init() {
    var gds = document.querySelectorAll('.plotly-graph-div');
    if (!gds.length) { setTimeout(init, 50); return; }
    gds.forEach(setupSync);
  }
  init();
})();
</script>
"""


def write_belief_regression_html(
    fig: go.Figure,
    path: str,
    include_plotlyjs: str = "cdn",
) -> None:
    """Write belief regression figure to HTML file.

    Args:
        fig: Plotly figure to save
        path: Output file path
        include_plotlyjs: How to include Plotly.js ("cdn", "directory", True for inline)
    """
    html_content = fig.to_html(include_plotlyjs=include_plotlyjs, full_html=True)
    # Tie the left/right subplots together so rotating/zooming one mirrors
    # the change to its sibling. Injected before </body> so the script runs
    # after Plotly has attached its event handlers to the .plotly-graph-div.
    if "</body>" in html_content:
        html_content = html_content.replace("</body>", _SYNC_SUBPLOTS_JS + "</body>", 1)
    else:
        html_content += _SYNC_SUBPLOTS_JS
    with open(path, "w") as f:
        f.write(html_content)


# Regex patterns for parsing belief regression scalar metrics
_OVERALL_RMSE_RE = re.compile(r"^(?P<analysis>[^/]+)/(?P<layer>.+)/eval_overall_rmse$")
_FACTOR_RMSE_RE = re.compile(r"^(?P<analysis>[^/]+)/(?P<layer>.+)/eval_factor_(?P<index>\d+)_rmse$")


def update_belief_regression_history_from_scalars(
    history: BeliefRegressionHistory,
    scalars: Mapping[str, float],
    *,
    step: int,
    analysis_name: str = "belief_regression",
) -> None:
    """Update belief regression history dict from namespaced scalar metrics.

    Parses scalar keys like:
    - 'belief_regression/layer_0/eval_overall_rmse'
    - 'belief_regression/layer_0/eval_factor_0_rmse'

    Args:
        history: Dict to update in-place, maps layer names to list of history entries
        scalars: Scalar metrics from activation analysis
        step: Current training step
        analysis_name: Prefix for the analysis (default: "belief_regression")
    """
    prefix = f"{analysis_name}/"

    # Collect metrics by layer
    by_layer: dict[str, dict[str, Any]] = {}

    for key, value in scalars.items():
        if not key.startswith(prefix):
            continue

        # Try overall RMSE
        match = _OVERALL_RMSE_RE.match(key)
        if match and match.group("analysis") == analysis_name:
            layer = match.group("layer")
            by_layer.setdefault(layer, {})["overall_rmse"] = float(value)
            continue

        # Try per-factor RMSE
        match = _FACTOR_RMSE_RE.match(key)
        if match and match.group("analysis") == analysis_name:
            layer = match.group("layer")
            index = int(match.group("index"))
            by_layer.setdefault(layer, {}).setdefault("factor_rmse", {})[index] = float(value)
            continue

    # Build history entries for each layer
    for layer, metrics in by_layer.items():
        if "overall_rmse" not in metrics:
            warnings.warn(
                f"Incomplete belief regression metrics for layer '{layer}' at step {step}",
                UserWarning,
                stacklevel=2,
            )
            continue

        # Convert indexed dicts to ordered lists
        factor_rmse_dict = metrics.get("factor_rmse", {})

        if not factor_rmse_dict:
            # No per-factor metrics, skip
            continue

        max_factor_idx = max(factor_rmse_dict.keys())
        factor_rmse_scores = [factor_rmse_dict.get(i, 0.0) for i in range(max_factor_idx + 1)]

        entry = {
            "step": step,
            "overall_rmse": metrics["overall_rmse"],
            "factor_rmse_scores": factor_rmse_scores,
            "num_factors": max_factor_idx + 1,
        }

        history.setdefault(layer, []).append(entry)


__all__ = [
    "plot_belief_regression",
    "plot_belief_regression_grid",
    "write_belief_regression_html",
    "update_belief_regression_history_from_scalars",
]

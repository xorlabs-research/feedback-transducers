"""Conditional structures for factored generative processes.

This module provides various conditional dependency structures that define
how factors in a factored generative process depend on each other.

Available structures:
- IndependentStructure: No conditional dependencies between factors
- SequentialConditional: One-way chain dependencies (factor i depends on i-1)
- FullyConditional: Mutual dependencies between all factors
- ConditionalTransitions: Hybrid structure (independent/sequential emissions, mutual transitions)
- FeedbackTransducerStructure: Perception-action feedback loop (agent T + N action-delayed U agents)
"""

from fwh_core.generative_processes.structures.conditional_transitions import (
    ConditionalTransitions,
)
from fwh_core.generative_processes.structures.feedback_transducer import (
    FeedbackTransducerStructure,
)
from fwh_core.generative_processes.structures.fully_conditional import (
    FullyConditional,
)
from fwh_core.generative_processes.structures.independent import (
    IndependentStructure,
)
from fwh_core.generative_processes.structures.protocol import (
    ConditionalContext,
    ConditionalStructure,
)
from fwh_core.generative_processes.structures.sequential_conditional import (
    SequentialConditional,
)

__all__ = [
    "ConditionalContext",
    "ConditionalStructure",
    "ConditionalTransitions",
    "FeedbackTransducerStructure",
    "FullyConditional",
    "IndependentStructure",
    "SequentialConditional",
]

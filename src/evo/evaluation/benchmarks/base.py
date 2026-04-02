"""
Abstract base class for all evaluation benchmarks.
Implements the interface defined in the implementation plan:
    setup_env(), run_episode(), compute_metrics()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np


@dataclass
class EpisodeResultBase:
    """Base result for a single episode."""
    success: bool
    steps: int
    frames: List[np.ndarray]


class BenchmarkBase(ABC):
    """
    Abstract benchmark interface.
    All benchmark implementations (LIBERO, MetaWorld, RobotWin, etc.)
    must implement these methods.
    """

    @abstractmethod
    def setup_env(self, task: Any, config: Dict[str, Any]):
        """Setup the simulation environment for a given task."""
        ...

    @abstractmethod
    async def run_episode(self, init_state: Any, ws: Any, max_steps: int) -> EpisodeResultBase:
        """Run a single evaluation episode."""
        ...

    @abstractmethod
    def compute_metrics(self, results: List[EpisodeResultBase]) -> Dict[str, float]:
        """Compute evaluation metrics from episode results."""
        ...

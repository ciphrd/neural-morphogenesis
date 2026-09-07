"""Picklable terminal scoring output shared by workers and preview rendering."""
from dataclasses import dataclass
import numpy as np
from domain_fitness import DomainEvaluation

@dataclass
class RolloutSnapshot:
    fitness: float
    evaluation: DomainEvaluation
    positions: np.ndarray
    diagnostics: dict
    timings: dict | None = None
    generation_timings: dict | None = None

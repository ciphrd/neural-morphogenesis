"""Seeded pycma ask/tell adapter in semantically scaled policy coordinates.

The float64 search samples are retained for tell(); only the simulator-facing
weights are rounded to float32. The generation loop separately evaluates an
unchanged incumbent; only this adapter's sampled batch enters tell().
"""
from __future__ import annotations

import numpy as np


class CmaOptimizer:
    def __init__(self, reference, scales, sigma, population, seed, covariance="diagonal"):
        # Lazy import: GA runs and GPU worker processes do not need pycma.
        import cma

        self.version = cma.__version__
        self.reference = np.asarray(reference, dtype=np.float64).copy()
        self.scales = np.asarray(scales, dtype=np.float64).copy()
        if (self.reference.ndim != 1 or self.reference.size < 2
                or self.scales.shape != self.reference.shape
                or not np.isfinite(self.reference).all()
                or not np.isfinite(self.scales).all() or np.any(self.scales <= 0)):
            raise ValueError("CMA requires finite reference weights and matching positive scales")
        if not np.isfinite(sigma) or sigma <= 0 or population < 2:
            raise ValueError("CMA requires positive sigma and population >= 2")
        if covariance not in ("diagonal", "full"):
            raise ValueError("unknown CMA covariance mode")
        self.initial_sigma = float(sigma)
        self.population = int(population)
        self.covariance = covariance
        self.rng = np.random.default_rng(seed)
        self.restarts = 0
        self.generation = 0
        self.last_restart_reasons = []
        self.samples = None
        self._reset()

    def _randn(self, *shape):
        return self.rng.standard_normal(shape)

    def _reset(self):
        import cma

        self.es = cma.CMAEvolutionStrategy(np.zeros(self.reference.size), self.initial_sigma, {
            "popsize": self.population,
            "CMA_diagonal": self.covariance == "diagonal",
            "CMA_elitist": False,
            # pycma's seed=0 means time-based randomness. Supply our own RNG
            # and disable its global NumPy reseeding, including for run seed 0.
            "seed": np.nan,
            "randn": self._randn,
            "verbose": -9,
            "verb_log": 0,
            "signals_filename": "",
        })

    def ask(self):
        if self.samples is not None:
            raise RuntimeError("CMA batch must be evaluated before asking again")
        self.samples = self.es.ask()
        return [(self.reference + self.scales * sample).astype(np.float32)
                for sample in self.samples]

    def tell(self, fitnesses):
        if self.samples is None:
            raise RuntimeError("CMA tell requires an outstanding batch")
        scores = np.asarray(fitnesses, dtype=np.float64)
        if scores.shape != (self.population,):
            raise ValueError("CMA fitness count does not match its sampled population")
        finite = np.isfinite(scores)
        self.last_restart_reasons = []
        if finite.any():
            # Nonfinite rollouts are worse than every valid candidate. A finite
            # sentinel avoids contaminating pycma's fitness-history arithmetic.
            safe_scores = scores.copy()
            if not finite.all():
                # Scaling only this invalid batch preserves all valid ranks
                # and leaves room for a finite, strictly worse sentinel.
                safe_scores[finite] /= max(1., float(np.max(np.abs(scores[finite]))))
                safe_scores[~finite] = 2.
            self.es.tell(self.samples, safe_scores.tolist())
            self.last_restart_reasons = list(self.es.stop())
            if self.last_restart_reasons:
                winner = int(np.argmin(safe_scores))
                self.reference = (self.reference + self.scales * self.samples[winner]).astype(np.float32).astype(np.float64)
        else:
            # There is no ranking information to learn from; reset the search
            # distribution without adapting it to arbitrary candidate order.
            self.last_restart_reasons = ["all-invalid"]
        self.samples = None
        self.generation += 1
        if self.last_restart_reasons:
            self.restarts += 1
            self._reset()

    def diagnostics(self):
        return {"name": "cma-es", "libraryVersion": self.version,
                "covariance": self.covariance, "sigma": float(self.es.sigma),
                "parents": int(self.es.sp.mu),
                "restarts": self.restarts, "restartReasons": self.last_restart_reasons,
                "generation": self.generation}

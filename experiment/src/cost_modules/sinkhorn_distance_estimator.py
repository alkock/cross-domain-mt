from experiment.src.models import BootstrapDraws
import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
import ot

# The epsilon for sinkhorn with sqeuclidean are estiamted.
# The estimation script is in wasserstein-metric/srcv2/epsilon_estimation.ipynb
EPSILONS = {
    "dinov2": 11.267332,
    "resnet50": 2.120524,
    "resnet18": 3.591754,
    "vgg16": 3.334540,
}

# For cosine the epsilon is set to 0.01, which is the smallest epsilon
# where cosine converge for all pairs of domains in our datasets.
# Estimation with trial-and-error 10x^-4, 10x^-3 did not converge for all pairs, while 10^-2 did.


class SinkhornDistanceEstimator:
    def __init__(self, cost="sqeuclidean", model="dinov2", n_jobs=-1):
        self.cost = cost
        self.n_jobs = n_jobs
        if cost == "cosine":
            self._epsilon = 0.01
        else:
            if model not in EPSILONS:
                raise ValueError(
                    f"Unknown model '{model}'. Available: {list(EPSILONS.keys())}"
                )
            self._epsilon = EPSILONS[model]

    def compute_cost_matrix(
        self, draws: BootstrapDraws
    ) -> tuple[np.ndarray, np.ndarray]:
        print(f"computing sinkhorn distance on {self.cost} (ε={self._epsilon})")
        n = len(draws.domain_names)
        pairs = [
            (i, j, a, b)
            for i, a in enumerate(draws.domain_names)
            for j, b in enumerate(draws.domain_names)
            if i < j
        ]

        def _compute_pair(i, j, a, b):
            scores = [
                self._distance(da, db)
                for da, db in zip(draws.repeats[a], draws.repeats[b])
            ]
            return i, j, float(np.mean(scores)), float(np.std(scores))

        results = Parallel(n_jobs=self.n_jobs)(
            delayed(_compute_pair)(i, j, a, b) for i, j, a, b in pairs
        )

        mean_mat = np.zeros((n, n))
        std_mat = np.zeros((n, n))
        for i, j, mean, std in results:
            mean_mat[i, j] = mean_mat[j, i] = mean
            std_mat[i, j] = std_mat[j, i] = std
        return mean_mat, std_mat

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute the distance between two bootstrap draws.

        Args:
            a: Feature matrix of shape (N_BOOTSTRAP, D) for domain A.
            b: Feature matrix of shape (N_BOOTSTRAP, D) for domain B.

        Returns:
            Sinkhorn distance estimate between the two sets of features.
        """
        if a.shape[0] != b.shape[0]:
            raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
        n = a.shape[0]

        M = cdist(a, b, metric=self.cost)

        # ot.unif -> every image has the same weight (1/n) in the transport plan.
        # ot.sinkhorn2 returns the scalar distance directly (not the transport plan).
        w, _ = ot.sinkhorn2(
            ot.unif(n),
            ot.unif(n),
            M,
            self._epsilon,
            method="sinkhorn_log",
            log=True,
        )

        return float(w)

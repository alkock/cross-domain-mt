from experiment.src.models import BootstrapDraws
import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed


class MeanCostEstimator:
    def __init__(self, n_jobs=-1):
        self.n_jobs = n_jobs

    def compute_cost_matrix(
        self, draws: BootstrapDraws
    ) -> tuple[np.ndarray, np.ndarray]:
        print("computing mean cost")
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
            Mean cost between the two sets of features, averaged over bootstrap samples.
        """
        cost_matrix = cdist(a, b, metric="euclidean")
        return float(np.mean(cost_matrix))

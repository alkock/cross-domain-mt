from models import BootstrapDraws
import numpy as np
from scipy.spatial.distance import cdist


class MeanCostEstimator:
    def __init__(self):
        pass

    def compute_cost_matrix(
        self, draws: BootstrapDraws
    ) -> tuple[np.ndarray, np.ndarray]:
        print("computing mean cost")
        n = len(draws.domain_names)
        mean_mat = np.zeros((n, n))
        std_mat = np.zeros((n, n))
        for i, a in enumerate(draws.domain_names):
            for j, b in enumerate(draws.domain_names):
                if i >= j:
                    continue
                scores = [
                    self._distance(da, db)
                    for da, db in zip(draws.draws[a], draws.draws[b])
                ]
                mean_mat[i, j] = mean_mat[j, i] = np.mean(scores)
                std_mat[i, j] = std_mat[j, i] = np.std(scores)
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

import numpy as np
from scipy.spatial.distance import cdist
from models import BootstrapDraws


class GaussianMMDEstimator:

    def __init__(self, sigma: float | None = None):
        # If None, bandwidth is estimated per pair via median heuristic
        self.sigma = sigma

    def compute_cost_matrix(
        self, draws: BootstrapDraws
    ) -> tuple[np.ndarray, np.ndarray]:
        print("computing gaussian mmd")
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
            Gaussian MMD estimate between the two sets of features.
            MMD²(A,B) = E[k(a,a')] − 2·E[k(a,b)] + E[k(b,b')]
            where k(x,y) = exp(-||x-y||² / (2σ²)) is the RBF kernel.
            sqrt(max(0, MMD²)) is returned to ensure a real-valued distance.
        """
        sq_dists_aa = cdist(a, a, metric="sqeuclidean")
        sq_dists_bb = cdist(b, b, metric="sqeuclidean")
        sq_dists_ab = cdist(a, b, metric="sqeuclidean")

        if self.sigma is None:
            all_dists = np.concatenate(
                [
                    sq_dists_aa.ravel(),
                    sq_dists_bb.ravel(),
                    sq_dists_ab.ravel(),
                ]
            )
            sigma2 = np.median(all_dists[all_dists > 0])
        else:
            sigma2 = self.sigma**2

        def rbf(sq_d):
            return np.exp(-sq_d / (2 * sigma2))

        K_aa = rbf(sq_dists_aa)
        K_bb = rbf(sq_dists_bb)
        K_ab = rbf(sq_dists_ab)

        n, m = len(a), len(b)
        np.fill_diagonal(K_aa, 0.0)
        np.fill_diagonal(K_bb, 0.0)

        mmd2 = (
            K_aa.sum() / (n * (n - 1)) - 2.0 * K_ab.mean() + K_bb.sum() / (m * (m - 1))
        )
        return float(np.sqrt(max(0.0, mmd2)))

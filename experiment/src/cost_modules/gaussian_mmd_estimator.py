import numpy as np
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from experiment.src.models import BootstrapDraws


class GaussianMMDEstimator:

    def __init__(self, sigma: float | None = None, n_jobs: int = -1):
        self.sigma = sigma
        self.n_jobs = n_jobs

    def compute_cost_matrix(
        self, draws: BootstrapDraws
    ) -> tuple[np.ndarray, np.ndarray]:
        print("computing gaussian mmd")
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
            Gaussian MMD estimate between the two sets of features.
            MMD²(A,B) = E[k(a,a')] − 2·E[k(a,b)] + E[k(b,b')]
            where k(x,y) = exp(-||x-y||² / (2σ²)) is the RBF kernel.
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

from dataclasses import dataclass, field
import os
import numpy as np
import pandas as pd


@dataclass
class Domain:
    name: str
    path: str


@dataclass
class DomainFeatures:
    domain: Domain
    features: np.ndarray  # (N, D)


@dataclass
class BootstrapDraws:
    domain_features: list[DomainFeatures]
    repeats: dict[
        str, list[np.ndarray]
    ]  # domain.name -> list of (n_bootstrap, D) arrays
    n_repeats: int
    n_bootstrap: int

    @property
    def domains(self) -> list[Domain]:
        return [df.domain for df in self.domain_features]

    @property
    def domain_names(self) -> list[str]:
        return [df.domain.name for df in self.domain_features]


@dataclass
class CostMatrix:
    cost_name: str
    model_name: str
    domains: list[Domain]
    mean: np.ndarray  # (n_domains, n_domains)
    std: np.ndarray  # (n_domains, n_domains)

    @property
    def domain_names(self) -> list[str]:
        return [d.name for d in self.domains]

    def save(self, output_dir: str) -> None:
        model_dir = os.path.join(output_dir, self.model_name)
        os.makedirs(model_dir, exist_ok=True)
        names = self.domain_names
        pd.DataFrame(self.mean, index=names, columns=names).to_csv(
            os.path.join(model_dir, f"{self.cost_name}_mean.csv")
        )
        pd.DataFrame(self.std, index=names, columns=names).to_csv(
            os.path.join(model_dir, f"{self.cost_name}_std.csv")
        )


@dataclass
class ModelResult:
    model_name: str
    cost_matrices: list[CostMatrix] = field(default_factory=list)

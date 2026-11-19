from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from .traces import Trace


class UncertaintyEstimator(ABC):
    uses_labels = True

    @abstractmethod
    def fit(
        self, traces: Sequence[Trace], labels: Sequence[Tensor] | None = None
    ) -> "UncertaintyEstimator":
        raise NotImplementedError

    @abstractmethod
    def predict_uncertainty(self, traces: Sequence[Trace]) -> list[Tensor]:
        raise NotImplementedError


def labeled_features(
    features: Sequence[Tensor], labels: Sequence[Tensor] | None
) -> tuple[Tensor, Tensor]:
    if labels is None:
        raise ValueError("This estimator requires token labels.")
    values = torch.cat(list(features))
    targets = torch.cat(list(labels))
    labeled = targets >= 0
    if targets[labeled].unique().numel() != 2:
        raise ValueError("Training requires both clean and defective tokens.")
    return values[labeled].float(), targets[labeled]


class LookbackLens(UncertaintyEstimator):
    def __init__(self) -> None:
        self.classifier = LogisticRegression(max_iter=500)

    def fit(
        self, traces: Sequence[Trace], labels: Sequence[Tensor] | None = None
    ) -> "LookbackLens":
        features, targets = labeled_features(
            [trace["lookback"] for trace in traces], labels
        )
        self.classifier.fit(features.numpy(), targets.numpy())
        return self

    def predict_uncertainty(self, traces: Sequence[Trace]) -> list[Tensor]:
        return [
            torch.from_numpy(
                self.classifier.predict_proba(trace["lookback"])[:, 1]
            )
            for trace in traces
        ]


class _MLPProbe(UncertaintyEstimator):
    feature: str
    widths: tuple[int, ...]

    def __init__(self, epochs: int = 5, seed: int = 42) -> None:
        self.epochs = epochs
        self.seed = seed
        self.model: nn.Module

    def fit(
        self, traces: Sequence[Trace], labels: Sequence[Tensor] | None = None
    ) -> "_MLPProbe":
        features, targets = labeled_features(
            [trace[self.feature] for trace in traces], labels
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            layers = []
            widths = (features.shape[1], *self.widths)
            for width_in, width_out in zip(widths, widths[1:]):
                layers.extend((nn.Linear(width_in, width_out), nn.ReLU()))
            self.model = nn.Sequential(*layers, nn.Linear(widths[-1], 1))
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            for _ in range(self.epochs):
                for batch in torch.randperm(len(features)).split(256):
                    optimizer.zero_grad()
                    logits = self.model(features[batch]).squeeze(-1)
                    loss_fn(logits, targets[batch].float()).backward()
                    optimizer.step()
        self.model.eval()
        return self

    @torch.inference_mode()
    def predict_uncertainty(self, traces: Sequence[Trace]) -> list[Tensor]:
        return [
            self.model(trace[self.feature].float()).squeeze(-1).sigmoid()
            for trace in traces
        ]


class ICRProbe(_MLPProbe):
    feature = "icr"
    widths = (128, 64, 32)


class SAPLMA(_MLPProbe):
    feature = "saplma"
    widths = (256, 128, 64)


class HaloScope(UncertaintyEstimator):
    uses_labels = False

    def __init__(
        self, rank: int = 5, quantile: float = 0.8, seed: int = 42
    ) -> None:
        if rank < 1 or not 0 < quantile < 1:
            raise ValueError("Use rank >= 1 and a quantile between 0 and 1.")
        self.rank = rank
        self.quantile = quantile
        self.seed = seed
        self.classifier = LogisticRegression(max_iter=500)

    def fit(
        self, traces: Sequence[Trace], labels: Sequence[Tensor] | None = None
    ) -> "HaloScope":
        features = torch.cat([trace["haloscope"] for trace in traces]).float()
        if features.shape[0] < 2:
            raise ValueError("HaloScope needs at least two training tokens.")
        self.pca = PCA(
            n_components=min(self.rank, *features.shape),
            svd_solver="randomized",
            random_state=self.seed,
        )
        vectors = features.numpy()
        projections = self.pca.fit_transform(vectors)
        membership = (projections**2 * self.pca.singular_values_).mean(axis=1)
        membership = torch.from_numpy(membership)
        pseudo_labels = membership > membership.quantile(self.quantile)
        if pseudo_labels.unique().numel() != 2:
            raise ValueError("HaloScope membership scores cannot be split.")
        self.classifier.fit(vectors, pseudo_labels.numpy())
        return self

    def predict_uncertainty(self, traces: Sequence[Trace]) -> list[Tensor]:
        return [
            torch.from_numpy(
                self.classifier.predict_proba(trace["haloscope"].float())[:, 1]
            )
            for trace in traces
        ]


class _TokenHead(nn.Module):
    def __init__(self, feature_dim: int, width: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(feature_dim, width * 2),
            nn.LayerNorm(width * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(width * 2, width),
            nn.LayerNorm(width),
            nn.GELU(),
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                width, 4, width * 4, activation="gelu", batch_first=True
            ),
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.classifier = nn.Linear(width, 1)

    def forward(self, values: Tensor, padding: Tensor | None = None) -> Tensor:
        projected = self.project(values)
        features = self.encoder(projected, src_key_padding_mask=padding)
        return self.classifier(features).squeeze(-1)


class UncertaintyQuantificationHead(UncertaintyEstimator):
    def __init__(self, epochs: int = 5, seed: int = 42) -> None:
        self.epochs = epochs
        self.seed = seed
        self.model: _TokenHead

    @staticmethod
    def _features(trace: Trace) -> Tensor:
        attention = (trace["lookback"], trace["recent_attention"])
        return torch.cat(attention, dim=-1)

    def fit(
        self, traces: Sequence[Trace], labels: Sequence[Tensor] | None = None
    ) -> "UncertaintyQuantificationHead":
        if labels is None:
            raise ValueError("This estimator requires token labels.")
        features = [self._features(trace) for trace in traces]
        targets = torch.cat(list(labels))
        if targets[targets >= 0].unique().numel() != 2:
            raise ValueError("Training needs clean and defective tokens.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.model = _TokenHead(features[0].shape[1], width=64)
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
            loss_fn = nn.BCEWithLogitsLoss()
            for _ in range(self.epochs):
                for batch in torch.randperm(len(features)).split(8):
                    values = [features[index] for index in batch]
                    padded = pad_sequence(values, batch_first=True)
                    length = torch.tensor([len(row) for row in values])
                    padding = torch.arange(padded.shape[1]) >= length[:, None]
                    target = pad_sequence(
                        [labels[index] for index in batch],
                        batch_first=True,
                        padding_value=-100,
                    )
                    labeled = target >= 0
                    if labeled.any():
                        optimizer.zero_grad()
                        logits = self.model(padded, padding)
                        actual = target[labeled].float()
                        loss = loss_fn(logits[labeled], actual)
                        loss.backward()
                        optimizer.step()
        self.model.eval()
        return self

    @torch.inference_mode()
    def predict_uncertainty(self, traces: Sequence[Trace]) -> list[Tensor]:
        return [
            self.model(self._features(trace).unsqueeze(0))[0].sigmoid()
            for trace in traces
        ]

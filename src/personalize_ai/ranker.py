from __future__ import annotations

from pathlib import Path
import time

from .monitoring import CHALLENGER_LATENCY


class BPRChallenger:
    """Loads the training-compatible BPR checkpoint for shadow or live ranking."""

    def __init__(self, checkpoint_path: Path) -> None:
        import torch
        from torch import nn

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        item_mapping = {int(key): int(value) for key, value in checkpoint["item_to_index"].items()}

        class BPRModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                dimension = int(config["embedding_dim"])
                feature_count = int(config["feature_count"])
                self.item_embedding = nn.Embedding(
                    len(item_mapping) + 1, dimension, padding_idx=0
                )
                self.correction = nn.Sequential(
                    nn.Linear(dimension * 4 + feature_count, 32),
                    nn.ReLU(),
                    nn.Dropout(0.30),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 1),
                )

            def forward(self, candidate, history, features):
                candidate_embedding = self.item_embedding(candidate)
                history_embeddings = self.item_embedding(history)
                mask = (history != 0).unsqueeze(-1)
                history_embedding = (history_embeddings * mask).sum(dim=1) / mask.sum(
                    dim=1
                ).clamp(min=1)
                inputs = torch.cat(
                    [
                        candidate_embedding,
                        history_embedding,
                        candidate_embedding * history_embedding,
                        torch.abs(candidate_embedding - history_embedding),
                        features,
                    ],
                    dim=1,
                )
                retrieval_prior = 2.0 * features[:, 8] + 0.02 * features[:, 2]
                return retrieval_prior + self.correction(inputs).squeeze(1)

        self.torch = torch
        self.model = BPRModel()
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.item_mapping = item_mapping
        self.max_history = int(config.get("max_history", 50))
        self.model_version = checkpoint_path.name

    def score(
        self, candidates: list[int], history: list[int], features: list[list[float]]
    ) -> list[float]:
        started = time.perf_counter()
        torch = self.torch
        encoded_history = [self.item_mapping.get(item, 0) for item in history[: self.max_history]]
        encoded_history += [0] * (self.max_history - len(encoded_history))
        history_tensor = torch.tensor(
            [encoded_history for _ in candidates], dtype=torch.long
        )
        with torch.inference_mode():
            values = self.model(
                torch.tensor(
                    [self.item_mapping.get(item, 0) for item in candidates],
                    dtype=torch.long,
                ),
                history_tensor,
                torch.tensor(features, dtype=torch.float32),
            )
        result = [float(value) for value in values.tolist()]
        CHALLENGER_LATENCY.observe(time.perf_counter() - started)
        return result

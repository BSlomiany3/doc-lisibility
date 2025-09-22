import torch
from torch import nn

from docReadibilityScorer_app.scorer import document_confidence


class DummyModel(nn.Module):
    """Petit modèle jouet qui sort des logits prédictibles."""
    def __init__(self, num_classes=3):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(28 * 28, num_classes)

    def forward(self, x, log_activations=False):
        logits = self.fc(self.flatten(x))
        if log_activations:
            return logits, {"dummy": logits}
        return logits


def test_document_confidence_basic():
    model = DummyModel(num_classes=3)

    list_chars = [torch.zeros(28, 28) for _ in range(5)]

    conf = document_confidence(model, list_chars)

    assert isinstance(conf, torch.Tensor)
    assert conf.ndim == 0
    assert 0.0 <= conf.item() <= 1.0
    assert conf.item() >= 0

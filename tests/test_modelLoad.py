import torch
import pytest

from docReadibilityScorer_app.model import VGGLikeNN


@pytest.mark.parametrize("batch_size", [1, 8])
def test_vgglikenn_forward(batch_size):
    """Vérifie que le modèle se charge et effectue un forward correct."""
    num_classes = 63
    model = VGGLikeNN([(1, 32), (32, 256), (256, 512), (512, 1024)], 63)

    x = torch.randn(batch_size, 1, 28, 28)

    logits = model(x)
    assert logits.shape == (batch_size, num_classes)

    logits2, activations = model(x, log_activations=True)
    assert logits2.shape == (batch_size, num_classes)
    assert "block1_out" in activations
    assert "block2_out" in activations
    assert "logits" in activations
    
    assert torch.equal(logits2, activations["logits"])

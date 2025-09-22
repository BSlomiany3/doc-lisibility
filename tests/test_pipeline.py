import torch
import pytest
from pathlib import Path
from shutil import copyfile

from docReadibilityScorer_app._pipeline import pipeline
from docReadibilityScorer_app.model import VGGLikeNN


@pytest.fixture
def mock_input_file(tmp_path):
    """Copie un mock .png dans un dossier temporaire pour le test."""
    tests_dir = Path(__file__).resolve().parent
    mock_dir = tests_dir / "mock_data"
    mock_png = next(mock_dir.glob("*.png"), None)
    
    dst = tmp_path / mock_png.name
    copyfile(mock_png, dst)
    return dst


@pytest.fixture
def pretrained_model():
    """Charge le modèle pré-entraîné inclus dans le package."""
    model_path = Path(__file__).resolve().parents[1] / "src" / "docReadibilityScorer_app" /  "fine_tuned_vgg.pth"

    model = VGGLikeNN([(1, 32), (32, 256), (256, 512), (512, 1024)], 63)
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict)
    return model


def test_pipeline_with_pretrained_and_mock(mock_input_file, pretrained_model, tmp_path):
    """Teste la pipeline complète avec un mock document et le modèle pré-entraîné du package."""
    output_folder = tmp_path / "outputs"

    new_file_path = pipeline(
        folder=mock_input_file.parent,
        file=mock_input_file.name,
        best_model=pretrained_model,
        output_folder=output_folder,
    )

    assert new_file_path.exists()
    assert new_file_path.parent == output_folder

    assert "score__" in new_file_path.name
    assert new_file_path.suffix == ".png"

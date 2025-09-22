import shutil
from pathlib import Path
import torch

from .segmentation import _load_binarize, _characters_extraction, _padding
from .scorer import document_confidence
from .model import VGGLikeNN


def pipeline(folder: str | Path, file: str, best_model: torch.nn.Module, output_folder: str | Path) -> Path:
    """
    Exécute la pipeline complète : binarisation, segmentation des caractères,
    padding, scoring du document avec le modèle, puis déplacement du fichier
    vers un dossier de sortie avec un nom enrichi du score.

    Paramètres
    ----------
    folder : str | Path
        Dossier contenant le fichier à traiter.
    file : str
        Nom du fichier à traiter (ex: 'doc.png').
    best_model : torch.nn.Module
        Modèle entraîné utilisé pour évaluer la lisibilité.
    output_folder : str | Path
        Dossier de sortie où déplacer le fichier traité.

    Retour
    ------
    Path
        Nouveau chemin du fichier déplacé (contenant le score dans son nom).
    """
    folder = Path(folder)
    output_folder = Path(output_folder)
    file_path = folder / file

    if not file_path.exists():
        raise FileNotFoundError(f"Fichier introuvable: {file_path}")

    bin_img = _load_binarize(file_path)
    _, _, chars = _characters_extraction(bin_img)
    list_chars = _padding(chars)

    score = float(document_confidence(best_model, list_chars).item())
    print(score)

    base_name = file_path.stem
    ext = file_path.suffix
    new_file_name = f"score__{score:.2f}__{base_name}{ext}"

    destination = output_folder / new_file_name
    output_folder.mkdir(parents=True, exist_ok=True)

    shutil.move(str(file_path), str(destination))

    return destination

### test parametrisation

folder = "src/data/input"
file = "doc6.png"
output_folder = "src/data/output"
model_path = Path(__file__).resolve().parents[1] / "docReadibilityScorer_app" /  "fine_tuned_vgg.pth"
model = VGGLikeNN([(1, 32), (32, 256), (256, 512), (512, 1024)], 63)
model.load_state_dict(torch.load(model_path))

pipeline(folder, file, model, output_folder)
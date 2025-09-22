from pathlib import Path
import numpy as np
import pytest

from docReadibilityScorer_app import segmentation

def _find_first_png_in_mock():
    root = Path("tests")
    mock_dir = root / "mock_data"

    if not mock_dir.exists() or not mock_dir.is_dir():
        pytest.skip("Dossier mock_data introuvable — placez-y un fichier .png pour exécuter ce test.")

    png_files = list(mock_dir.glob("*.png"))
    if not png_files:
        pytest.skip("Aucun fichier .png trouvé dans mock_data — ajoutez un exemple .png.")
    return png_files[0]


def test_load_binarize_from_mock():
    """Vérifie que _load_binarize lit une image .png et renvoie un tableau 2D uint8 avec valeurs {0,1}."""
    png_path = _find_first_png_in_mock()
    arr = segmentation._load_binarize(png_path)
    assert isinstance(arr, np.ndarray), "Retourne bien un numpy.ndarray"
    assert arr.dtype == np.uint8, "dtype doit être uint8"
    assert arr.ndim == 2, "Image binarisée attendue en 2D (grayscale)"
    uniques = set(np.unique(arr).tolist())
    assert uniques.issubset({0, 1}), f"Valeurs attendues dans {{0,1}}, trouvé: {uniques}"


def test_pipeline_extract_and_pad_first_character():
    """
    Pipeline minimal :
    - charge la première image mock,
    - extrait caractères,
    - pad le premier caractère et vérifie la taille/format.
    Si aucun caractère détecté, le test est skip.
    """
    png_path = _find_first_png_in_mock()
    bin_img = segmentation._load_binarize(png_path)

    line_breaks, writings, characters = segmentation._characters_extraction(bin_img)

    if not characters:
        pytest.skip("Aucun caractère détecté dans l'image mock — impossible de tester le padding.")

    first_char = characters[0]
    assert isinstance(first_char, np.ndarray)
    assert first_char.ndim == 2

    padded_list = segmentation._padding([first_char])
    assert isinstance(padded_list, list) and len(padded_list) == 1
    padded = padded_list[0]

    assert padded.shape == (28, 28), "Le caractère doit être redimensionné/paddé en 28x28"
    assert padded.dtype == np.uint8
    assert set(np.unique(padded)).issubset({0, 1})

    expected_ink = int(np.sum(first_char == 0))
    actual_ink = int(np.sum(padded == 1))
    assert actual_ink == expected_ink, "Le padding doit préserver le nombre de pixels d'encre"

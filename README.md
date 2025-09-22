# Doc-Lisibility

Un outil de **scoring de lisibilité des documents scannés**.  
Le but est de séparer automatiquement les scans de **bonne qualité** de ceux de **mauvaise qualité**, grâce à une pipeline basée sur un modèle convolutionnel (type VGG-like) entraîné sur des caractères extraits.

---

## 📂 Structure du projet

- `research/`  
  Contient les expériences et prototypes :  
  - limites des approches classiques ML,  
  - segmentation manuelle,  
  - comparaison de modèles,  
  - fine-tuning du meilleur candidat.

- `src/docReadibilityScorer_app/`  
  Package Python principal :  
  - `segmentation.py` → binarisation et extraction des caractères  
  - `model.py` → définition du modèle convolutionnel  
  - `scorer.py`, `utils.py` → fonctions de scoring/auxiliaires  
  - `pipeline.py` → pipeline complète
  - `fine_tuned_vgg.pth` → modèle pré-entraîné utilisé dans la pipeline

- `tests/`  
  Tests unitaires et d’intégration avec **pytest**, incluant un dossier `mock_data/` pour les exemples `.png`.

- `documentation/`  
  Documentation MkDocs.

---

## ⚙️ Installation

Cloner le repo et installer en mode développement :

```bash
git clone https://github.com/BSlomiany3/doc-lisibility.git
cd doc-lisibility
python -m venv .venv
source .venv/Scripts/activate
pip install -e .[dev]
```

---

## ⚙️ Utilisation

- Ajout des images à scorer dans le dossier `src/data/input/`
- Modification des inputs du pipeline dans le fichier `pipeline.py`
- run `python -m docReadibilityScorer_app._pipeline` à la ligne de commande
# Doc-Lisibility

Bienvenue dans la documentation de **Doc-Lisibility**, un outil qui permet de scorer automatiquement
la lisibilité de documents scannés, afin de séparer les scans de bonne qualité des scans de mauvaise qualité.

---

## Objectif

- Automatiser l’évaluation de la qualité des scans
- Faciliter le traitement de gros volumes de documents
- Mettre à disposition une API Python simple (`pipeline`) et un package réutilisable

---

## Fonctionnalités

- Prétraitement et binarisation des scans
- Extraction et segmentation des caractères
- Normalisation des caractères pour prédiction
- CNN VGG-like pré-entraîné
- Pipeline complète : fichier → score → renommage/déplacement

---

## Lien avec la recherche

Le dossier `research/` présente les travaux exploratoires :
- Limites des méthodes ML classiques
- Tests de segmentation manuelle
- Comparaison d’architectures CNN
- Fine-tuning du meilleur modèle retenu

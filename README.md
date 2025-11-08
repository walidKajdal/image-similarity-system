# Image Similarity System

Un système de recherche d'images similaires utilisant l'extraction de caractéristiques et différentes métriques de similarité.

## Description

Ce projet implémente un système de recherche d'images similaires basé sur l'analyse de contenu visuel. Le système utilise plusieurs méthodes d'extraction de caractéristiques (histogrammes de couleurs, moments statistiques) et métriques de similarité (cosinus, euclidienne) pour trouver les images les plus similaires dans un dataset.

## Fonctionnalités

- **Chargement automatique** de datasets avec support des sous-dossiers
- **Extraction de caractéristiques** :
  - Histogrammes de couleurs RGB
  - Moments statistiques de couleur
  - Pixels bruts (optionnel)
- **Métriques de similarité** :
  - Similarité cosinus
  - Distance euclidienne
- **Comparaison automatique** des différentes méthodes
- **Visualisation** des résultats avec matplotlib

## Installation

1. Clonez le repository :
```bash
git clone https://github.com/walidKajdal/image-similarity-system.git
cd image-similarity-system
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

1. Modifiez le chemin du dataset dans le script `image_similarity.py` :
```python
dataset_path = "chemin/vers/votre/dataset/"
```

2. Exécutez le script :
```bash
python image_similarity.py
```

## Configuration

- `dataset_path` : Chemin vers le dossier contenant les images
- `reference_idx` : Index de l'image de référence (0-based)
- `k` : Nombre d'images similaires à retourner
- `image_size` : Taille de redimensionnement des images (défaut: 128x128)

## Structure du Dataset

Le système supporte une structure hiérarchique :
```
dataset/
├── classe1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── classe2/
│   ├── image1.jpg
│   └── ...
└── ...
```

## Formats Supportés

- JPG/JPEG
- PNG
- BMP

## Exemple de Sortie

Le système affiche :
- Les images similaires trouvées
- Les scores de similarité
- Une comparaison des performances des différentes méthodes
- Graphiques de visualisation

## Licence

Ce projet est sous licence MIT.

## Auteur

Walid Kajdal

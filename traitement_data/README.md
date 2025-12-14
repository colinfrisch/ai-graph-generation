# Scripts Disponibles - Guide Rapide

## 📝 Tous les Scripts du Projet

### 1. `test_pipeline.py` - Test Rapide (SANS dataset)
**Ce qu'il fait** : Teste le pipeline complet avec une image synthétique

**Utilisation** :
```bash
python test_pipeline.py
```

**Durée** : 1-2 minutes

**Sortie** :
- `outputs/visualizations/test_pipeline.png`
- Affiche toutes les étapes du pipeline

**Quand l'utiliser** : Pour vérifier que tout fonctionne avant de télécharger le dataset

---

### 2. `explore_dataset.py` - Exploration Dataset
**Ce qu'il fait** : Explore le dataset réel et sauvegarde des exemples

**Utilisation** :
```bash
python explore_dataset.py
```

**Durée** : 5-10 minutes (télécharge le dataset)

**Sortie** :
- `data/raw/sample_000.png` à `sample_009.png`
- Statistiques dans le terminal

**Quand l'utiliser** : Première fois que vous voulez voir le dataset

---

### 3. `single_image_analysis.py` - Analyse UNE Image en Détail
**Ce qu'il fait** : Analyse complète d'une seule image avec tous les détails

**Utilisation** :
```bash
python single_image_analysis.py 0      # Analyse l'image 0
python single_image_analysis.py 42     # Analyse l'image 42
python single_image_analysis.py 100    # Analyse l'image 100
```

**Durée** : 10-20 secondes par image

**Sortie** :
- `outputs/visualizations/image_X_analysis.png`
- Détails complets dans le terminal :
  ```
  Node 0: pos=(150, 200), shape=circle, size=(80, 80)
  Edge 0: from (150, 200) to (400, 150), length=268.7px

  Features: nodes=8, edges=12, density=0.167
  Type: sparse, Complexity: moderate
  ```

**Quand l'utiliser** :
- Objectif 2 : Extraire info d'une image spécifique
- Voir tous les détails (nœuds, arêtes, positions, formes)

---

### 4. `batch_analysis.py` - Analyse de 100 Images
**Ce qu'il fait** : Traite 100 images et génère statistiques complètes

**Utilisation** :
```bash
python batch_analysis.py              # Traite 100 images
python batch_analysis.py --num 50     # Traite 50 images
python batch_analysis.py --num 200    # Traite 200 images
```

**Durée** : 15-30 minutes (selon nombre d'images)

**Sortie** :
- `outputs/results/dataset_features.csv` - Features de toutes les images
- `outputs/results/enriched_captions.csv` - Captions enrichis
- `outputs/results/analysis_report.txt` - Rapport texte
- `outputs/visualizations/distribution_analysis.png` - Graphiques

**Quand l'utiliser** :
- Comprendre la distribution du dataset
- Enrichir les descriptions de plusieurs images
- Générer des statistiques pour le rapport

**Ce que vous obtenez** :
```
dataset_features.csv:
  num_nodes  num_edges  graph_density  complexity  ...
  8          12         0.167          moderate
  15         28         0.267          complex
  ...

enriched_captions.csv:
  original                      augmented                                        graph_type  complexity
  "A diagram..."                "A diagram... [Visual: sparse, moderate | ...]" sparse      moderate
```

---

### 5. `cv_validation.py` - Validation Computer Vision
**Ce qu'il fait** : Prouve que c'est un projet CV avec visualisations de chaque technique

**Utilisation** :
```bash
python cv_validation.py 0      # Valide sur l'image 0
python cv_validation.py 5      # Valide sur l'image 5
```

**Durée** : 10-20 secondes

**Sortie** :
- `outputs/visualizations/cv_validation.png` - 9 panels montrant :
  1. Image originale
  2. Grayscale
  3. CLAHE
  4. Bilateral filter
  5. Canny edges
  6. Otsu threshold
  7. Connected components
  8. Hough Transform
  9. Liste des concepts CV

**Quand l'utiliser** :
- Valider que c'est du Computer Vision
- Créer une figure pour le rapport
- Montrer tous les concepts CV appliqués

---

### 6. `demo_pipeline.py` - Démo Complète avec Dataset
**Ce qu'il fait** : Démo complète sur images réelles (comme `test_pipeline.py` mais avec vrai dataset)

**Utilisation** :
```bash
python demo_pipeline.py
```

**Durée** : 5-10 minutes

**Sortie** :
- Traite ~10 images
- `outputs/visualizations/pipeline_demo.png`
- `outputs/results/extracted_features.csv`

**Quand l'utiliser** : Démo complète pour présentation

---

## 🎯 Quel Script pour Quel Objectif ?

### Objectif 1 : Comprendre la dataset visuellement
```bash
python explore_dataset.py        # Voir des exemples
python batch_analysis.py         # Statistiques complètes
```
→ Résultats : `dataset_features.csv`, `distribution_analysis.png`

### Objectif 2 : Extraire information des images
```bash
python single_image_analysis.py 0    # Une image en détail
python batch_analysis.py             # Extraction sur 100 images
```
→ Résultats : Nœuds, arêtes, positions affichés dans le terminal + CSV

### Objectif 3 : Enrichir descriptions textuelles
```bash
python batch_analysis.py         # Génère enriched_captions.csv
```
→ Résultats : `enriched_captions.csv` avec colonnes original/augmented

### Objectif 4 : Valider projet Computer Vision
```bash
python cv_validation.py 0        # Figure montrant tous les concepts CV
```
→ Résultats : `cv_validation.png` + liste concepts dans terminal

---

## 📊 Workflow Recommandé

### Découverte (15 minutes)
```bash
# 1. Test rapide sans dataset
python test_pipeline.py

# 2. Explorer le dataset
python explore_dataset.py

# 3. Analyser une image en détail
python single_image_analysis.py 0
```

### Analyse Complète (30 minutes)
```bash
# 4. Traiter 100 images
python batch_analysis.py

# 5. Validation CV
python cv_validation.py 0
```

### Pour le Rapport (5 minutes)
```bash
# Générer toutes les figures nécessaires
python cv_validation.py 5
python batch_analysis.py --num 100
```

Résultats dans `outputs/` :
- `visualizations/cv_validation.png` → Concepts CV
- `visualizations/distribution_analysis.png` → Statistiques dataset
- `results/dataset_features.csv` → Données brutes
- `results/enriched_captions.csv` → Enrichissements
- `results/analysis_report.txt` → Rapport texte

---

## 🆘 Problèmes Fréquents

### "Module not found"
```bash
# Vérifier que le venv est activé
./venv/Scripts/activate  # Windows

# Réinstaller
pip install -r requirements.txt
```

### "Dataset download fails"
- Vérifier connexion internet
- Le dataset fait plusieurs GB, attendre quelques minutes
- Utiliser `test_pipeline.py` pour tester sans dataset

### "Out of memory"
```bash
# Traiter moins d'images
python batch_analysis.py --num 50
```

---

## 📁 Arborescence des Sorties

```
outputs/
├── visualizations/
│   ├── test_pipeline.png              # Test synthétique
│   ├── cv_validation.png              # Validation CV (objectif 4)
│   ├── distribution_analysis.png      # Stats dataset (objectif 1)
│   ├── image_0_analysis.png           # Détails image 0 (objectif 2)
│   ├── image_1_analysis.png
│   ├── pipeline_demo.png              # Démo complète
│   └── batch/
│       ├── sample_000.png
│       └── ...
└── results/
    ├── dataset_features.csv           # Features extraites (objectif 1, 2)
    ├── enriched_captions.csv          # Captions enrichis (objectif 3)
    ├── analysis_report.txt            # Rapport statistiques
    └── extracted_features.csv         # Demo pipeline
```

---

## 🚀 Commandes Ultra-Rapides

```bash
# Test complet en 3 commandes
python test_pipeline.py             # 2 min - Test sans dataset
python batch_analysis.py            # 30 min - Analyse complète
python cv_validation.py 0           # 1 min - Validation CV

# Tout est dans outputs/ après ça !
```

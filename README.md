# Grid-run + Combined Heatmaps

Ce dépôt contient des scripts pour lancer une grille d'entraînement sur différentes paires d'hyperparamètres (par ex. LR vs LR_DECAY) pour deux modèles (CNN et MLP), collecter les métriques et générer des heatmaps comparatives (CNN vs MLP) pour les métriques `accuracy`, `loss` et `train_time_s`.

Fichiers clés
- `run_grid.sh` (alias historique) and `run_grid_lr_lrdecay.sh` : lance la grille pour la paire LR vs LR_DECAY et produit un CSV par modèle (ex: `results_lr_vs_lrdecay_cnn.csv`, `results_lr_vs_lrdecay_mlp.csv`).
- `run_grid_lr_batch.sh` : grille LR vs BATCH (produit `results_lr_vs_batch_*.csv`).
- `run_grid_lr_patience.sh` : grille LR vs PATIENCE (produit `results_lr_vs_patience_*.csv`).
- `run_grid_lr_angle.sh` : grille LR vs ANGLE (produit `results_lr_vs_angle_*.csv`).
- `scripts/append_from_log.py` : parse les logs et ajoute une ligne dans le CSV correspondant ; maintenant accepte une liste optionnelle de paramètres supplémentaires (ex: `BATCH`, `PATIENCE`, `ANGLE`) qui seront lus depuis les variables d'environnement et ajoutés au CSV.
- `plot_combined_heatmaps.py` : lit plusieurs CSV (un par modèle) et génère, pour chaque métrique, une image contenant les heatmaps des modèles côte-à-côte. Tu peux passer `--prefix` pour nommer les fichiers selon la paire d'hyperparamètres (ex: `lr_vs_lrdecay_accuracy_combined.png`).
- `plots/` : dossier de sortie pour les images.

Pré-requis
- Python 3
- Bibliothèques Python pour les plots : pandas, matplotlib, seaborn

Installation rapide :

```bash
pip install pandas matplotlib seaborn
```

Usage rapide — exemples

1) Recréer les heatmaps pour LR vs LR_DECAY (si CSV existants à la racine)

```bash
python3 plot_combined_heatmaps.py --csv results_lr_vs_lrdecay_cnn.csv results_lr_vs_lrdecay_mlp.csv --labels cnn mlp --out plots --prefix lr_vs_lrdecay
```

2) Lancer la grille complète (LR vs LR_DECAY)

- Dry run (prévisualise les commandes sans exécuter) :

```bash
DRY_RUN=1 bash run_grid.sh
```

- Lancer la grille réelle (exemple rapide) :

```bash
STEPS=15 bash run_grid.sh
```

3) Lancer une grille différente (exemples)

- LR vs BATCH :

```bash
DRY_RUN=1 bash run_grid_lr_batch.sh
# ou run réellement
STEPS=15 bash run_grid_lr_batch.sh
```

- LR vs PATIENCE :

```bash
DRY_RUN=1 bash run_grid_lr_patience.sh
STEPS=15 bash run_grid_lr_patience.sh
```

- LR vs ANGLE :

```bash
DRY_RUN=1 bash run_grid_lr_angle.sh
STEPS=15 bash run_grid_lr_angle.sh
```

Notes techniques
- Les scripts créent un CSV par modèle. Les CSV contiennent au moins les colonnes : `<PARAM1>,<PARAM2>,loss,accuracy,train_time_s` où `<PARAM1>` et `<PARAM2>` correspondent à la paire testée (ex: `LR,PATIENCE`).
- `scripts/append_from_log.py` prend désormais un troisième paramètre optionnel (liste comma-separated) indiquant les noms de paramètres supplémentaires à lire depuis les variables d'environnement et à inclure dans la CSV. Les run scripts appellent ce script en lui passant exactement les deux paramètres testés.
- `plot_combined_heatmaps.py` accepte `--prefix` pour intégrer le nom du couple d'hyperparamètres dans les noms de fichiers de sortie.

Créer un rapport PDF (optionnel)

Si ImageMagick est installé, tu peux convertir les images en PDF :

```bash
convert plots/lr_vs_lrdecay_accuracy_combined.png plots/lr_vs_lrdecay_loss_combined.png plots/lr_vs_lrdecay_train_time_s_combined.png combined_report_lr_vs_lrdecay.pdf
```

Extensions possibles (je peux faire si tu veux)
- ajouter `requirements.txt` ;
- modifier `plot_combined_heatmaps.py` pour accepter un CSV unique avec une colonne `model` et construire directement les figures ;
- générer automatiquement un PDF/HTML à la fin de chaque script.

---

README mis à jour. Dis-moi quelle option tu veux ensuite (ex: rendre tous les scripts exécutables, exécuter une des nouvelles grilles, ou générer automatiquement un PDF à la fin de chaque run).

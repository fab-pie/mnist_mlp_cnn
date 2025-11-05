# Explication détaillée de `mnist_mlp.py`

Ce document explique le fichier `mnist_mlp.py` ligne par ligne. L'objectif est de rendre chaque portion de code claire pour une lecture pédagogique : quelles sont les responsabilités des blocs, pourquoi certaines fonctions sont appelées, et quels effets attendus sur l'entraînement et l'export du modèle.

Fichier analysé : `mnist_mlp.py`

---

1. from enum import Enum
   - Importe la classe `Enum` du module standard `enum`. Elle est utilisée pour définir un type énuméré (ici `SamplingMod`) pour les options de remontage géométrique.

2. from pathlib import Path
   - `Path` sert à manipuler des chemins de fichiers de façon portable.

3. from typing import Callable
   - Import de type utilisé pour les annotations (ici pour indiquer qu'une liste contient des callables qui prennent un Tensor et renvoient un Tensor).

4. from tinygrad import Tensor, TinyJit, nn
   - Import principal de tinygrad : `Tensor` (tensors et opérations), `TinyJit` (décorateur pour JIT), `nn` (module réseau dans tinygrad).

5. from tinygrad.device import Device
   - Permet de sélectionner la back-end (par ex. WEBGPU) à la fin pour l'export.

6. from tinygrad.helpers import getenv, trange
   - `getenv`: lit des variables d'environnement de façon robuste (avec valeur par défaut).
   - `trange`: itérateur/affichage de progression similaire à `tqdm`.

7. from tinygrad.nn.datasets import mnist
   - Fonction qui charge/donne les datasets MNIST (X_train, Y_train, X_test, Y_test).

8. from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
   - Helpers pour sauvegarder/charger l'état des paramètres et serialiser des poids en `.safetensors`.

9. from export_model import export_model
   - Fonction locale (fichier `export_model.py`) pour exporter le modèle au format JS/WebGPU.

10. import math
    - Module math standard pour constantes et fonctions trigonométriques.

---

11. class SamplingMod(Enum):
12.   BILINEAR = 0
13.   NEAREST = 1
    - Énumération définissant deux modes d'échantillonnage pour la transformation géométrique : bilinéaire ou nearest (plus rapide mais moins lisse).

---

14. def geometric_transform(X: Tensor, angle_deg: Tensor, scale: Tensor, shift_x: Tensor, shift_y: Tensor, sampling: SamplingMod) -> Tensor:
    - Fonction qui applique une transformation géométrique (rotation + échelle + translation) à un batch d'images `X`.
    - Entrées :
      - `X`: Tensor de forme (B, C, H, W)
      - `angle_deg`: Tensor de taille B contenant angles en degrés
      - `scale`, `shift_x`, `shift_y`: Tensor de taille B contenant paramètres par image
      - `sampling`: méthode d'échantillonnage (BILINEAR ou NEAREST)

15.     B, C, H, W = X.shape
    - Récupère dimensions batch, channels, hauteur et largeur.

16.     angle = angle_deg * math.pi / 180.0
17.     cos_a, sin_a = Tensor.cos(angle), Tensor.sin(angle)
    - Convertit degrés en radians et calcule cos/sin pour chaque élément du batch (Tensor-aware).

18.     R11, R12, T13 = cos_a * scale, -sin_a * scale, shift_x
19.     R21, R22, T23 = sin_a * scale, cos_a * scale, shift_y
    - Calcule les composantes de la matrice affine 2x3 (rotation*mise à l'échelle + translation).

20.     row1 = Tensor.cat(R11.reshape(B, 1), R12.reshape(B, 1), T13.reshape(B, 1), dim=1).reshape(B, 1, 3)
21.     row2 = Tensor.cat(R21.reshape(B, 1), R22.reshape(B, 1), T23.reshape(B, 1), dim=1).reshape(B, 1, 3)
22.     row3 = Tensor([[0.0, 0.0, 1.0]]).expand(B, 1, 3)
23.     affine_matrix = Tensor.cat(row1, row2, row3, dim=1)
    - Construit la matrice affine 3x3 par batch (en coordonnées homogènes) :
      [[R11, R12, T13],
       [R21, R22, T23],
       [0,   0,    1   ]]  pour chaque élément du batch.

24.     x_idx, y_idx = Tensor.arange(W).float(), Tensor.arange(H).float()
25.     grid_y, grid_x = y_idx.reshape(-1, 1).expand(H, W), x_idx.reshape(1, -1).expand(H, W)
26.     coords = Tensor.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
27.     coords_homo = Tensor.cat(coords, Tensor.ones(H * W, 1), dim=1).reshape(1, H * W, 3).expand(B, H * W, 3)
    - Prépare la grille de coordonnées (tous les points cibles) et la met en forme homogène (x, y, 1) pour la multiplication matricielle par batch.

28.     transformed_coords = coords_homo.matmul(affine_matrix.permute(0, 2, 1))
    - Applique la transformation affine par multiplication matricielle batched ; on obtient des coordonnées sources correspondantes.

29.     match sampling:
      case SamplingMod.NEAREST:
        x_idx = transformed_coords[:, :, 0].round().clip(0, W - 1).int()
        y_idx = transformed_coords[:, :, 1].round().clip(0, H - 1).int()
        return X.reshape(B, C * H * W).gather(1, y_idx * W + x_idx).reshape(B, C, H, W)
    - Cas nearest : arrondir les coordonnées transformées et faire un gather pour récupérer les pixels correspondants.

30.      case SamplingMod.BILINEAR:
        x_prime, y_prime = transformed_coords[:, :, 0],  transformed_coords[:, :, 1]
        x0, y0 = x_prime.floor().int(), y_prime.floor().int()
        dx, dy = x_prime - x0.float(), y_prime - y0.float()
        x1, y1 = x0 + 1, y0 + 1
        x0, y0 = x0.clip(0, W - 1), y0.clip(0, H - 1)
        x1, y1 = x1.clip(0, W - 1), y1.clip(0, H - 1)
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy
        X_flat = X.reshape(B, C * H * W)
        v00 = X_flat.gather(1, y0 * W + x0)
        v10 = X_flat.gather(1, y0 * W + x1)
        v01 = X_flat.gather(1, y1 * W + x0)
        v11 = X_flat.gather(1, y1 * W + x1)
        return ((w00 * v00) + (w10 * v10) + (w01 * v01) + (w11 * v11)).reshape(B, C, H, W)
    - Cas bilinéaire : calcul des quatre voisins, calcul des poids et interpolation bilinéaire. Retourne la grille reconstruite.

---

31. def normalize(X: Tensor) -> Tensor:
32.   return X * 2 / 255 - 1
    - Normalisation simple : convertit intensités 0..255 en range [-1, 1]. Utile pour stabiliser l'entraînement.

---

33. class Model:
34.   def __init__(self):
35.     self.layers: list[Callable[[Tensor], Tensor]] = [
36.       lambda x: x.flatten(1),
37.       nn.Linear(784, 512), Tensor.silu,
38.       nn.Linear(512, 512), Tensor.silu,
39.       nn.Linear(512, 10),
40.     ]
41.
42.   def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)
    - Définition d'un MLP simple : entrée aplatie (784), deux couches cachées 512 avec activation SiLU, puis sortie 10 logits.
    - `x.sequential(self.layers)` applique successivement les layers définis.

---

43. if __name__ == "__main__":
    - Point d'entrée script : lecture des variables d'environnement, préparation du dataset et boucle d'entraînement.

44.   B = int(getenv("BATCH", 512))
45.   LR = float(getenv("LR", 0.02))
46.   LR_DECAY = float(getenv("LR_DECAY", 0.9))
47.   PATIENCE = float(getenv("PATIENCE", 50))
    - Lecture des hyperparamètres via variables d'environnement avec valeurs par défaut. `PATIENCE` contrôlera la décroissance du LR.

48.   ANGLE = float(getenv("ANGLE", 15))
49.   SCALE = float(getenv("SCALE", 0.1))
50.   SHIFT = float(getenv("SHIFT", 0.1))
51.   SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))
    - Paramètres utilisés pour la transformation géométrique aléatoire des images (data augmentation).

52.   model_name = Path(__file__).name.split('.')[0]
53.   dir_name = Path(__file__).parent / model_name
54.   dir_name.mkdir(exist_ok=True)
    - Prépare un dossier `mnist_mlp/` où seront sauvegardés les poids et les artefacts exportés.

55.   X_train, Y_train, X_test, Y_test = mnist()
    - Charge MNIST (méthode fournie par tinygrad) ; X sont des images et Y sont labels.

56.   model = Model()
57.   opt = nn.optim.Muon(nn.state.get_parameters(model))
    - Instancie le modèle et un optimiseur (ici `Muon` dans tinygrad). `get_parameters` récupère les paramètres du modèle.

58.   @TinyJit
59.   @Tensor.train()
60.   def train_step() -> Tensor:
    - Déclare la fonction d'entraînement pour un pas ; décorée par TinyJit pour accélération et `Tensor.train()` pour comportement train.

61.     samples = Tensor.randint(B, high=int(X_train.shape[0]))
62.     angle_deg = (Tensor.rand(B) * 2 * ANGLE - ANGLE)
63.     scale = 1.0 + (Tensor.rand(B) * 2 * SCALE - SCALE)
64.     shift_x = (Tensor.rand(B) * 2 * SHIFT - SHIFT)
65.     shift_y = (Tensor.rand(B) * 2 * SHIFT - SHIFT)
    - Prépare un batch aléatoire d'indices et des paramètres d'augmentation par image : angle, scale, shifts.

66.     opt.zero_grad()
67.     input = normalize(geometric_transform(X_train[samples], angle_deg, scale, shift_x, shift_y, SAMPLING))
68.     loss = model(input).sparse_categorical_crossentropy(Y_train[samples]).backward()
69.     return loss.realize(*opt.schedule_step())
    - Calcule perte sur le batch après augmentation, rétroprop et met à jour via l'optimiseur. `schedule_step` applique l'étape d'optimisation et retourne éventuellement le lr.

---

70.   @TinyJit
71.   def get_test_acc() -> Tensor: return (model(normalize(X_test)).argmax(axis=1) == Y_test).mean() * 100
    - Fonction JIT pour évaluer l'accuracy sur l'ensemble test (en %).

72.   test_acc, best_acc, best_since = float('nan'), 0, 0
73.   for i in (t:=trange(getenv("STEPS", 70))):
74.     loss = train_step()
    - Boucle d'entraînement principale ; affiche progression via `trange`.

75.     if (i % 10 == 9) and (test_acc := get_test_acc().item()) > best_acc:
76.       best_since = 0
77.       best_acc = test_acc
78.       state_dict = get_state_dict(model)
79.       safe_save(state_dict, dir_name / f"{model_name}.safetensors")
80.       continue
    - Tous les 10 itérations, on évalue sur test ; si on améliore, on sauvegarde les poids (early snapshot) et on réinitialise `best_since`.

81.     if (best_since := best_since + 1) % PATIENCE == PATIENCE - 1:
82.       best_since = 0
83.       opt.lr *= LR_DECAY
84.       state_dict = safe_load(dir_name / f"{model_name}.safetensors")
85.       load_state_dict(model, state_dict)
    - Si aucune amélioration pendant `PATIENCE` évaluations, on réduit le learning rate (multiplié par LR_DECAY) et on recharge le meilleur modèle sauvegardé.

86.     t.set_description(f"lr: {opt.lr.item():2.2e}  loss: {loss.item():2.2f}  accuracy: {best_acc:2.2f}%")
    - Met à jour la description de la barre de progression avec lr courant, perte et meilleure accuracy.

---

87.   Device.DEFAULT = "WEBGPU"
88.   model = Model()
89.   state_dict = safe_load(dir_name / f"{model_name}.safetensors")
90.   load_state_dict(model, state_dict)
91.   input = Tensor.randn(1, 1, 28, 28)
92.   prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)
93.   safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")
94.   with open(dir_name / f"{model_name}.js", "w") as text_file: text_file.write(prg)
    - Après l'entraînement, on force l'export du modèle :
      - on choisit `Device.DEFAULT = "WEBGPU"` pour produire une export compatible WebGPU
      - on recrée une instance `Model`, on charge le meilleur état sauvegardé
      - on appelle `export_model` pour générer le code JS/WebGPU et l'état serialisé
      - on sauvegarde le `.webgpu.safetensors` et le `.js` final dans le répertoire du modèle

---

Notes générales et conseils
- Les transformations géométriques sont faites par batch et appliquées en tant que data augmentation (utile pour robustesse). Quand `SAMPLING` = `NEAREST`, c'est plus rapide mais moins lisse ; `BILINEAR` est préférable pour des images.
- Le schéma de sauvegarde (snapshot tous les 10 pas) + reload après décroissance du LR est utile pour éviter de continuer à partir d'un état dégradé.
- Le code exporte le modèle entraîné pour l'utiliser côté client (WebGPU) — c'est le workflow complet : entraînement -> sauvegarde -> export.

Si vous voulez, je peux produire une version du même document avec :
- annotations insérées directement dans une copie du script (commentaires inline),
- une version simplifiée pour débutants, ou
- un notebook explicatif montrant quelques pas d'entraînement et visualisations.

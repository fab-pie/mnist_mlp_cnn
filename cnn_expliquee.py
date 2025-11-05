# --- Importation des librairies nécessaires ---

from enum import Enum                   # Pour définir des énumérations (ici, le mode d’échantillonnage)
from pathlib import Path                # Pour gérer les chemins de fichiers/dossiers facilement
from typing import Callable             # Pour typer les fonctions dans la classe Model
from tinygrad import Tensor, TinyJit, nn # Tensor = objet principal, TinyJit = compilation JIT, nn = couches de réseau
from tinygrad.device import Device       # Pour choisir sur quel device exécuter le modèle (CPU, GPU, WebGPU, etc.)
from tinygrad.helpers import getenv, trange  # getenv = récupérer des variables d’environnement, trange = barre de progression
from tinygrad.nn.datasets import mnist       # Import du dataset MNIST (images de chiffres manuscrits)
from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load, safe_save
# Fonctions pour sauvegarder / recharger les poids du modèle
from export_model import export_model        # Fonction personnalisée pour exporter le modèle (probablement vers WebGPU/JS)

import math  # Librairie standard pour les calculs trigonométriques

# --- Définition d'une énumération pour le type de sampling (échantillonnage) ---

class SamplingMod(Enum):
  BILINEAR = 0  # Interpolation bilinéaire (plus lisse)
  NEAREST = 1   # Interpolation par plus proche voisin (plus rapide mais moins précis)

# --- Fonction de transformation géométrique appliquée aux images (rotation, zoom, translation) ---

def geometric_transform(X: Tensor, angle_deg: Tensor, scale: Tensor, shift_x: Tensor, shift_y: Tensor, sampling: SamplingMod) -> Tensor:
    B, C, H, W = X.shape  # B=Batch size, C=canaux (1 pour MNIST), H=hauteur, W=largeur

    # Conversion des angles de degrés en radians
    angle = angle_deg * math.pi / 180.0
    cos_a, sin_a = Tensor.cos(angle), Tensor.sin(angle)

    # Matrice de rotation + échelle + translation (affine)
    R11, R12, T13 = cos_a * scale, -sin_a * scale, shift_x
    R21, R22, T23 = sin_a * scale, cos_a * scale, shift_y

    # Construction de la matrice 3x3 pour la transformation affine
    row1 = Tensor.cat(R11.reshape(B, 1), R12.reshape(B, 1), T13.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row2 = Tensor.cat(R21.reshape(B, 1), R22.reshape(B, 1), T23.reshape(B, 1), dim=1).reshape(B, 1, 3)
    row3 = Tensor([[0.0, 0.0, 1.0]]).expand(B, 1, 3)
    affine_matrix = Tensor.cat(row1, row2, row3, dim=1)

    # Création de la grille de coordonnées (x,y) des pixels
    x_idx, y_idx = Tensor.arange(W).float(), Tensor.arange(H).float()
    grid_y, grid_x = y_idx.reshape(-1, 1).expand(H, W), x_idx.reshape(1, -1).expand(H, W)
    coords = Tensor.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    coords_homo = Tensor.cat(coords, Tensor.ones(H * W, 1), dim=1).reshape(1, H * W, 3).expand(B, H * W, 3)

    # Application de la matrice de transformation
    transformed_coords = coords_homo.matmul(affine_matrix.permute(0, 2, 1))

    # --- Cas 1 : interpolation par plus proche voisin ---
    match sampling:
      case SamplingMod.NEAREST:
        x_idx = transformed_coords[:, :, 0].round().clip(0, W - 1).int()
        y_idx = transformed_coords[:, :, 1].round().clip(0, H - 1).int()
        # On prend les pixels les plus proches dans X
        return X.reshape(B, C * H * W).gather(1, y_idx * W + x_idx).reshape(B, C, H, W)

      # --- Cas 2 : interpolation bilinéaire ---
      case SamplingMod.BILINEAR:
        x_prime, y_prime = transformed_coords[:, :, 0],  transformed_coords[:, :, 1]
        x0, y0 = x_prime.floor().int(), y_prime.floor().int()
        dx, dy = x_prime - x0.float(), y_prime - y0.float()

        # Coordonnées des pixels voisins
        x1, y1 = x0 + 1, y0 + 1
        x0, y0 = x0.clip(0, W - 1), y0.clip(0, H - 1)
        x1, y1 = x1.clip(0, W - 1), y1.clip(0, H - 1)

        # Poids d’interpolation
        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        X_flat = X.reshape(B, C * H * W)
        v00 = X_flat.gather(1, y0 * W + x0)
        v10 = X_flat.gather(1, y0 * W + x1)
        v01 = X_flat.gather(1, y1 * W + x0)
        v11 = X_flat.gather(1, y1 * W + x1)

        # Combinaison pondérée des 4 voisins
        return ((w00 * v00) + (w10 * v10) + (w01 * v01) + (w11 * v11)).reshape(B, C, H, W)


# --- Normalisation des pixels dans l’intervalle [-1, 1] ---
def normalize(X: Tensor) -> Tensor:
  return X * 2 / 255 - 1


# --- Définition du modèle CNN ---

class Model:
  def __init__(self):
    # Séquence de couches : convolution → activation → batchnorm → maxpool → linear
    self.layers: list[Callable[[Tensor], Tensor]] = [
      nn.Conv2d(1, 32, 5), Tensor.silu,            # 1 canal -> 32 filtres, activation SiLU
      nn.Conv2d(32, 32, 5), Tensor.silu,           # Deuxième conv
      nn.BatchNorm(32), Tensor.max_pool2d,         # Normalisation + pooling
      nn.Conv2d(32, 64, 3), Tensor.silu,           # Troisième conv
      nn.Conv2d(64, 64, 3), Tensor.silu,           # Quatrième conv
      nn.BatchNorm(64), Tensor.max_pool2d,         # Normalisation + pooling
      lambda x: x.flatten(1), nn.Linear(576, 10),  # Passage en fully-connected (10 classes)
    ]

  def __call__(self, x:Tensor) -> Tensor:
    # Applique chaque couche à la suite (sequential)
    return x.sequential(self.layers)


# --- Partie exécutable du script ---

if __name__ == "__main__":
  # Chargement des hyperparamètres depuis les variables d’environnement
  B = int(getenv("BATCH", 512))
  LR = float(getenv("LR", 0.02))
  LR_DECAY = float(getenv("LR_DECAY", 0.9))
  PATIENCE = float(getenv("PATIENCE", 50))

  # Paramètres d’augmentation de données
  ANGLE = float(getenv("ANGLE", 15))
  SCALE = float(getenv("SCALE", 0.1))
  SHIFT = float(getenv("SHIFT", 0.1))
  SAMPLING = SamplingMod(getenv("SAMPLING", SamplingMod.NEAREST.value))

  # Préparation des répertoires de sauvegarde
  model_name = Path(__file__).name.split('.')[0]
  dir_name = Path(__file__).parent / model_name
  dir_name.mkdir(exist_ok=True)

  # Chargement du dataset MNIST
  X_train, Y_train, X_test, Y_test = mnist()

  # Initialisation du modèle et de l’optimiseur
  model = Model()
  opt = nn.optim.Muon(nn.state.get_parameters(model))

  # --- Étape d’entraînement compilée (JIT) ---
  @TinyJit
  @Tensor.train()
  def train_step() -> Tensor:
    samples = Tensor.randint(B, high=int(X_train.shape[0]))  # Batch aléatoire
    angle_deg = (Tensor.rand(B) * 2 * ANGLE - ANGLE)
    scale = 1.0 + (Tensor.rand(B) * 2 * SCALE - SCALE)
    shift_x = (Tensor.rand(B) * 2 * SHIFT - SHIFT)
    shift_y = (Tensor.rand(B) * 2 * SHIFT - SHIFT)

    opt.zero_grad()  # Réinitialise les gradients
    input = normalize(geometric_transform(X_train[samples], angle_deg, scale, shift_x, shift_y, SAMPLING))
    loss = model(input).sparse_categorical_crossentropy(Y_train[samples]).backward()
    return loss.realize(*opt.schedule_step())

  # --- Fonction de test (calcul de l’exactitude sur les données de test) ---
  @TinyJit
  def get_test_acc() -> Tensor:
    return (model(normalize(X_test)).argmax(axis=1) == Y_test).mean() * 100

  # --- Boucle principale d'entraînement ---
  test_acc, best_acc, best_since = float('nan'), 0, 0
  for i in (t:=trange(getenv("STEPS", 70))):  # 70 itérations par défaut
    loss = train_step()

    # Tous les 10 pas, si la précision s’améliore → sauvegarde des poids
    if (i % 10 == 9) and (test_acc := get_test_acc().item()) > best_acc:
      best_since = 0
      best_acc = test_acc
      state_dict = get_state_dict(model)
      safe_save(state_dict, dir_name / f"{model_name}.safetensors")
      continue

    # Si aucune amélioration depuis "PATIENCE" itérations → on réduit le LR
    if (best_since := best_since + 1) % PATIENCE == PATIENCE - 1:
      best_since = 0
      opt.lr *= LR_DECAY
      state_dict = safe_load(dir_name / f"{model_name}.safetensors")
      load_state_dict(model, state_dict)

    # Affichage dynamique de la progression
    t.set_description(f"lr: {opt.lr.item():2.2e}  loss: {loss.item():2.2f}  accuracy: {best_acc:2.2f}%")

  # --- Exportation du modèle pour WebGPU ---
  Device.DEFAULT = "WEBGPU"  # Change le périphérique d'exécution
  model = Model()
  state_dict = safe_load(dir_name / f"{model_name}.safetensors")
  load_state_dict(model, state_dict)
  input = Tensor.randn(1, 1, 28, 28)
  prg, *_, state = export_model(model, Device.DEFAULT.lower(), input, model_name=model_name)

  # Sauvegarde des poids et du programme WebGPU
  safe_save(state, dir_name / f"{model_name}.webgpu.safetensors")
  with open(dir_name / f"{model_name}.js", "w") as text_file:
    text_file.write(prg)

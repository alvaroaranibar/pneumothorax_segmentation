# Pneumothorax Segmentation Pipeline – Single-stage at 512×512
# -----------------------------------------------------------------------------
# - Binary segmentation (pneumothorax) with U-Net (EfficientNetB4 encoder)
# - Pretrained ImageNet encoder (transfer learning) via `segmentation_models`
# - Loss = DiceLoss + BinaryCELoss (equal weights)
# - Optimizer = Adam with CosineDecay LR schedule (1e-4 -> 1e-6)
# - Single-stage training ONLY at 512×512 (originals 1024×1024 are resized on-the-fly)
# - Albumentations augmentations (exact recipe provided by user)
# - tf.data streaming from disk (no loading full dataset in RAM)
# - Optional automatic threshold search (binary threshold + small-component removal)
# - Metrics & plots: IoU/Loss curves, confusion matrix, Accuracy/Recall/Precision/F1
#
# Requirements (pip):
#   pip install -U tensorflow segmentation-models albumentations opencv-python scikit-learn matplotlib
#
# Folder layout (matching filenames image<->mask):
#   png_images/xxxx.png   (RGB-like, uint8 [0..255])
#   png_masks/xxxx.png    (grayscale, values {0,255})
# -----------------------------------------------------------------------------

import os, glob, random, math, json
from typing import List, Tuple

import numpy as np
import tensorflow as tf
# Forcing to import the newest keras
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import segmentation_models as sm
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
from tensorflow.keras import mixed_precision

# ------------------------------
# CONFIG
# ------------------------------
mixed_precision.set_global_policy('mixed_float16')
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMAGES_DIR = 'png_images'   # input images (1024x1024, uint8 [0..255])
MASKS_DIR  = 'png_masks'    # masks        (1024x1024, values {0,255})

BACKBONE = 'efficientnetb4'
EPOCHS = 300       #Prev: 80
BATCH_SIZE = 32   #Prev: 4
IMG_SIZE = 512  # single-stage resolution

LR_MAX = 1e-4
LR_MIN = 1e-6

# Thresholds for post-processing
FALLBACK_BINARY_THR = 0.21
FALLBACK_REMOVE_AREA = 2048

AUTO_SEARCH_THRESHOLDS = True           # set False to use fallback values directly
AUTO_SEARCH_MAX_SAMPLES = 256           # number of validation images for threshold sweep
BINARY_THR_GRID = np.linspace(0.05, 0.5, 10)
REMOVE_AREA_GRID = [0, 256, 512, 1024, 2048, 4096]

VAL_SPLIT = 0.15

# ------------------------------
# FILE PAIRING & SPLIT
# ------------------------------

def list_pairs(images_dir: str, masks_dir: str):
    imgs = sorted(glob.glob(os.path.join(images_dir, '*.png')))
    msks = []
    for p in imgs:
        mp = os.path.join(masks_dir, os.path.basename(p))
        if not os.path.exists(mp):
            raise FileNotFoundError(f"Mask not found for image: {p} -> {mp}")
        msks.append(mp)
    return imgs, msks

all_imgs, all_msks = list_pairs(IMAGES_DIR, MASKS_DIR)
print(f"Total pairs: {len(all_imgs)}")

train_i, val_i, train_m, val_m = train_test_split(
    all_imgs, all_msks, test_size=VAL_SPLIT, random_state=SEED, shuffle=True)
print(f"Train: {len(train_i)} | Val: {len(val_i)}")

# ------------------------------
# ALBUMENTATIONS (512x512)
# ------------------------------

train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.OneOf([
        A.RandomBrightnessContrast(p=0.3),
        A.RandomGamma(p=0.3),
    ], p=0.3),

    A.OneOf([
        A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.3),
        A.GridDistortion(p=0.3),
        A.OpticalDistortion(distort_limit=2, p=0.3),
    ], p=0.3),

    A.Affine(
        translate_percent={"x": 0.2, "y": 0.2},
        scale=(0.8, 1.2),
        rotate=(-20, 20),
        border_mode=cv2.BORDER_CONSTANT,
        p=0.5
    ),

    A.Resize(IMG_SIZE, IMG_SIZE),
], p=1)

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])

# ------------------------------
# TF.DATA LOADER (albumentations via tf.py_function)
# ------------------------------

def _load_and_augment(img_path_b: bytes, msk_path_b: bytes, augment: bool):
    img_path = img_path_b.numpy().decode('utf-8')
    msk_path = msk_path_b.numpy().decode('utf-8')

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)          # BGR uint8
    msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)       # uint8
    if img is None or msk is None:
        raise ValueError(f"Failed to read: {img_path} or {msk_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # Albumentations expects RGB
    tr = train_transform if augment else val_transform
    out = tr(image=img, mask=msk)
    img_t, msk_t = out['image'], out['mask']

    img_t = img_t.astype(np.float32) / 255.0               # [0,1]
    msk_t = (msk_t > 127).astype(np.float32)[..., None]    # {0,1} with channel
    return img_t, msk_t


def make_dataset(imgs: List[str], msks: List[str], batch: int, augment: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((imgs, msks))
    if augment:
        ds = ds.shuffle(buffer_size=len(imgs), seed=SEED, reshuffle_each_iteration=True)

    def _map(i, m):
        img, msk = tf.py_function(func=lambda ii, mm: _load_and_augment(ii, mm, augment),
                                   inp=[i, m], Tout=[tf.float32, tf.float32])
        img.set_shape([IMG_SIZE, IMG_SIZE, 3])
        msk.set_shape([IMG_SIZE, IMG_SIZE, 1])
        return img, msk

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

train_ds = make_dataset(train_i, train_m, BATCH_SIZE, augment=True)
val_ds   = make_dataset(val_i,   val_m,   BATCH_SIZE, augment=False)

# ------------------------------
# MODEL & LOSS (segmentation_models)
# ------------------------------

sm.set_framework('tf.keras')
keras.backend.set_image_data_format('channels_last')

loss = sm.losses.DiceLoss() + sm.losses.BinaryCELoss()

# === Metrics Casted for mixed precision ===
def iou_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return sm.metrics.iou_score(y_true, y_pred)

def f1_metric(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return sm.metrics.f1_score(y_true, y_pred)

metrics = [iou_metric, f1_metric]

steps_per_epoch = math.ceil(len(train_i) / BATCH_SIZE)

def make_optimizer(total_steps: int):
    # CosineDecay with floor LR_MIN: lr = ((1-alpha)*cosine + alpha) * LR_MAX, where alpha = LR_MIN/LR_MAX
    alpha = LR_MIN / LR_MAX
    schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LR_MAX,
        decay_steps=total_steps,
        alpha=alpha,
        name='cosine_decay'
    )
    return tf.keras.optimizers.Adam(learning_rate=schedule)

model = sm.Unet(
    backbone_name=BACKBONE,
    encoder_weights='imagenet',
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    classes=1,
    activation='sigmoid',
)

optimizer = make_optimizer(total_steps=EPOCHS * steps_per_epoch)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# ------------------------------
# TRAIN
# ------------------------------

ckpt_path = 'best_512.h5'
callbacks = [
    keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_best_only=True,
        monitor='val_iou_score',  # guarda el mejor modelo según IoU
        mode='max',
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_iou_score',   # vigilamos IoU en validación
        mode='max',                # porque queremos maximizar
        patience=20,               # nº de épocas sin mejora antes de parar
        restore_best_weights=True, # vuelve al mejor modelo automáticamente
        verbose=1
    ),
]


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    verbose=1,
    callbacks=callbacks,
)

# Load best weights
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)

# ------------------------------
# POST-PROCESS: small component removal
# ------------------------------

def remove_small_components(mask_bin: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area (expects HxW binary {0,1})."""
    m = (mask_bin > 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros_like(m)
    for label in range(1, num_labels):  # skip background
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == label] = 1
    return out

# ------------------------------
# VALIDATION PREDICTIONS (probabilities)
# ------------------------------

all_probs, all_gts = [], []
for batch_imgs, batch_msks in val_ds:
    probs = model.predict(batch_imgs, verbose=0)
    all_probs.append(probs)
    all_gts.append(batch_msks.numpy())

probs = np.concatenate(all_probs, axis=0)
gts   = np.concatenate(all_gts, axis=0)

# ------------------------------
# THRESHOLD SWEEP (optional)
# ------------------------------
if AUTO_SEARCH_THRESHOLDS:
    rng = np.random.RandomState(SEED)
    idx = rng.choice(len(probs), size=min(AUTO_SEARCH_MAX_SAMPLES, len(probs)), replace=False)
    P = probs[idx]
    G = gts[idx]

    best_iou = -1.0
    best_thr = FALLBACK_BINARY_THR
    best_area = FALLBACK_REMOVE_AREA

    for thr in BINARY_THR_GRID:
        bin_pred = (P >= thr).astype(np.uint8)
        for area in REMOVE_AREA_GRID:
            refined = np.stack([remove_small_components(b[...,0], area) for b in bin_pred], axis=0)
            iou = sm.metrics.iou_score(
                tf.convert_to_tensor(G),
                tf.convert_to_tensor(refined[..., None].astype(np.float32))
            ).numpy()
            if iou > best_iou:
                best_iou, best_thr, best_area = float(iou), float(thr), int(area)
    print(f"[AUTO] Selected thresholds -> binary_thr={best_thr:.4f}, remove_area={best_area} (IoU={best_iou:.4f})")
else:
    best_thr = FALLBACK_BINARY_THR
    best_area = FALLBACK_REMOVE_AREA
    print(f"[FALLBACK] Using thresholds -> binary_thr={best_thr:.4f}, remove_area={best_area}")

# Apply thresholds to full validation set
bin_pred_full = (probs >= best_thr).astype(np.uint8)
refined_full = np.stack([remove_small_components(b[...,0], best_area) for b in bin_pred_full], axis=0)[..., None]

# ------------------------------
# METRICS (Accuracy, Recall, Precision, F1) + Confusion Matrix
# ------------------------------

# Flatten
y_true = gts.astype(np.uint8).reshape(-1)
y_pred = refined_full.astype(np.uint8).reshape(-1)

cm = confusion_matrix(y_true, y_pred, labels=[0,1])
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)

print("Confusion matrix (rows=gt, cols=pred):\n", cm)
print(f"Accuracy:  {acc*100:.2f}%")
print(f"Recall:    {rec*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"F1-score:  {f1*100:.2f}%")

# ------------------------------
# PLOTS: Loss & IoU curves, Confusion matrix
# ------------------------------

plt.figure(); plt.plot(history.history['loss'], label='train'); plt.plot(history.history['val_loss'], label='val');
plt.title('Loss (Dice+BCE)'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend(); plt.grid(True)

plt.figure(); plt.plot(history.history['iou_score'], label='train'); plt.plot(history.history['val_iou_score'], label='val');
plt.title('IoU'); plt.xlabel('Epoch'); plt.ylabel('IoU'); plt.legend(); plt.grid(True)

plt.figure();
plt.imshow(cm, interpolation='nearest'); plt.title('Confusion Matrix'); plt.colorbar();
plt.xticks([0,1], ['BG','PTX']); plt.yticks([0,1], ['BG','PTX']);
th = cm.max()/2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha='center', va='center', color='white' if cm[i, j] > th else 'black')
plt.ylabel('True label'); plt.xlabel('Predicted label'); plt.tight_layout()

plt.show()

# ------------------------------
# SAVE thresholds used (for reproducibility)
# ------------------------------
with open('thresholds.json', 'w') as f:
    json.dump({'binary_threshold': float(best_thr), 'removal_area': int(best_area)}, f, indent=2)
print('Saved thresholds.json')

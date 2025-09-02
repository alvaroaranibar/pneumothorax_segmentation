import os, glob, random
import numpy as np
import tensorflow as tf
# Forcing to import the newest keras
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import segmentation_models as sm

# ------------------------------
# CONFIG
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = 512
BINARY_THR = 0.05
REMOVE_AREA = 0
DATASET_DIR = "shangai_dataset_PTX-498-v2-fix"
MODEL_PATH = "best_512.h5"

# ------------------------------
# LOAD MODEL
# ------------------------------
sm.set_framework("tf.keras")
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={
        "iou_score": sm.metrics.iou_score,
        "f1-score": sm.metrics.f1_score,
        "dice_loss": sm.losses.DiceLoss(),
        "binary_crossentropy": sm.losses.BinaryCELoss()
    }
)
print("✅ Modelo cargado:", MODEL_PATH)

# ------------------------------
# DATA LOADER
# ------------------------------
def get_all_pairs(dataset_dir):
    imgs, msks = [], []
    for site in ["SiteA", "SiteB", "SiteC"]:
        site_path = os.path.join(dataset_dir, site)
        img_files = sorted(glob.glob(os.path.join(site_path, "*.1.img.png")))
        for img_f in img_files:
            base = os.path.basename(img_f).split(".")[0]  # e.g. "1"
            mask_f = os.path.join(site_path, f"{base}.2.mask.png")
            if os.path.exists(mask_f):
                imgs.append(img_f)
                msks.append(mask_f)
    return imgs, msks

all_imgs, all_msks = get_all_pairs(DATASET_DIR)
print(f"Total imágenes de test: {len(all_imgs)}")

def preprocess_image(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    mask = (mask > 127).astype(np.uint8)

    return img, mask

# ------------------------------
# PREDICT & METRICS
# ------------------------------
y_true, y_pred = [], []

for img_path, mask_path in zip(all_imgs, all_msks):
    img, mask = preprocess_image(img_path, mask_path)
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0,...,0]
    
    # threshold
    pred_bin = (pred >= BINARY_THR).astype(np.uint8)
    # small component removal (optional, here REMOVE_AREA=0)
    if REMOVE_AREA > 0:
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(pred_bin, connectivity=8)
        filtered = np.zeros_like(pred_bin)
        for label in range(1, num_labels):
            if stats[label, cv2.CC_STAT_AREA] >= REMOVE_AREA:
                filtered[labels == label] = 1
        pred_bin = filtered

    y_true.append(mask.flatten())
    y_pred.append(pred_bin.flatten())

y_true = np.concatenate(y_true)
y_pred = np.concatenate(y_pred)

# Confusion matrix & metrics
cm = confusion_matrix(y_true, y_pred, labels=[0,1])
acc = accuracy_score(y_true, y_pred)
prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

# IoU manual
intersection = np.logical_and(y_true == 1, y_pred == 1).sum()
union = np.logical_or(y_true == 1, y_pred == 1).sum()
iou = intersection / union if union > 0 else 0

print("\n📊 Resultados finales:")
print("Confusion matrix (rows=gt, cols=pred):\n", cm)
print(f"Accuracy: {acc*100:.2f}%")
print(f"Recall: {rec*100:.2f}%")
print(f"Precision: {prec*100:.2f}%")
print(f"F1-score: {f1*100:.2f}%")
print(f"IoU: {iou*100:.2f}%")

import itertools

def plot_confusion_matrix(cm, classes, normalize=False, title="Confusion Matrix", cmap=plt.cm.Blues):
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}",
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig("confusion_shangai.png", dpi=300)
    plt.close()

# Call the function
plot_confusion_matrix(cm, classes=["Background (0)", "Object (1)"])
print("📁 Confusion matrix saved as 'confusion_shangai.png'")
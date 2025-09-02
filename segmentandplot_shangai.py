import os, glob, random
import numpy as np
import tensorflow as tf
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
import cv2
import matplotlib.pyplot as plt
import segmentation_models as sm
import matplotlib.patches as mpatches

# ------------------------------
# CONFIG
# ------------------------------
SEED = 70
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
print("✅ Model loaded:", MODEL_PATH)

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
print(f"Total test images: {len(all_imgs)}")

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
# PLOT SAMPLE PREDICTIONS (6 images)
# ------------------------------
idxs = random.sample(range(len(all_imgs)), 6)

plt.figure(figsize=(18, 12))
for i, idx in enumerate(idxs):
    img, mask = preprocess_image(all_imgs[idx], all_msks[idx])
    pred = model.predict(np.expand_dims(img, axis=0), verbose=0)[0, ..., 0]
    pred_bin = (pred >= BINARY_THR).astype(np.uint8)

    # Create overlay image
    overlay = img.copy()
    overlay = (overlay * 255).astype(np.uint8)

    # Green = Ground Truth, Red = Prediction
    overlay[mask == 1] = [0, 255, 0]       # GT in green
    overlay[pred_bin == 1] = [255, 0, 0]   # Prediction in red

    # Where both overlap, set to Yellow
    overlap = np.logical_and(mask == 1, pred_bin == 1)
    overlay[overlap] = [255, 255, 0]

    plt.subplot(2, 3, i+1)
    plt.imshow(overlay)
    plt.title(f"Overlay: GT vs Prediction ({os.path.basename(all_imgs[idx])})")
    plt.axis("off")

# Legend
gt_patch = mpatches.Patch(color='green', label='Ground Truth')
pred_patch = mpatches.Patch(color='red', label='Prediction')
overlap_patch = mpatches.Patch(color='yellow', label='Overlap')
plt.figlegend(handles=[gt_patch, pred_patch, overlap_patch], loc='upper center', ncol=3, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# ------------------------------
# CONFIG
# ------------------------------
DATASET_DIR = "shangai_dataset_PTX-498-v2-fix"
SITE = "SiteA"   # puedes cambiar a SiteB o SiteC
IMG_NAME = "1.1.img.png"
MSK_NAME = "1.2.mask.png"

# rutas
img_path = os.path.join(DATASET_DIR, SITE, IMG_NAME)
msk_path = os.path.join(DATASET_DIR, SITE, MSK_NAME)

# carga imagen y máscara
img = cv2.imread(img_path, cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
msk = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

# ------------------------------
# AUGMENTATIONS (forzado p=1, pero guardamos prob original)
# ------------------------------
augmentations = [
    (A.HorizontalFlip(p=1.0), "HorizontalFlip", 0.5),
    (A.RandomBrightnessContrast(p=1.0), "RandomBrightnessContrast", 0.3),
    (A.RandomGamma(p=1.0), "RandomGamma", 0.3),
    (A.ElasticTransform(alpha=120, sigma=120*0.05, p=1.0), "ElasticTransform", 0.3),
    (A.GridDistortion(p=1.0), "GridDistortion", 0.3),
    (A.OpticalDistortion(distort_limit=2, p=1.0), "OpticalDistortion", 0.3),
    (A.Affine(translate_percent={"x": 0.2, "y": 0.2},
              scale=(0.8, 1.2), rotate=(-20, 20),
              border_mode=cv2.BORDER_CONSTANT, p=1.0), "Affine", 0.5),
]

# ------------------------------
# FUNCIÓN PARA SUPERPONER MÁSCARA
# ------------------------------
def overlay_mask(image, mask, alpha=0.4, color=(255, 0, 0)):
    """Superpone la máscara en la imagen."""
    overlay = image.copy()
    overlay[mask > 127] = color  # píxeles de la máscara
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

# ------------------------------
# PLOTEO
# ------------------------------
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

# original
axes[0].imshow(overlay_mask(img, msk))
axes[0].set_title("Original")
axes[0].axis("off")

# augmentations
for i, (aug, name, prob) in enumerate(augmentations, start=1):
    out = aug(image=img, mask=msk)
    img_aug, msk_aug = out["image"], out["mask"]
    axes[i].imshow(overlay_mask(img_aug, msk_aug))
    axes[i].set_title(f"{name} (p={prob})")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

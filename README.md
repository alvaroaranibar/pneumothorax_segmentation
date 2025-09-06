# Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer Learning in a U-Net Architecture

Pneumothorax, the abnormal accumulation of air in the pleural space, can be life-threatening if undetected. Chest X-rays are the first-line diagnostic tool, but small cases may be subtle.  

We propose an automated deep-learning pipeline using a **U-Net with an EfficientNet-B4 encoder** to segment pneumothorax regions. Trained on the **SIIM-ACR dataset** with data augmentation and a combined **binary cross-entropy plus Dice loss**, the model achieved:

- **IoU:** 0.7008  
- **Dice score:** 0.8241  

on the independent **PTX-498 dataset**.  

These results demonstrate that the model can accurately localize pneumothoraces and support radiologists.  

👉 [Preprint available](https://doi.org/10.48550/arXiv.2509.03950 )  

## Pneumothorax Segmentation Pipeline – Single-stage at 512×512

The pipeline leverages **transfer learning**, **mixed precision**, and **efficient augmentation strategies** to balance accuracy and training efficiency.  

---

## Key Features
- **Architecture:** U-Net with EfficientNetB4 backbone (ImageNet-pretrained).  
- **Loss Function:** Dice Loss + Binary Cross-Entropy (equal weights).  
- **Optimizer:** Adam with cosine-decay learning rate schedule (1e-4 → 1e-6).  
- **Resolution:** Single-stage training at 512×512 (originals resized on-the-fly).  
- **Augmentation:** Advanced augmentations via Albumentations.  
- **Data Handling:** `tf.data` streaming from disk (no need to load full dataset in memory).  
- **Post-processing:** Optional automatic threshold search (binary threshold + small-component removal).  
- **Evaluation:** IoU/Loss curves, confusion matrix, Accuracy, Recall, Precision, and F1-score.  

---

## Requirements
Install dependencies with:

`bash
pip install -U tensorflow segmentation-models albumentations opencv-python scikit-learn matplotlib

## Dataset

This work uses two datasets:

SIIM-ACR Pneumothorax Segmentation Challenge (2019)

12,047 PNG images (1024×1024, 3 channels) with corresponding masks.

Masks are binary PNG (values {0, 255}).

Split: 85% training, 15% validation.

PTX-498 Dataset (Wang et al.)

498 positive pneumothorax cases collected from 3 hospitals in Shanghai.

Used exclusively for final evaluation.

Folder layout (image ↔ mask filenames must match):

png_images/xxxx.png   # RGB-like images, uint8 [0..255]
png_masks/xxxx.png    # Grayscale masks, values {0,255}

Data Preprocessing & Augmentation

Resizing: All images and masks resized to 512×512.

Train augmentations (Albumentations):

Horizontal flip (p=0.5)

Random brightness/contrast or gamma adjustment

Elastic, grid, or optical distortions

Affine transforms (translation, scaling, rotation)

Validation augmentations: Only resizing.

Training Setup

GPU: NVIDIA RTX 3090 (24 GB VRAM).

Mixed Precision: Enabled (float16 for most ops, float32 for critical ops).

Batch size: 32

Epochs: 300

Early stopping: Patience of 20 epochs (monitored via validation IoU).

Checkpointing: Best model saved based on validation IoU.

Post-processing

Binarization: Pixel-level thresholding (probabilities → binary mask).

Small component removal: Eliminates connected regions below a minimum area.

Automatic threshold search: Grid search on validation set for optimal binary threshold (BT) and removal threshold (RT).

Evaluation Metrics

Internal validation: Monitored during training using IoU and F1-score.

Final evaluation (PTX-498):

Accuracy

Recall

Precision

F1-score (Dice coefficient)

Intersection over Union (IoU)

Confusion Matrix

Results

Training (RTX 3090):

Step time: ~450 ms

Time per epoch: ~2m 24s

Total training time: ~12h 15m

Final metrics on PTX-498:

Accuracy: 98.42%

Recall: 74.18%

Precision: 92.69%

F1-score: 82.41%

IoU: 70.08%

Example Outputs

Loss & IoU curves during training.

Confusion matrix on validation/test sets.

Sample predictions vs ground-truth masks.

Reproducibility

Thresholds are automatically saved to thresholds.json:

{
  "binary_threshold": 0.05,
  "removal_area": 0
}

References

SIIM-ACR Pneumothorax Segmentation Challenge (2019).

Wang et al., PTX-498 dataset.

Segmentation Models (Keras/TensorFlow).

Albumentations image augmentation library.


---

¿Quieres que también genere un **diagrama visual en Mermaid (pipeline)** dentro del README para mostrar gráficamente el flujo (dataset → preprocessing → training → post-processing → evaluation)?


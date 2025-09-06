# Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer Learning in a U-Net Architecture

Pneumothorax, the abnormal accumulation of air in the pleural space, can be life-threatening if undetected. Chest X-rays are the first-line diagnostic tool, but small cases may be subtle.  

We propose an automated deep-learning pipeline using a **U-Net with an EfficientNet-B4 encoder** to segment pneumothorax regions. Trained on the **SIIM-ACR dataset** with data augmentation and a combined **binary cross-entropy plus Dice loss**, the model achieved:

- **IoU:** 0.7008  
- **Dice score:** 0.8241  

on the independent **PTX-498 dataset**.  

These results demonstrate that the model can accurately localize pneumothoraces and support radiologists.  

👉 [Preprint available](https://doi.org/10.48550/arXiv.2509.03950 )  

## 📖 Description
- U-net architecture to preserve high-level context and fine details.
- Transfer learning in an EfficientNetB4 backbone with ImageNet pretrained weights to accelerate convergence.
- 12,047 images from the XSIIM-ACR Pneumothorax Segmentation Challenge dataset, trained on the Runpod cloud platform.
- Cross entropy + Dice loss.

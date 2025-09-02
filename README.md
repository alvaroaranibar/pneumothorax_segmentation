# Chest X-ray Pneumothorax Segmentation Using EfficientNet-B4 Transfer Learning in a U-Net Architecture

## 📖 Descripción
Pneumothorax, the abnormal accumulation of air in the pleural space, can be life-threatening if undetected. Chest X-rays are the first-line diagnostic tool, but small cases may be subtle.  

We propose an automated deep-learning pipeline using a **U-Net with an EfficientNet-B4 encoder** to segment pneumothorax regions. Trained on the **SIIM-ACR dataset** with data augmentation and a combined **binary cross-entropy plus Dice loss**, the model achieved:

- **IoU:** 0.7008  
- **Dice score:** 0.8241  

on the independent **PTX-498 dataset**.  

These results demonstrate that the model can accurately localize pneumothoraces and support radiologists.  

👉 [Read the full paper (PDF)](link-to-your-paper.pdf)  

---

## ⚙️ Dependencias
El proyecto está implementado en **Python 3.9+**. Para instalar las dependencias principales:  

```bash
pip install -r requirements.txt

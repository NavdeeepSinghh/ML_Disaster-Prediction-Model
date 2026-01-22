# 🌍 ML Disaster Prediction Model

## 📌 Overview
This project presents an **intelligent multimodal disaster detection and classification system** that leverages **deep learning models for image and text analysis** to automatically identify disaster-related events from social media content. The system aims to support **early warning mechanisms** and **emergency response systems** by analyzing both **visual and semantic information**.

---

## 1. Introduction

### 1.1 Background
Natural and human-induced disasters such as **fire incidents, floods, land degradation, damaged infrastructure, and large-scale human damage** pose significant risks to public safety, infrastructure, and the environment. With the rapid growth of **social media platforms**, millions of users share **real-time images and textual updates** during emergency situations. These posts often provide the **earliest indicators** of an unfolding disaster, sometimes even before official agencies release alerts.

Traditional disaster monitoring systems rely heavily on **manual verification and reporting**, which is time-consuming, resource-intensive, and prone to human error. Recent advances in **deep learning, computer vision, and natural language processing (NLP)** enable automated systems capable of analyzing large-scale **multimodal data streams** with high accuracy. This project leverages these advancements to build a robust disaster prediction pipeline.

---

### 1.2 Motivation
Emergency responders and authorities often face challenges such as **delayed information**, **lack of situational awareness**, and **overwhelming volumes of unverified reports** during crisis events. Automating disaster detection using machine learning can significantly reduce response time and improve decision-making.

A **multimodal approach**, integrating **visual cues from images** and **semantic cues from textual descriptions**, provides a more comprehensive understanding of disaster scenarios. This approach improves robustness compared to single-modality models. The motivation is further strengthened by the success of deep learning architectures such as **ResNet, EfficientNet, Vision Transformer (ViT), MobileNetV2**, and NLP models like **BiLSTM and BERT**, which have demonstrated strong performance in complex classification tasks.

---

### 1.3 Objectives
The main objectives of this project are:

1. To design and develop an **automated disaster classification system** capable of identifying:
   - Fire  
   - Water / Flood  
   - Land Damage  
   - Human Damage  
   - Damaged Infrastructure  
   - Non-Damage scenarios  

2. To train and compare **state-of-the-art image classification models**, including:
   - AlexNet-style CNN  
   - MobileNetV2  
   - EfficientNet  
   - Vision Transformer (ViT)  
   - ResNet-50  

3. To evaluate **textual disaster descriptions** using advanced NLP models such as:
   - BiLSTM with GloVe embeddings  
   - BERT  
   and compare their performance with baseline text classifiers.

4. To integrate **image-based and text-based predictions** into a **unified multimodal pipeline** to improve overall detection reliability.

---

## 4. Dataset and Data Split

### 4.3 Data Split
Both image and text datasets were divided as follows:

- **70% Training**
- **15% Validation**
- **15% Testing**

This split ensures **fair evaluation**, effective hyperparameter tuning, and prevents **data leakage** across datasets.

---

## 5. Experiments and Results

This section describes the **experimental setup, training parameters, evaluation metrics, comparative analysis, and model behavior** for both image-based and text-based disaster classification systems.

---

### 5.1 Experimental Setup

#### 5.1.1 Hardware & Software
- **Programming Language:** Python  
- **Frameworks:** PyTorch, TensorFlow  
- **Execution Environment:** GPU-enabled Google Colab / Local Machine  
- **Libraries Used:** NumPy, Pandas, OpenCV, HuggingFace Transformers  

#### 5.1.2 Training Parameters
- **Optimizer:** Adam / SGD  
- **Learning Rate:** 1e-4 to 1e-3 (tuned per model)  
- **Batch Size:** 16–32  
- **Epochs:** 20–30 (depending on convergence)  
- **Loss Function:** Cross-Entropy Loss (multi-class classification)  

---

### 5.2 Evaluation and Results

#### 5.2.1 Image Classification Results
Images were classified into **six categories**: fire, water/flood, land damage, human damage, damaged infrastructure, and non-damage.

##### Model Comparison Summary

| Model | Accuracy | Observations |
|------|----------|--------------|
| AlexNet-style CNN | ~63% | Baseline model; struggled with complex textures and lighting |
| MobileNetV2 | ~86% | Lightweight and efficient; moderate performance |
| EfficientNet-B0/B1 | ~81% | Strong generalization across visual variations |
| Vision Transformer (ViT) | ~90% | Captured global features well; sensitive to dataset size |
| **ResNet-50 (Best)** | **~93%** | Best overall performance and training stability |

##### Why ResNet-50 Performed Best
- Residual connections prevented **gradient vanishing**.
- Deep hierarchical features captured **flames, flood patterns, debris, and human presence**.
- Robust to noise and real-world visual variations.

---

### 5.2.1.1 Image Classification – Key Insights
- Fire and flood classes achieved high precision due to distinct visual cues.
- Land damage and human damage showed moderate confusion due to overlapping textures.
- CNN-based architectures proved more stable than transformer-based models for limited datasets.

---

### 5.2.2 Training Curves and Convergence Analysis
Training and validation accuracy/loss curves were analyzed for EfficientNet, Vision Transformer, and ResNet-50.

- EfficientNet showed **smooth convergence** with minimal overfitting.
- ViT performance improved significantly after **fine-tuning**.
- ResNet-50 demonstrated the **most stable learning behavior** and highest validation accuracy.

*(Training curves are included in the repository under the `images/` directory.)*

---

## 6. Multimodal Disaster Classification Results

To improve robustness, predictions from **image-based models** and **text-based models** were combined into a **multimodal fusion pipeline**.

### 6.1 Text Classification Performance
- **BiLSTM + GloVe:** Captured sequential context effectively; moderate accuracy.
- **BERT:** Achieved superior performance due to contextualized embeddings and pretrained language understanding.

### 6.2 Multimodal Fusion Strategy
- Image and text predictions were combined using **late fusion**.
- Weighted confidence scores from both modalities were used for final classification.

### 6.3 Multimodal Results and Analysis
- Multimodal models consistently outperformed **single-modality systems**.
- Textual context helped disambiguate visually similar disaster scenes.
- Image features improved reliability when textual descriptions were vague or noisy.
- Overall, multimodal fusion improved **accuracy, robustness, and real-world reliability**.

---

## 🧠 Technologies Used
- Python  
- PyTorch / TensorFlow  
- CNN Architectures (ResNet, EfficientNet, MobileNetV2)  
- Vision Transformer (ViT)  
- NLP Models (BiLSTM + GloVe, BERT)  
- Multimodal Learning Pipelines  

---

## 🚀 Future Scope
- Real-time social media data ingestion
- Web-based or mobile disaster alert system
- Advanced multimodal fusion techniques
- Integration with government disaster-response platforms

---

## 📜 License
This project is intended for **academic and research purposes**.

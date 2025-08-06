# Retinal OCT Disease Detection Platform using Deep Learning

## 📌 Project Overview

This project presents a **Streamlit-based web application** for automated classification of retinal diseases using Optical Coherence Tomography (OCT) images. Utilizing a pre-trained deep learning model, the platform can accurately classify OCT images into one of four categories:

- **CNV** (Choroidal Neovascularization)
- **DME** (Diabetic Macular Edema)
- **Drusen**
- **Normal Retina**

The model was trained on a large-scale dataset of over 84,000 expert-labeled OCT scans, ensuring clinical relevance and robust performance. The application enables medical professionals and researchers to upload OCT images and receive instant predictions with detailed diagnostic insights.

---

## 🧪 Key Features

- **Easy Image Upload:** Upload OCT images with one click for classification.
- **Multi-class Prediction:** Detects four retinal disease categories (CNV, DME, Drusen, Normal).
- **Diagnostic Context:** Displays relevant medical information for each predicted disease.
- **Interactive UI:** Responsive and user-friendly interface built with Streamlit.
- **Custom Metrics:** Uses TensorFlow Addons’ F1-score metric to monitor model performance.
- **Optimized Performance:** Caching for fast model loading and efficient image processing.
- **Modular Codebase:** Clean, maintainable architecture for ease of extension and debugging.

---

## 🧠 Technologies Used

### Programming Languages & Frameworks
- **Python** – Main programming language
- **Streamlit** – Web app framework for interactive UI
- **TensorFlow & Keras** – Model development and inference
- **TensorFlow Addons (tfa)** – Custom F1-score metric
- **NumPy** – Numerical operations and image preprocessing

### Deep Learning
- Transfer learning with **MobileNetV3Large** pretrained on ImageNet
- CNN-based architecture tailored for multi-class classification (4 classes)
- Image preprocessing: resizing, normalization, and tensor manipulation

### MLOps / Deployment
- Deployment on **Streamlit Cloud** or local environments
- Efficient resource management with caching decorators
- Temporary file handling for secure and smooth image upload process

---

## 📁 Dataset Information

- **Source:** Aggregated from UC San Diego, Beijing Tongren Eye Center, and other reputed institutions.
- **Size:** 84,495 high-resolution OCT images.
- **Classes:** CNV, DME, Drusen, Normal.
- **Structure:** Split into training, validation, and testing sets.
- **Labeling:** Expert-verified, tiered annotation process.

---

## 🧪 Model Training & Evaluation Workflow

1. **Import Libraries**  
   TensorFlow, Pandas, NumPy, Matplotlib, Seaborn, and TensorFlow Addons for advanced metrics.

2. **Load Dataset**  
   Used `image_dataset_from_directory` for loading training and validation images, resized to 224x224 with batch size 32.

3. **Analyze Class Distribution**  
   Used `Counter` and label decoding to ensure balanced class representation.

4. **Define Model Architecture**  
   - Base model: MobileNetV3Large pretrained on ImageNet.  
   - Custom Dense layer with 4 output classes and softmax activation.  
   - Optional freezing of base layers for transfer learning.

5. **Compile Model**  
   - Optimizer: Adam (learning rate = 0.0001)  
   - Loss: Categorical crossentropy  
   - Metrics: Accuracy and macro F1-score (via TensorFlow Addons)

6. **Train Model**  
   15 epochs training on training and validation datasets.

7. **Evaluate & Save**  
   Evaluated on validation set, saved model as `Eye_disease_prediction_model.h5` and training history as `training_history.pkl`.

---

## ✅ Final Outcome

- A fully trained CNN model capable of classifying retinal OCT images into four categories.
- A Streamlit web app that loads this model to provide instant disease prediction and diagnostic insights.
- Reduced manual workload for ophthalmologists with a reliable and interpretable AI tool.

---

## 🛠 Getting Started

### Prerequisites

- Python 3.10
- TensorFlow 2.10+
- Streamlit
- TensorFlow Addons
- NumPy
- Other dependencies listed in `requirements.txt`

### Installation

```bash
git clone https://github.com/yourusername/retinal-oct-disease-detection.git
cd retinal-oct-disease-detection
pip install -r requirements.txt

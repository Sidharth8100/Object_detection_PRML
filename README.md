# 🧠 Object Recognition on CIFAR-10

This project explores various machine learning and deep learning techniques for object recognition using the **CIFAR-10** dataset — a benchmark dataset of 60,000 32x32 color images across 10 categories.

🔗 **[Project Website](https://aaryanagrawal96.github.io/object-recognition-home/)**  
📂 **[CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)**

---

## 📌 Overview

The goal of this project is to evaluate and compare different classification algorithms on the CIFAR-10 dataset. We explore both traditional ML models and deep learning architectures, emphasizing the role of preprocessing and feature extraction in handling high-dimensional image data.

---

## 🛠️ Methods Implemented

### 🔍 Preprocessing & Feature Engineering
- Data Normalization
- Dimensionality Reduction using **PCA**

### 🧪 Machine Learning Models
- Linear Regression
- K-Nearest Neighbors (KNN)  
- Logistic Regression  
- Support Vector Machines (SVM - Linear & RBF)  
- Decision Trees  
- Naive Bayes  
- Gaussian Mixture Models (GMM)  
- DBSCAN  
- K-Means Clustering  

### 🤖 Deep Learning Models
- Artificial Neural Networks (ANN)  
- Convolutional Neural Networks (CNN)

---

## 📊 Results

| Model                                          | Accuracy (%) |
|------------------------------------------------|--------------|
| Linear Regression                              | 10.96        |
| K-Nearest Neighbors (KNN)                      | 39.98        |
| Logistic Regression                            | 40.28        |
| SVM (Linear Kernel)                            | 38.47        |
| SVM (RBF Kernel)                               | 42.10        |
| DBSCAN                                         | 30.64        |
| Naive Bayes (Gaussian)                         | 36.98        |
| Gaussian Mixture Model (GMM)                   | 36.50        |
| Decision Tree                                  | 27.28        |
| K-Means Clustering                             | 22.69        |
| Artificial Neural Network (8 layers, L2 reg.)  | 52.00        |
| Convolutional Neural Network (Basic CNN)       | **80.50**     |

📌 The **CNN** model significantly outperforms traditional methods, underscoring:
- The **power of deep learning** in image recognition tasks.
- The **importance of feature extraction** and representation in high-dimensional spaces.

---

## 💡 Key Takeaways

- Dimensionality reduction (PCA) improves traditional model performance but may still fall short on complex image data.
- Deep learning models, especially CNNs, are highly effective for image classification due to their hierarchical feature learning.

---

## 🚀 Getting Started

To run the code:
1. Clone the repository.
2. Install dependencies using:

   ```bash
   pip install -r requirements.txt
3. Run the provided notebooks/scripts to train and evaluate models.

---

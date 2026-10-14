# 🌿 Plant Disease Detection Using VGG16

##  Project Overview

This project focuses on developing a deep learning model for plant disease classification using **VGG16 and Transfer Learning**. The model uses images from the PlantVillage dataset to identify plant diseases from leaf images.

The project explores how convolutional neural networks can support automated plant disease recognition and contribute to precision agriculture.

## Objectives

- Develop a plant disease classification system using VGG16.
- Perform exploratory data analysis (EDA) on the PlantVillage dataset.
- Preprocess images and prepare training, validation, and testing datasets.
- Apply data augmentation and fine-tuning to improve model generalization.
- Evaluate the model using standard classification metrics.
- Visualize model performance and save the trained model.

## 📂 Dataset

**Dataset:** PlantVillage

**Dataset source:** [PlantVillage Dataset on GitHub](https://github.com/spMohanty/PlantVillage-Dataset)

The dataset contains images of plant leaves belonging to different plant species and disease categories.

### Dataset Information

| Attribute | Description |
|---|---|
| Dataset name | PlantVillage |
| Total images identified | 20,624 |
| Number of classes | 15 |
| Image type | Plant leaf images |
| Task | Multiclass image classification |
| Input image size | 224 × 224 pixels |
| Dataset split | Training, validation, and testing |

## Technologies Used

- Python
- TensorFlow
- Keras
- VGG16
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- OpenCV
- Google Colab

## Project Workflow

1. Install the required Python libraries.
2. Clone and locate the PlantVillage dataset.
3. Verify dataset directories and image files.
4. Check for corrupted images, duplicate files, and class imbalance.
5. Perform exploratory data analysis and visualize class distributions.
6. Preprocess images and apply VGG16-compatible input scaling.
7. Split the dataset into training, validation, and testing sets.
8. Apply data augmentation to the training data.
9. Build the VGG16 transfer learning model.
10. Train the classification head with the convolutional base frozen.
11. Fine-tune selected VGG16 layers.
12. Evaluate the model on the held-out test set.
13. Generate performance plots and a confusion matrix.
14. Save the trained model and class-name mapping.

##  Model Architecture

The project uses VGG16, a convolutional neural network pretrained on ImageNet, as the feature extraction backbone.

### Architecture Components

- **Base model:** VGG16
- **Pretrained weights:** ImageNet
- **Input shape:** 224 × 224 × 3
- **Feature extraction:** Convolutional layers
- **Pooling:** Global Average Pooling
- **Regularization:** Dropout
- **Classification head:** Fully connected Dense layers
- **Output activation:** Softmax
- **Loss function:** Categorical or sparse categorical cross-entropy, depending on label encoding
- **Optimization:** Adam optimizer, subject to the training configuration

The base model is initially frozen to preserve pretrained features. Selected convolutional layers are subsequently unfrozen for fine-tuning.

## 📊 Exploratory Data Analysis

The following analyses are performed:

- Total number of images and classes.
- Number of images per class.
- Class distribution bar chart.
- Sample images from different classes.
- Image dimensions and channel information.
- Image integrity and corrupted-file checks.
- Duplicate image checks.
- Training, validation, and testing distribution.
- Analysis of class imbalance.

## Model Evaluation

The model is evaluated using the following metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report
- Training and validation accuracy curves
- Training and validation loss curves

### Performance Results

| Metric | Test Result |
|---|---|
| Accuracy | 97.83% |
| Precision | 97.92%|
| Recall | 97.83% |
| F1-score | 97.82% |

## 📁 Project Structure

```text
plant-disease-detection/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── plant_disease_vgg16.ipynb
│
├── models/
│   └── vgg16_plant_disease.keras
│
├── results/
│   ├── class_distribution.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── confusion_matrix.png
│
└── src/
    └── predict.py
```

## 🔮 Future Improvements

- Compare VGG16 with ResNet50, MobileNetV2, and ConvNeXt-Tiny.
- Improve classification performance on difficult disease categories.
- Investigate the effect of data augmentation and fine-tuning.
- Evaluate generalization on plant images captured under real-world conditions.
- Develop a web application for image-based disease prediction.
- Explore model optimization for mobile and edge deployment.

## ⚠️ Limitations

PlantVillage images are largely collected under controlled conditions. Consequently, a model trained on this dataset may perform differently on real-world photographs with varying backgrounds, lighting, camera quality, and disease severity.

Predictions should be validated before being used to make agricultural treatment decisions.

## 👨‍💻 Author

**Olaitan Ajewole**

Computer Science | Deep Learning | Computer Vision

GitHub: (https://github.com/HORLAHETAN))





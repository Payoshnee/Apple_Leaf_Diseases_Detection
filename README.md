# Enhanced Apple Leaf Disease Classification Using EfficientNetB0

A deep learning project for classifying apple leaf conditions using **EfficientNetB0**, **green channel extraction**, and **Contrast Limited Adaptive Histogram Equalization (CLAHE)**.

The system is designed to identify four apple leaf classes:

- Healthy
- Apple Scab
- Cedar Apple Rust
- Black Rot

The project combines image preprocessing, data augmentation, transfer learning, and model evaluation to build an accurate and computationally efficient apple leaf disease classification system.

---

## Project Overview

Apple leaf diseases can reduce crop quality and yield when they are not detected at an early stage. Manual disease identification is often time-consuming, subjective, and dependent on agricultural experts.

This project proposes an automated image-classification pipeline using a pretrained **EfficientNetB0** model. Green channel extraction and CLAHE are used to improve the visibility of disease-related patterns such as lesions, discoloration, and texture variations.

### Main objectives

- Automatically classify healthy and diseased apple leaves.
- Improve disease-feature visibility through image preprocessing.
- Reduce overfitting using data augmentation and dropout.
- Use transfer learning to achieve high accuracy with fewer computational resources.
- Support future deployment on mobile and edge devices.

---

## Methodology

The complete workflow consists of the following stages:

1. Dataset collection
2. Image preprocessing
3. Data augmentation
4. Dataset splitting
5. EfficientNetB0 transfer learning
6. Model training
7. Performance evaluation
8. Disease prediction

```text
Apple Leaf Image
        |
        v
Green Channel Extraction
        |
        v
CLAHE Enhancement
        |
        v
Resize and Data Augmentation
        |
        v
EfficientNetB0 Feature Extraction
        |
        v
Global Average Pooling
        |
        v
Dense Layer + Dropout
        |
        v
Softmax Classification
        |
        v
Healthy / Scab / Rust / Black Rot
```

---

## Dataset

The project uses apple leaf images obtained from the **PlantVillage dataset**.

### Dataset summary

| Property | Value |
|---|---:|
| Total images | 2,000 |
| Number of classes | 4 |
| Images per class | 500 |
| Original image size | 256 × 256 pixels |
| Training split | 70% |
| Validation split | 20% |
| Testing split | 10% |

### Recommended directory structure

```text
dataset/
├── healthy/
│   ├── image_001.jpg
│   └── ...
├── black_rot/
│   ├── image_001.jpg
│   └── ...
├── rust/
│   ├── image_001.jpg
│   └── ...
└── scab/
    ├── image_001.jpg
    └── ...
```

> Ensure that the class-folder names match the label mapping used in the training and prediction scripts.

---

## Image Preprocessing

### 1. Green Channel Extraction

Plant leaves reflect a large amount of green light. Extracting the green channel can improve the contrast between healthy tissue and disease-affected regions.

For an RGB image:

```text
I(x, y) = [R(x, y), G(x, y), B(x, y)]
```

The preprocessing stage selects:

```text
I_green(x, y) = G(x, y)
```

This can make discoloration, lesions, and texture changes more visible to the model.

### 2. CLAHE

CLAHE improves local contrast without excessively amplifying image noise. It divides an image into small regions, enhances each region independently, and combines the regions using interpolation.

Typical OpenCV implementation:

```python
import cv2

image = cv2.imread("leaf.jpg")
green_channel = image[:, :, 1]

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

enhanced_image = clahe.apply(green_channel)
```

---

## Data Augmentation

Data augmentation is used to simulate variations in leaf orientation, illumination, scale, and position.

The proposed transformations include:

- Resize to `256 × 256`
- Random crop to `224 × 224`
- Horizontal flip
- Vertical flip
- Rotation between `-30°` and `+30°`
- Brightness adjustment
- Contrast adjustment
- Affine transformations
- Random masking or coarse dropout
- Image normalization

Example augmentation pipeline:

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2

train_transform = A.Compose([
    A.Resize(256, 256),
    A.RandomCrop(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Affine(p=0.5),
    A.CoarseDropout(p=0.5),
    A.Normalize(mean=(0.5,), std=(0.5,)),
    ToTensorV2()
])
```

Adapt the normalization and channel configuration according to whether the model receives grayscale, repeated three-channel, or RGB images.

---

## Model Architecture

The project uses **EfficientNetB0** pretrained on ImageNet as the feature-extraction backbone.

### Classification head

```text
EfficientNetB0
    |
GlobalAveragePooling2D
    |
Dense Layer: 256 units, ReLU
    |
Dropout: 0.5
    |
Dense Layer: 4 units, Softmax
```

### Training configuration

| Parameter | Value |
|---|---|
| Base model | EfficientNetB0 |
| Pretrained weights | ImageNet |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss function | Categorical cross-entropy |
| Output activation | Softmax |
| Number of output classes | 4 |
| Dropout rate | 0.5 |
| Reported training duration | 100 epochs |

Example TensorFlow/Keras model:

```python
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0

NUM_CLASSES = 4
IMAGE_SIZE = (224, 224)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(*IMAGE_SIZE, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(*IMAGE_SIZE, 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()
```

After training the classification head, selected EfficientNetB0 layers can be unfrozen for fine-tuning with a lower learning rate.

---

## Suggested Repository Structure

```text
apple-leaf-disease-classification/
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── preprocessing.ipynb
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── efficientnetb0_apple_leaf.keras
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── confusion_matrix.png
│   └── classification_report.txt
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/apple-leaf-disease-classification.git
cd apple-leaf-disease-classification
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux or macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Suggested `requirements.txt`:

```text
tensorflow
opencv-python
albumentations
numpy
pandas
matplotlib
scikit-learn
pillow
jupyter
```

Install `torch` and `torchvision` as well when using `ToTensorV2` or a PyTorch implementation.

---

## Training

Place the dataset in the required directory structure and run the training script used by the repository.

Example:

```bash
python src/train.py
```

A typical training workflow should:

1. Load images and labels.
2. Apply green channel extraction.
3. Apply CLAHE.
4. Apply training augmentations.
5. Split the dataset into training, validation, and testing sets.
6. Load EfficientNetB0 with ImageNet weights.
7. Train the classification head.
8. Fine-tune selected base-model layers.
9. Save the best-performing model.
10. Export training curves and evaluation metrics.

---

## Evaluation

Evaluate the trained model on the unseen test dataset:

```bash
python src/evaluate.py
```

Recommended evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Per-class classification report

---

## Reported Results

The Results section of the accompanying study reports the following overall performance:

| Metric | Reported value |
|---|---:|
| Training accuracy | 99.89% |
| Validation accuracy | 99.25% |
| Test accuracy | 98.23% |

### Reported class-wise performance

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.9703 | 0.9899 | 0.9800 | 99 |
| Multiple Disease / Black Rot* | 0.9899 | 0.9899 | 0.9899 | 99 |
| Rust | 1.0000 | 0.9899 | 0.9949 | 99 |
| Scab | 0.9694 | 0.9596 | 0.9645 | 99 |

\*The manuscript describes **Black Rot** as one of the four target classes, while the final metric table uses the label **Multiple Disease**. Confirm the final class name before reproducing or publishing the results.

---

## Predicting a New Image

A prediction script should perform the same preprocessing used during training.

Example usage:

```bash
python src/predict.py --image path/to/apple_leaf.jpg
```

Example output:

```text
Predicted class: Rust
Confidence: 98.74%
```

Basic prediction logic:

```python
import numpy as np
import tensorflow as tf

CLASS_NAMES = ["black_rot", "healthy", "rust", "scab"]

model = tf.keras.models.load_model(
    "models/efficientnetb0_apple_leaf.keras"
)

image = preprocess_image("sample_leaf.jpg")
prediction = model.predict(
    np.expand_dims(image, axis=0),
    verbose=0
)[0]

predicted_index = int(np.argmax(prediction))

print("Predicted class:", CLASS_NAMES[predicted_index])
print("Confidence:", float(prediction[predicted_index]))
```

The `preprocess_image()` function must use the same image size, channel format, CLAHE configuration, and normalization used during model training.

---

## Key Advantages

- Uses transfer learning for efficient model training.
- Enhances disease symptoms through green channel extraction.
- Improves local contrast using CLAHE.
- Uses data augmentation to improve generalization.
- Has fewer parameters than many larger CNN architectures.
- Can potentially be deployed on mobile and edge devices.
- Supports early and automated plant disease identification.

---

## Limitations

- Images are primarily collected under controlled conditions.
- Performance may decrease for complex natural backgrounds.
- Lighting, shadows, overlapping leaves, and blur may affect predictions.
- A dataset of 2,000 images may not represent every real-world disease variation.
- The same preprocessing pipeline must be applied during training and inference.
- Reported accuracy values should be verified using the final saved model and test split.

---

## Future Work

Possible extensions include:

- Expanding the dataset with real orchard images.
- Adding more apple disease categories.
- Applying hyperparameter optimization.
- Comparing EfficientNetB0 with MobileNet, DenseNet, ResNet, and Vision Transformers.
- Using Grad-CAM or SHAP for explainable predictions.
- Performing disease-region localization or object detection.
- Building an Android or web-based application.
- Deploying a quantized model on Raspberry Pi or smartphones.
- Testing the model under real-world lighting and background conditions.

---

## Reproducibility Notes

Before publishing the repository, verify the following values against the final experiment:

- The Abstract reports a test accuracy of `97.47%`, while the Results section reports `98.23%`.
- One methodology paragraph mentions `30 epochs`, while the final experiments describe `100 epochs`.
- The target class is described as `Black Rot`, but one results table uses `Multiple Disease`.
- The model input channel format should be documented clearly after green channel extraction.

Keeping a single experiment configuration file is recommended:

```yaml
dataset:
  total_images: 2000
  classes:
    - healthy
    - black_rot
    - rust
    - scab
  train_split: 0.70
  validation_split: 0.20
  test_split: 0.10

model:
  architecture: EfficientNetB0
  pretrained_weights: imagenet
  image_size: [224, 224]
  dropout: 0.5
  num_classes: 4

training:
  optimizer: Adam
  learning_rate: 0.0001
  loss: categorical_crossentropy
  epochs: 100
```

---

## Author

**Payoshnee Parag Joshi**  
B.Tech, Computer Science and Engineering  
Specialization in Cloud Computing and Automation  
VIT Bhopal University

---

## Citation

When using this project or methodology in academic work, cite the accompanying study:

```bibtex
@article{joshi_apple_leaf_efficientnetb0,
  title   = {Enhanced Apple Leaf Disease Classification Using EfficientNetB0 with Green Channel Extraction and CLAHE-Based Preprocessing},
  author  = {Joshi, Payoshnee Parag},
  note    = {B.Tech research project, VIT Bhopal University}
}
```

---

## License

Add a license before publicly distributing the repository.

For open-source academic projects, the MIT License is a common choice. Dataset usage must also follow the original PlantVillage or Kaggle dataset license and terms.

---

## Acknowledgements

- PlantVillage dataset contributors
- Kaggle
- EfficientNet authors
- TensorFlow/Keras
- OpenCV
- Albumentations
- VIT Bhopal University


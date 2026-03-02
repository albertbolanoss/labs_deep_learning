#!/usr/bin/env python3
"""
Script to create the advanced fruit detection notebook
"""
import json

# Create notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU"
    },
    "nbformat": 4,
    "nbformat_minor": 0
}

# Helper function to add cells
def add_markdown(text):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split("\n")
    })

def add_code(code):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split("\n")
    })

# Title
add_markdown("""# Advanced Fruit Single Shot Detection (SSD) Model

This notebook implements a state-of-the-art object detection model using Transfer Learning to detect and localize fruits with high precision.

**Key Features:**
- Transfer Learning with MobileNetV2 backbone
- Advanced loss functions (Focal Loss + Smooth L1 Loss)
- Proper data augmentation for object detection
- Comprehensive evaluation and visualization
- Model checkpointing and saving

**Author:** Computer Vision Engineer  
**Dataset:** Fruit Images for Object Detection (Kaggle)""")

# Section 1: Setup
add_markdown("## 1. Setup & Imports\n\nInstall and import all necessary libraries.")

add_code("""# Install required packages
!pip install -q tensorflow opencv-python-headless matplotlib scikit-learn

# Core imports
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

# Google Colab specific
from google.colab import userdata

print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)""")

# Section 2: Configuration
add_markdown("## 2. Configuration & Hyperparameters\n\nDefine all configuration parameters.")

add_code("""# Dataset Configuration
KAGGLE_DATASET = "mbkinaci/fruit-images-for-object-detection"
BASE_DIR = "/content/fruit_data"
MODEL_TO_SAVE = "fruit_detector_v1.keras"

# Model Configuration
IMG_WIDTH = 224
IMG_HEIGHT = 224
GRID_SIZE = 7
NUM_CLASSES = 3

# Class Mapping
CLASS_MAP = {"apple": 0, "banana": 1, "orange": 2}
INV_CLASS_MAP = {0: "apple", 1: "banana", 2: "orange"}
CLASS_COLORS = {0: 'red', 1: 'yellow', 2: 'orange'}

# Training Hyperparameters
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
PATIENCE = 10

# Loss weights
LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

print("Configuration loaded successfully!")
print(f"Image Size: {IMG_WIDTH}x{IMG_HEIGHT}")
print(f"Grid Size: {GRID_SIZE}x{GRID_SIZE}")
print(f"Classes: {list(CLASS_MAP.keys())}")""")

# Section 3: Dataset Download
add_markdown("## 3. Dataset Download\n\nDownload the fruit detection dataset from Kaggle.")

add_code("""# Configure Kaggle API
os.environ['KAGGLE_USERNAME'] = userdata.get('KAGGLE_USERNAME')
os.environ['KAGGLE_KEY'] = userdata.get('KAGGLE_KEY')

# Download and extract dataset
if not os.path.exists(BASE_DIR):
    print("Downloading dataset...")
    !kaggle datasets download -d {KAGGLE_DATASET}
    !unzip -q fruit-images-for-object-detection.zip -d {BASE_DIR}
    print(f"Dataset downloaded to {BASE_DIR}")
else:
    print("Dataset already exists.")

# Verify structure
print("\\nDataset structure:")
for folder in os.listdir(BASE_DIR):
    folder_path = os.path.join(BASE_DIR, folder)
    if os.path.isdir(folder_path):
        print(f"  {folder}/")""")

# Section 4: Data Parsing
add_markdown("""## 4. Data Parsing & Preprocessing

Parse XML annotations and load images with proper normalization.""")

add_code("""def get_image_dimensions(img_path):
    \"\"\"Get actual image dimensions from file.\"\"\"
    try:
        img = cv2.imread(img_path)
        if img is not None:
            height, width = img.shape[:2]
            return width, height
        return None, None
    except Exception as e:
        print(f"Error reading {img_path}: {e}")
        return None, None

def parse_xml_annotation(xml_path, img_dir):
    \"\"\"Parse XML annotation file and extract bounding boxes.\"\"\"
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        filename = root.find("filename").text
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            return None, None
        
        w_orig, h_orig = get_image_dimensions(img_path)
        if w_orig is None or h_orig is None:
            return None, None
        
        boxes = []
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name not in CLASS_MAP:
                continue
            
            label = CLASS_MAP[name]
            bndbox = obj.find("bndbox")
            
            xmin = int(bndbox.find("xmin").text)
            ymin = int(bndbox.find("ymin").text)
            xmax = int(bndbox.find("xmax").text)
            ymax = int(bndbox.find("ymax").text)
            
            # Normalize coordinates
            xmin_norm = max(0, min(1, xmin / w_orig))
            ymin_norm = max(0, min(1, ymin / h_orig))
            xmax_norm = max(0, min(1, xmax / w_orig))
            ymax_norm = max(0, min(1, ymax / h_orig))
            
            if xmax_norm > xmin_norm and ymax_norm > ymin_norm:
                boxes.append([xmin_norm, ymin_norm, xmax_norm, ymax_norm, label])
        
        return img_path, boxes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None, None

def load_dataset(directory):
    \"\"\"Load all images and annotations from a directory.\"\"\"
    images = []
    annotations = []
    
    print(f"Loading data from: {directory}")
    xml_count = 0
    valid_count = 0
    
    for root_dir, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".xml"):
                xml_count += 1
                xml_path = os.path.join(root_dir, file)
                img_path, boxes = parse_xml_annotation(xml_path, root_dir)
                
                if img_path is None or not boxes:
                    continue
                
                try:
                    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_arr = img_to_array(img) / 255.0
                    
                    images.append(img_arr)
                    annotations.append(boxes)
                    valid_count += 1
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
    
    print(f"  Processed {xml_count} XML files")
    print(f"  Loaded {valid_count} valid images")
    
    return np.array(images, dtype="float32"), annotations

# Load training data
print("\\n=== Loading Training Data ===")
X_train_full, train_annotations = load_dataset(os.path.join(BASE_DIR, "train_zip", "train"))

# Load test data
print("\\n=== Loading Test Data ===")
X_test_full, test_annotations = load_dataset(os.path.join(BASE_DIR, "test_zip", "test"))

# Split test into validation and test
print("\\n=== Splitting Test Data ===")
X_val, X_test, val_annotations, test_annotations = train_test_split(
    X_test_full, test_annotations,
    test_size=0.5, random_state=SEED, shuffle=True
)

print("\\n=== Dataset Summary ===")
print(f"Training samples: {len(X_train_full)}")
print(f"Validation samples: {len(X_val)}")
print(f"Test samples: {len(X_test)}")
print(f"Image shape: {X_train_full[0].shape}")""")

# Section 5: Visualization
add_markdown("""## 5. Data Visualization

Visualize sample images with ground truth bounding boxes.""")

add_code("""def visualize_sample(images, annotations, index, title="Sample"):
    \"\"\"Visualize an image with its bounding boxes.\"\"\"
    img = images[index]
    boxes = annotations[index]
    
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img)
    
    img_h, img_w = img.shape[:2]
    
    for box in boxes:
        xmin_n, ymin_n, xmax_n, ymax_n, label_idx = box
        
        xmin = xmin_n * img_w
        ymin = ymin_n * img_h
        xmax = xmax_n * img_w
        ymax = ymax_n * img_h
        
        width = xmax - xmin
        height = ymax - ymin
        
        class_name = INV_CLASS_MAP[int(label_idx)]
        color = CLASS_COLORS[int(label_idx)]
        
        rect = patches.Rectangle(
            (xmin, ymin), width, height,
            linewidth=3, edgecolor=color, facecolor='none'
        )
        ax.add_patch(rect)
        
        ax.text(
            xmin, ymin - 5, class_name.upper(),
            color='white', fontsize=14, weight='bold',
            bbox=dict(facecolor=color, alpha=0.7, edgecolor='none', pad=3)
        )
    
    ax.axis('off')
    ax.set_title(f"{title} - Index: {index}", fontsize=16, weight='bold')
    plt.tight_layout()
    plt.show()

# Visualize random samples
print("Visualizing training samples...\\n")
for i in range(3):
    idx = random.randint(0, len(X_train_full) - 1)
    visualize_sample(X_train_full, train_annotations, idx, "Training Sample")""")

# Section 6: Target Encoding
add_markdown("""## 6. Target Encoding for Grid-Based Detection

Convert bounding box annotations to grid-based format (YOLO/SSD style).

**Output Format:** Each grid cell contains:
- Confidence (1 if object present, 0 otherwise)
- Center X, Center Y (normalized)
- Width, Height (normalized)
- Class probabilities (one-hot encoded)

**Tensor Shape:** `(GRID_SIZE, GRID_SIZE, 5 + NUM_CLASSES)`""")

add_code("""def encode_target(annotations, grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
    \"\"\"Encode bounding box annotations into grid-based format.\"\"\"
    target = np.zeros((grid_size, grid_size, 5 + num_classes), dtype=np.float32)
    
    for box in annotations:
        xmin, ymin, xmax, ymax, label = box
        
        # Calculate center and size
        cx = (xmin + xmax) / 2.0
        cy = (ymin + ymax) / 2.0
        w = xmax - xmin
        h = ymax - ymin
        
        # Determine responsible grid cell
        col = int(cx * grid_size)
        row = int(cy * grid_size)
        
        col = min(col, grid_size - 1)
        row = min(row, grid_size - 1)
        
        # Assign if cell is empty
        if target[row, col, 0] == 0:
            target[row, col, 0] = 1.0  # Confidence
            target[row, col, 1:5] = [cx, cy, w, h]  # Box
            target[row, col, 5 + int(label)] = 1.0  # Class
    
    return target

# Encode all datasets
print("Encoding targets...")
y_train = np.array([encode_target(ann) for ann in train_annotations])
y_val = np.array([encode_target(ann) for ann in val_annotations])
y_test = np.array([encode_target(ann) for ann in test_annotations])

print(f"\\nEncoded target shapes:")
print(f"  y_train: {y_train.shape}")
print(f"  y_val: {y_val.shape}")
print(f"  y_test: {y_test.shape}")

total_objects_train = np.sum(y_train[:, :, :, 0])
total_objects_val = np.sum(y_val[:, :, :, 0])
print(f"\\nTotal objects:")
print(f"  Training: {int(total_objects_train)}")
print(f"  Validation: {int(total_objects_val)}")""")

# Section 7: Data Augmentation
add_markdown("""## 7. Data Augmentation

Implement augmentation techniques suitable for object detection.""")

add_code("""def augment_image_and_boxes(image, boxes, flip_prob=0.5):
    \"\"\"Apply data augmentation to image and update bounding boxes.\"\"\"
    aug_image = image.copy()
    aug_boxes = [box.copy() for box in boxes]
    
    # Horizontal flip
    if random.random() < flip_prob:
        aug_image = np.fliplr(aug_image)
        for box in aug_boxes:
            xmin, ymin, xmax, ymax, label = box
            box[0] = 1.0 - xmax
            box[2] = 1.0 - xmin
    
    # Brightness adjustment
    if random.random() < 0.5:
        brightness_factor = random.uniform(0.8, 1.2)
        aug_image = np.clip(aug_image * brightness_factor, 0, 1)
    
    # Contrast adjustment
    if random.random() < 0.5:
        contrast_factor = random.uniform(0.8, 1.2)
        mean = aug_image.mean()
        aug_image = np.clip((aug_image - mean) * contrast_factor + mean, 0, 1)
    
    return aug_image, aug_boxes

class DataGenerator(keras.utils.Sequence):
    \"\"\"Custom data generator with augmentation.\"\"\"
    def __init__(self, images, annotations, batch_size=BATCH_SIZE, shuffle=True, augment=False):
        self.images = images
        self.annotations = annotations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indexes = np.arange(len(self.images))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.images) / self.batch_size))
    
    def __getitem__(self, index):
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.images))
        batch_indexes = self.indexes[start_idx:end_idx]
        
        X_batch = []
        y_batch = []
        
        for idx in batch_indexes:
            image = self.images[idx]
            boxes = self.annotations[idx]
            
            if self.augment:
                image, boxes = augment_image_and_boxes(image, boxes)
            
            target = encode_target(boxes)
            X_batch.append(image)
            y_batch.append(target)
        
        return np.array(X_batch), np.array(y_batch)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

# Create data generators
train_generator = DataGenerator(
    X_train_full, train_annotations,
    batch_size=BATCH_SIZE, shuffle=True, augment=True
)

val_generator = DataGenerator(
    X_val, val_annotations,
    batch_size=BATCH_SIZE, shuffle=False, augment=False
)

print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches per epoch: {len(val_generator)}")""")

# Section 8: Model Architecture
add_markdown("""## 8. Model Architecture - Transfer Learning with MobileNetV2

Build the detection model using pre-trained MobileNetV2 backbone.""")

add_code("""def build_detection_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), grid_size=GRID_SIZE, num_classes=NUM_CLASSES):
    \"\"\"Build object detection model with MobileNetV2 backbone.\"\"\"
    inputs = layers.Input(shape=input_shape, name='input_image')
    
    # Load pre-trained MobileNetV2
    backbone = MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        alpha=1.0
    )
    
    # Freeze early layers
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    
    x = backbone(inputs, training=False)
    
    # Detection head
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='det_conv1')(x)
    x = layers.BatchNormalization(name='det_bn1')(x)
    x = layers.Dropout(0.3, name='det_dropout1')(x)
    
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='det_conv2')(x)
    x = layers.BatchNormalization(name='det_bn2')(x)
    x = layers.Dropout(0.2, name='det_dropout2')(x)
    
    # Output layer
    output_channels = 5 + num_classes
    x = layers.Conv2D(output_channels, (1, 1), padding='same', name='det_output')(x)
    
    # Resize to grid size
    outputs = layers.Resizing(grid_size, grid_size, name='output_resize')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='FruitDetector')
    return model

# Build model
print("Building model...")
model = build_detection_model()
model.summary()

trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
non_trainable_params = np.sum([K.count_params(w) for w in model.non_trainable_weights])

print(f"\\nModel Parameters:")
print(f"  Trainable: {trainable_params:,}")
print(f"  Non-trainable: {non_trainable_params:,}")
print(f"  Total: {trainable_params + non_trainable_params:,}")""")

# Section 9: Loss Functions
add_markdown("""## 9. Custom Loss Functions

Implement advanced loss functions:
1. Focal Loss for classification
2. Smooth L1 Loss for bounding box regression
3. Combined detection loss""")

add_code("""def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    \"\"\"Focal Loss for addressing class imbalance.\"\"\"
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    return K.sum(loss, axis=-1)

def smooth_l1_loss(y_true, y_pred, beta=1.0):
    \"\"\"Smooth L1 Loss for bounding box regression.\"\"\"
    diff = K.abs(y_true - y_pred)
    less_than_beta = K.cast(K.less(diff, beta), 'float32')
    loss = (less_than_beta * 0.5 * diff ** 2 / beta) + ((1 - less_than_beta) * (diff - 0.5 * beta))
    return K.sum(loss, axis=-1)

def detection_loss(y_true, y_pred):
    \"\"\"Combined detection loss function.\"\"\"
    # Extract components
    true_conf = y_true[..., 0:1]
    true_box = y_true[..., 1:5]
    true_class = y_true[..., 5:]
    
    pred_conf = K.sigmoid(y_pred[..., 0:1])
    pred_box = y_pred[..., 1:5]
    pred_class = K.softmax(y_pred[..., 5:])
    
    # Object masks
    obj_mask = true_conf
    noobj_mask = 1 - obj_mask
    
    # 1. Confidence Loss
    conf_loss_obj = obj_mask * K.binary_crossentropy(true_conf, pred_conf)
    conf_loss_noobj = noobj_mask * K.binary_crossentropy(true_conf, pred_conf)
    conf_loss = K.sum(conf_loss_obj) + LAMBDA_NOOBJ * K.sum(conf_loss_noobj)
    
    # 2. Localization Loss
    loc_loss = obj_mask * smooth_l1_loss(true_box, pred_box)
    loc_loss = LAMBDA_COORD * K.sum(loc_loss)
    
    # 3. Classification Loss
    class_loss = obj_mask * focal_loss(true_class, pred_class)
    class_loss = K.sum(class_loss)
    
    total_loss = conf_loss + loc_loss + class_loss
    return total_loss

def confidence_accuracy(y_true, y_pred):
    \"\"\"Calculate accuracy of confidence predictions.\"\"\"
    true_conf = y_true[..., 0]
    pred_conf = K.sigmoid(y_pred[..., 0])
    pred_conf_binary = K.cast(K.greater(pred_conf, 0.5), 'float32')
    return K.mean(K.equal(true_conf, pred_conf_binary))

def mean_iou(y_true, y_pred):
    \"\"\"Calculate mean IoU for predicted boxes.\"\"\"
    obj_mask = y_true[..., 0]
    true_box = y_true[..., 1:5]
    pred_box = y_pred[..., 1:5]
    intersection = K.sum(K.minimum(true_box, pred_box) * obj_mask, axis=-1)
    union = K.sum(K.maximum(true_box, pred_box) * obj_mask, axis=-1)
    iou = intersection / (union + K.epsilon())
    return K.mean(iou)

print("Loss functions defined successfully!")
print(f"\\nLoss weights:")
print(f"  Coordinate loss weight: {LAMBDA_COORD}")
print(f"  No-object loss weight: {LAMBDA_NOOBJ}")""")

# Section 10: Model Compilation
add_markdown("""## 10. Model Compilation

Compile the model with custom loss and optimizer.""")

add_code("""# Compile model
optimizer = optimizers.Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss=detection_loss,
    metrics=[confidence_accuracy, mean_iou]
)

print("Model compiled successfully!")
print(f"\\nTraining configuration:")
print(f"  Optimizer: Adam")
print(f"  Learning rate: {LEARNING_RATE}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {EPOCHS}")
print(f"  Early stopping patience: {PATIENCE}")""")

# Save notebook
output_path = "notebooks/fruit-single-detector/02_advanced_training_model.ipynb"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=2, ensure_ascii=False)

print(f"Notebook created successfully: {output_path}")

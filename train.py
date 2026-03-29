import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import (
    Dense, GlobalAveragePooling2D, Dropout, Input, Concatenate,
    BatchNormalization, GaussianNoise, Multiply, Layer
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,
    LearningRateScheduler
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import math
import cv2

# =============================
# Enable Mixed Precision for Faster Training
# =============================
# Mixed precision disabled — causes NaN on CPU-only setups
# If you have a GPU, uncomment the line below:
# tf.keras.mixed_precision.set_global_policy('mixed_float16')
print("ℹ️ Using float32 precision (stable for CPU training)")

# =============================
# Dataset Paths
# =============================

metadata_path = "archive/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"
image_dir1 = "archive/skin-cancer-mnist-ham10000/HAM10000_images_part_1"
image_dir2 = "archive/skin-cancer-mnist-ham10000/HAM10000_images_part_2"

# =============================
# Class Mapping (7-class HAM10000)
# =============================

CLASS_MAP = {
    "akiec": "Actinic Keratoses",
    "bcc":   "Basal Cell Carcinoma",
    "bkl":   "Benign Keratosis",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic Nevi",
    "vasc":  "Vascular Lesion",
}

CLASSES = list(CLASS_MAP.values())
NUM_CLASSES = len(CLASSES)

# =============================
# Load Metadata
# =============================

df = pd.read_csv(metadata_path)
df["label"] = df["dx"].map(CLASS_MAP)

# =============================
# Encode Metadata Features
# =============================

# Sex: male=0, female=1, unknown=0.5
sex_map = {"male": 0.0, "female": 1.0}
df["sex_encoded"] = df["sex"].map(sex_map).fillna(0.5)

# Localization one-hot
loc_dummies = pd.get_dummies(df["localization"], prefix="loc")
df = pd.concat([df, loc_dummies], axis=1)

# Age normalized (0-1)
df["age"] = df["age"].fillna(df["age"].median())
df["age_norm"] = df["age"] / 100.0

METADATA_COLS = ["age_norm", "sex_encoded"] + [c for c in df.columns if c.startswith("loc_")]
NUM_META_FEATURES = len(METADATA_COLS)

# =============================
# Get Image Path
# =============================

def get_image_path(image_id):
    path1 = os.path.join(image_dir1, image_id + ".jpg")
    path2 = os.path.join(image_dir2, image_id + ".jpg")
    return path1 if os.path.exists(path1) else path2

df["path"] = df["image_id"].apply(get_image_path)

# =============================
# Oversample Minority Classes (Balance Dataset)
# =============================

# Partial oversampling — 3x median count (much smaller than matching max)
median_count = int(df["label"].value_counts().median())
target_count = min(median_count * 3, df["label"].value_counts().max())
oversampled_dfs = []
for cls in CLASSES:
    cls_df = df[df["label"] == cls]
    if len(cls_df) < target_count:
        oversampled = cls_df.sample(target_count, replace=True, random_state=42)
        oversampled_dfs.append(oversampled)
    else:
        oversampled_dfs.append(cls_df)
df_balanced = pd.concat(oversampled_dfs, ignore_index=True)
print(f"Original dataset: {len(df)} samples")
print(f"Balanced dataset: {len(df_balanced)} samples")
print(f"Class distribution:\n{df_balanced['label'].value_counts()}")

# =============================
# Train / Val / Test Split
# =============================

train_df, temp_df = train_test_split(df_balanced, test_size=0.2, random_state=42, stratify=df_balanced["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

print(f"\nTrain: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# =============================
# Class Weights (additional safety for imbalance)
# =============================

weights = compute_class_weight("balanced", classes=np.array(CLASSES), y=train_df["label"].values)
class_weight_dict = {i: w for i, w in enumerate(weights)}
print("Class weights:", class_weight_dict)

# =============================
# NEW FEATURE: Hair Removal Preprocessing (DullRazor Algorithm)
# =============================

def remove_hair(image_array):
    """Remove dark hair from dermoscopy images using morphological blackhat filter.
    
    Args:
        image_array: numpy array (H, W, 3) in uint8 [0-255] or float32 [0-1]
    
    Returns:
        Cleaned image as same dtype as input.
    """
    was_float = image_array.dtype == np.float32 or image_array.dtype == np.float64
    if was_float:
        img = (image_array * 255).astype(np.uint8)
    else:
        img = image_array.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Blackhat filter — highlights thin dark structures (hair)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    # Threshold to create hair mask
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    
    # Dilate mask slightly to cover hair edges
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    
    # Inpaint the hair regions
    cleaned = cv2.inpaint(img, mask, inpaintRadius=6, flags=cv2.INPAINT_TELEA)
    
    if was_float:
        return cleaned.astype(np.float32) / 255.0
    return cleaned

print("✅ Hair removal preprocessing loaded")

# =============================
# NEW FEATURE: CutMix / MixUp Augmentation
# =============================

def cutmix_mixup_batch(images, labels, meta=None, prob=0.5):
    """Apply CutMix or MixUp to a batch of images with given probability.
    
    Args:
        images: batch of images (B, H, W, C)
        labels: batch of one-hot labels (B, num_classes)
        meta: batch of metadata (B, num_features) or None
        prob: probability of applying augmentation to each sample
    
    Returns:
        Augmented (images, labels, meta) tuple
    """
    batch_size = len(images)
    if batch_size < 2:
        return images, labels, meta
    
    # Decide: apply augmentation with probability `prob`
    if np.random.random() > prob:
        return images, labels, meta
    
    # Randomly choose CutMix or MixUp
    use_cutmix = np.random.random() > 0.5
    
    # Shuffle indices for pairing
    indices = np.random.permutation(batch_size)
    
    if use_cutmix:
        # CutMix: cut a rectangle from one image and paste onto another
        lam = np.random.beta(1.0, 1.0)  # Beta distribution for lambda
        h, w = images.shape[1], images.shape[2]
        
        # Calculate cut dimensions
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(h * cut_ratio)
        cut_w = int(w * cut_ratio)
        
        # Random position
        cy = np.random.randint(h)
        cx = np.random.randint(w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        
        # Apply CutMix
        mixed_images = images.copy()
        mixed_images[:, y1:y2, x1:x2, :] = images[indices, y1:y2, x1:x2, :]
        
        # Adjust lambda based on actual cut area
        lam = 1 - ((y2 - y1) * (x2 - x1)) / (h * w)
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
        
    else:
        # MixUp: linear interpolation of images and labels
        lam = np.random.beta(0.4, 0.4)  # Sharper distribution for MixUp
        mixed_images = lam * images + (1 - lam) * images[indices]
        mixed_labels = lam * labels + (1 - lam) * labels[indices]
    
    # Mix metadata proportionally too
    if meta is not None:
        mixed_meta = lam * meta + (1 - lam) * meta[indices]
    else:
        mixed_meta = None
    
    return mixed_images, mixed_labels, mixed_meta

print("✅ CutMix/MixUp augmentation loaded")

# =============================
# Custom Data Generator (Image + Metadata) with Strong Augmentation
# =============================

IMG_SIZE = 224  # EfficientNetB4 works fine at 224, much faster than 300
BATCH_SIZE = 32  # Larger batch = fewer steps per epoch = faster

def dual_input_generator(dataframe, augment=False, shuffle=True, apply_cutmix=False):
    """Yields (image, metadata), label tuples for training.
    
    Args:
        dataframe: pandas DataFrame with image paths and labels
        augment: whether to apply standard augmentations
        shuffle: whether to shuffle the data
        apply_cutmix: whether to apply CutMix/MixUp (training only)
    """
    if augment:
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            brightness_range=[0.7, 1.3],
            channel_shift_range=30,
            fill_mode='reflect',
        )
    else:
        datagen = ImageDataGenerator(rescale=1./255)

    # Reset index so we can map filenames back to rows
    df_reset = dataframe.reset_index(drop=True)

    img_gen = datagen.flow_from_dataframe(
        dataframe=df_reset,
        x_col="path",
        y_col="label",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        classes=CLASSES,
        shuffle=shuffle,
    )

    meta_values = df_reset[METADATA_COLS].values.astype("float32")
    n_samples = len(df_reset)

    while True:
        imgs, labels = next(img_gen)
        batch_size = len(imgs)

        # Get the indices that were just used by the generator
        # After next() is called, batch_index has advanced, so we look back
        if img_gen.batch_index == 0:
            # Wrapped around — last batch of the epoch
            idx_start = n_samples - batch_size
            indices = img_gen.index_array[idx_start:] if img_gen.index_array is not None else np.arange(batch_size) % n_samples
        else:
            idx_end = img_gen.batch_index * BATCH_SIZE
            idx_start = idx_end - batch_size
            indices = img_gen.index_array[idx_start:idx_end] if img_gen.index_array is not None else np.arange(batch_size) % n_samples

        # Safety fallback
        if len(indices) != batch_size:
            indices = np.arange(batch_size) % n_samples

        meta_batch = meta_values[indices]
        
        # NEW: Apply CutMix/MixUp augmentation during training
        if apply_cutmix and augment:
            imgs, labels, meta_batch = cutmix_mixup_batch(
                imgs, labels, meta=meta_batch, prob=0.5
            )
        
        yield [imgs, meta_batch], labels

# Enable CutMix/MixUp for training generator
train_gen = dual_input_generator(train_df, augment=True, shuffle=True, apply_cutmix=True)
val_gen = dual_input_generator(val_df, augment=False, shuffle=False, apply_cutmix=False)

train_steps = len(train_df) // BATCH_SIZE
val_steps = len(val_df) // BATCH_SIZE

# =============================
# NEW FEATURE: True Focal Loss (replaces stable_categorical_crossentropy)
# =============================

def focal_loss(alpha=0.25, gamma=2.0):
    """Focal Loss — focuses learning on hard-to-classify minority examples.
    
    Focal Loss down-weights well-classified examples and focuses on the ones
    the model is getting wrong. This is critical for imbalanced datasets like
    HAM10000 where Melanocytic Nevi dominates (67% of samples).
    
    Args:
        alpha: Balancing factor (0-1). Lower = less weight on easy classes.
        gamma: Focusing parameter. Higher = more focus on hard examples.
               gamma=0 is equivalent to standard cross-entropy.
    """
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        cross_entropy = -y_true * tf.math.log(y_pred)
        # Focal modulation: (1 - p_t)^gamma
        focal_weight = tf.pow(1.0 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)
    return loss_fn

print("✅ Focal Loss (α=0.25, γ=2.0) loaded — replaces standard cross-entropy")

# =============================
# Cosine Annealing LR Schedule with Warmup
# =============================

def cosine_annealing_with_warmup(epoch, total_epochs=70, warmup_epochs=5,
                                  max_lr=1e-3, min_lr=1e-6):
    """Cosine annealing schedule with linear warmup."""
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))

# =============================
# NEW FEATURE: Cross-Attention Fusion Block
# =============================

def build_model(fine_tune_from=0):
    """Build dual-input model with Cross-Attention Fusion.
    
    Instead of simple concatenation, the metadata branch produces learned
    attention weights that modulate which image features are important.
    This lets the model learn context-dependent visual attention:
    e.g., "for elderly patients, focus more on border irregularity features."
    
    Args:
        fine_tune_from: Layer index from which to unfreeze EfficientNet.
                        0 = all frozen, -1 = all unfrozen
    """
    # ── Image Branch — EfficientNetB4 ──
    base_model = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    # Freeze/unfreeze layers
    if fine_tune_from == 0:
        base_model.trainable = False
    elif fine_tune_from == -1:
        base_model.trainable = True
    else:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_from]:
            layer.trainable = False

    img_features = GlobalAveragePooling2D()(base_model.output)
    img_features = BatchNormalization()(img_features)
    img_features = Dense(512, activation="relu", name="img_dense_512")(img_features)
    img_features = Dropout(0.4)(img_features)
    img_features = Dense(256, activation="relu", name="img_dense_256")(img_features)
    img_features = Dropout(0.3)(img_features)

    # ── Metadata Branch ──
    meta_input = Input(shape=(NUM_META_FEATURES,), name="metadata_input")
    meta_features = Dense(128, activation="relu")(meta_input)
    meta_features = BatchNormalization()(meta_features)
    meta_features = Dense(64, activation="relu")(meta_features)
    meta_features = Dropout(0.3)(meta_features)
    meta_features = Dense(32, activation="relu")(meta_features)

    # ── NEW: Cross-Attention Fusion ──
    # Metadata generates attention weights to modulate image features
    # This replaces the simple Concatenate()
    attention_weights = Dense(256, activation="sigmoid", name="cross_attention")(meta_features)
    attended_img = Multiply(name="attention_fusion")([img_features, attention_weights])
    
    # Also keep the metadata path for direct contribution
    meta_dense = Dense(64, activation="relu", name="meta_bridge")(meta_features)
    
    # Combine attended image features with metadata
    combined = Concatenate(name="fusion_concat")([attended_img, meta_dense])
    combined = BatchNormalization()(combined)
    combined = Dense(256, activation="relu")(combined)
    combined = Dropout(0.4)(combined)
    combined = Dense(128, activation="relu")(combined)
    combined = Dropout(0.3)(combined)

    # Output with float32 for numerical stability
    predictions = Dense(NUM_CLASSES, activation="softmax", dtype="float32")(combined)

    model = Model(inputs=[base_model.input, meta_input], outputs=predictions)
    return model, base_model

print("✅ Cross-Attention Fusion model architecture loaded")

# =============================
# Phase 1: Train Head Only (frozen backbone)
# =============================

print("\n" + "="*60)
print("PHASE 1: Training classification head (backbone frozen)")
print("  + Focal Loss | CutMix/MixUp | Cross-Attention Fusion")
print("="*60)

model, base_model = build_model(fine_tune_from=0)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4, clipnorm=1.0),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=["accuracy"]
)

model.summary()

phase1_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True, mode="max"),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ModelCheckpoint("skin_cancer_model_v2.h5", monitor="val_accuracy",
                    save_best_only=True, mode="max", verbose=1),
]

history1 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=8,
    class_weight=class_weight_dict,
    callbacks=phase1_callbacks,
    workers=1,
    use_multiprocessing=False,
    max_queue_size=10,
)

print(f"\nPhase 1 Best Val Accuracy: {max(history1.history['val_accuracy']):.4f}")

# =============================
# Phase 2: Fine-tune top 60% of backbone
# =============================

print("\n" + "="*60)
print("PHASE 2: Fine-tuning backbone (top 60% unfrozen)")
print("  + Focal Loss | CutMix/MixUp | Cross-Attention Fusion")
print("="*60)

total_layers = len(base_model.layers)
freeze_until = int(total_layers * 0.4)  # Freeze bottom 40%, unfreeze top 60%

base_model.trainable = True
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False

trainable_count = sum(1 for layer in base_model.layers if layer.trainable)
print(f"EfficientNetB4: {total_layers} total layers, {trainable_count} trainable (unfrozen)")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=focal_loss(alpha=0.25, gamma=2.0),
    metrics=["accuracy"]
)

phase2_callbacks = [
    EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True, mode="max"),
    ModelCheckpoint("skin_cancer_model_v2.h5", monitor="val_accuracy",
                    save_best_only=True, mode="max", verbose=1),
    LearningRateScheduler(
        lambda epoch: cosine_annealing_with_warmup(
            epoch, total_epochs=20, warmup_epochs=2, max_lr=1e-4, min_lr=1e-6
        )
    ),
]

# Recreate generators (they might have been exhausted)
train_gen = dual_input_generator(train_df, augment=True, shuffle=True, apply_cutmix=True)
val_gen = dual_input_generator(val_df, augment=False, shuffle=False, apply_cutmix=False)

history2 = model.fit(
    train_gen,
    steps_per_epoch=train_steps,
    validation_data=val_gen,
    validation_steps=val_steps,
    epochs=20,
    class_weight=class_weight_dict,
    callbacks=phase2_callbacks,
    workers=1,
    use_multiprocessing=False,
    max_queue_size=10,
)

print(f"\nPhase 2 Best Val Accuracy: {max(history2.history['val_accuracy']):.4f}")

# =============================
# Evaluate Model
# =============================

print("\n" + "="*60)
print("EVALUATION")
print("="*60)

test_gen = dual_input_generator(test_df, augment=False, shuffle=False)
test_steps = len(test_df) // BATCH_SIZE
loss, accuracy = model.evaluate(test_gen, steps=test_steps)
print(f"\n🎯 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

if accuracy >= 0.95:
    print("✅ TARGET ACHIEVED: Model accuracy ≥ 95%!")
elif accuracy >= 0.90:
    print("🟡 GOOD: Model accuracy ≥ 90%. Consider more training or data.")
else:
    print("🔴 BELOW TARGET: Consider adjusting hyperparameters or adding more data.")

# =============================
# Save Artifacts
# =============================

model.save("skin_cancer_model_v2.h5")

# Save class labels and metadata column info
artifacts = {
    "classes": CLASSES,
    "class_map": CLASS_MAP,
    "metadata_cols": METADATA_COLS,
    "num_meta_features": NUM_META_FEATURES,
    "img_size": IMG_SIZE,
    "model_backbone": "EfficientNetB4",
    "training_phases": 2,
    "v6_features": {
        "focal_loss": {"alpha": 0.25, "gamma": 2.0},
        "cutmix_mixup": True,
        "cross_attention_fusion": True,
        "hair_removal": "available_at_inference"
    }
}
with open("model_artifacts.json", "w") as f:
    json.dump(artifacts, f, indent=2)

# Save combined training history
hist1 = pd.DataFrame(history1.history)
hist2 = pd.DataFrame(history2.history)
hist_combined = pd.concat([hist1, hist2], ignore_index=True)
hist_combined.to_csv("training_history.csv", index=False)

print("\n✅ Model and artifacts saved successfully!")
print(f"   - Model: skin_cancer_model_v2.h5")
print(f"   - Artifacts: model_artifacts.json")
print(f"   - History: training_history.csv")
print(f"\n🆕 v6.0 Training Features:")
print(f"   ✅ Focal Loss (α=0.25, γ=2.0)")
print(f"   ✅ CutMix/MixUp Augmentation (50% prob)")
print(f"   ✅ Cross-Attention Fusion (metadata → image attention)")
print(f"   ✅ Hair Removal (DullRazor, available for inference)")
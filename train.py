import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

# =============================
# Dataset Paths
# =============================

metadata_path = "archive/skin-cancer-mnist-ham10000/HAM10000_metadata.csv"

image_dir1 = "archive/skin-cancer-mnist-ham10000/HAM10000_images_part_1"
image_dir2 = "archive/skin-cancer-mnist-ham10000/HAM10000_images_part_2"

# =============================
# Load Metadata
# =============================

df = pd.read_csv(metadata_path)

# Convert labels to string (required for binary classification)
df["label"] = df["dx"].apply(lambda x: "malignant" if x == "mel" else "benign")

# =============================
# Get Image Path
# =============================

def get_image_path(image_id):
    path1 = os.path.join(image_dir1, image_id + ".jpg")
    path2 = os.path.join(image_dir2, image_id + ".jpg")

    if os.path.exists(path1):
        return path1
    else:
        return path2

df["path"] = df["image_id"].apply(get_image_path)

# =============================
# Train / Val / Test Split
# =============================

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# =============================
# Image Generators
# =============================

IMG_SIZE = 224
BATCH_SIZE = 32

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

val_data = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_data = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="path",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

# =============================
# Load MobileNetV2 Model
# =============================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Freeze base model
for layer in base_model.layers:
    layer.trainable = False

# =============================
# Custom Layers
# =============================

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# =============================
# Compile Model
# =============================

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =============================
# Train Model
# =============================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# =============================
# Evaluate Model
# =============================

loss, accuracy = model.evaluate(test_data)

print("Test Accuracy:", accuracy)

# =============================
# Save Model
# =============================

model.save("skin_cancer_model.h5")

print("Model saved successfully!")
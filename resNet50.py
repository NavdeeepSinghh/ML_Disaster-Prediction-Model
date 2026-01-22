import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# -------------------------------
# 1. CONFIGURATION
# -------------------------------
BASE_DIR = "/Users/navdeepsingh/Desktop/ML_Disaster_Prediction/Comprehensive Disaster Dataset(CDD)/JPEG_Converted"
MODEL_SAVE_PATH = "resnet50_disaster_classifier.keras"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# ✅ Reduced Epochs
INITIAL_EPOCHS = 2
FINE_TUNE_EPOCHS = 2

# -------------------------------
# 2. VERIFY & LOAD DATASETS
# -------------------------------
if not os.path.exists(BASE_DIR):
    raise FileNotFoundError(f"❌ Dataset not found: {BASE_DIR}")
else:
    print(f"✅ Dataset found: {BASE_DIR}")

train_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.2,
    subset="training",
    seed=SEED,
    shuffle=True,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    BASE_DIR,
    validation_split=0.2,
    subset="validation",
    seed=SEED,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_ds.class_names
print("✅ Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)

# -------------------------------
# 3. DATA AUGMENTATION
# -------------------------------
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# -------------------------------
# 4. BUILD MODEL — ResNet50 Backbone
# -------------------------------
base_model = ResNet50(input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
base_model.trainable = False  # Freeze initially

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.4)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = keras.Model(inputs, outputs)

# -------------------------------
# 5. INITIAL TRAINING
# -------------------------------
model.compile(optimizer=keras.optimizers.Adam(0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    keras.callbacks.ModelCheckpoint("best_resnet50_model.keras", save_best_only=True)
]

print("\n🚀 Training Phase 1: Frozen ResNet50")
history = model.fit(train_ds, validation_data=val_ds, epochs=INITIAL_EPOCHS, callbacks=callbacks)

# -------------------------------
# 6. FINE-TUNING — Unfreeze Later Layers
# -------------------------------
base_model.trainable = True
fine_tune_at = 100

for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("\n🚀 Training Phase 2: Fine-Tuning (Unfreezing Top Layers)")
total_epochs = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks
)

# -------------------------------
# 7. PERFORMANCE VISUALIZATION
# -------------------------------
acc = history.history["accuracy"] + history_fine.history["accuracy"]
val_acc = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
loss = history.history["loss"] + history_fine.history["loss"]
val_loss = history.history["val_loss"] + history_fine.history["val_loss"]

plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.axvline(x=INITIAL_EPOCHS - 1, color="red", linestyle="--", label="Fine-tuning Starts")
plt.legend()
plt.title("Accuracy")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(loss, label="Train Loss")
plt.plot(val_loss, label="Val Loss")
plt.axvline(x=INITIAL_EPOCHS - 1, color="red", linestyle="--", label="Fine-tuning Starts")
plt.legend()
plt.title("Loss")

plt.show()

# -------------------------------
# 8. SAVE FINAL MODEL
# -------------------------------
model.save(MODEL_SAVE_PATH)
print("\n✅ Saved final model:", MODEL_SAVE_PATH)
print("✅ Best model saved:", "best_resnet50_model.keras")

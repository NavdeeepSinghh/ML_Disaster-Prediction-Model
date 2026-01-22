import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# ====================================================================
# SETTINGS
# ====================================================================

# --- Path to your main project folder ---
folder_path = "/Users/navdeepsingh/Desktop/ML_Disaster_Prediction"

# --- Name of your trained model file ---
model_filename = "disaster_classifier_improved.h5"

# --- Name of the image you want to test ---
# Using the new, simplified filename to avoid errors.
image_filename = "05_01_0026.jpg"

# --------------------------------------------------------------------
# The script will construct the full paths from the settings above.
model_path = os.path.join(folder_path, model_filename)
local_image_path = os.path.join(folder_path, image_filename)

# This list MUST match the class order from your training output
class_names = [
    'Damaged_Infrastructure',
    'Fire_Disaster',
    'Human_Damage',
    'Land_Disaster',
    'Non_Damage',
    'Water_Disaster'
]
# ====================================================================


# === STEP 1: VERIFY FILES EXIST ===
if not os.path.exists(model_path):
    print(f"❌ Error: Model file not found at {model_path}")
    exit()

if not os.path.exists(local_image_path):
    print(f"❌ Error: Image file not found at {local_image_path}")
    exit()

print(f"✅ Model found: {model_path}")
print(f"✅ Image found: {local_image_path}")


# === STEP 2: LOAD THE TRAINED MODEL ===
print("\nLoading trained model...")
model = tf.keras.models.load_model(model_path, compile=False)
print("✅ Model loaded successfully.")


# === STEP 3: LOAD AND PREPROCESS THE IMAGE ===
print("Processing image...")
img = image.load_img(local_image_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
print("✅ Image preprocessed.")


# === STEP 4: RUN PREDICTION ===
print("Running prediction...")
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class_index = np.argmax(score)
predicted_class = class_names[predicted_class_index]
confidence = 100 * np.max(score)
print("✅ Prediction complete.")


# === FINAL RESULT ===
print("\n" + "="*40)
print("           PREDICTION RESULT")
print("="*40)
print(f"Image File:      {image_filename}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence:      {confidence:.2f}%")
print("\n--- Full Scores ---")
for i, class_name in enumerate(class_names):
    print(f"{class_name:<25}: {100 * score[i]:.2f}%")
print("="*40)
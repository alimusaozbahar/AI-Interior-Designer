import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. DATA AUGMENTATION
# =========================

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.20
)

# =========================
# 2. TRAIN DATA
# =========================

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="training",
    shuffle=True
)

# =========================
# 3. VALIDATION DATA
# =========================

val_data = datagen.flow_from_directory(
    "dataset",
    target_size=(224, 224),
    batch_size=16,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

# =========================
# 4. BASE MODEL
# =========================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

# Transfer Learning
base_model.trainable = False

# =========================
# 5. MODEL
# =========================

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.4),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(5, activation="softmax")
])

# =========================
# 6. COMPILE
# =========================

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# =========================
# 7. EARLY STOPPING
# =========================

early_stop = EarlyStopping(
    monitor="val_accuracy",
    patience=5,
    restore_best_weights=True
)

# =========================
# 8. TRAIN
# =========================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=15,
    callbacks=[early_stop]
)

# =========================
# 9. SAVE MODEL
# =========================

model.save("model.h5")
model.save("interior_style_model.keras")

print("Model başarıyla kaydedildi.")
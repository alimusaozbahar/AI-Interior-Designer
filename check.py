import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = tf.keras.models.load_model("model.h5")

# Sınıf isimleri
classes = [
    "industrial",
    "modern",
    "scandinavian",
    "rustic",
    "traditional"
]

# Test resmi
img_path = "test_images/oda1.jpg"

# Resmi oku
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)

# Normalize et
img_array = img_array / 255.0

# Batch boyutu ekle
img_array = np.expand_dims(img_array, axis=0)

# Tahmin yap
prediction = model.predict(img_array)[0]

print("\n--- SONUÇLAR ---\n")

for cls, score in zip(classes, prediction):
    print(f"{cls}: %{score*100:.2f}")

best_index = np.argmax(prediction)

print("\nEn Baskın Stil:")
print(classes[best_index])
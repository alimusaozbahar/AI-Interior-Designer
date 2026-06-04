import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Modeli yükle
model = tf.keras.models.load_model("interior_style_model.keras")

# Sınıf isimleri
class_names = [
    "industrial",
    "rustic",
    "modern",
    "scandinavian",
    "traditional"
]

# Test fotoğrafı
img_path = "test.jpg"

img = image.load_img(img_path, target_size=(224,224))
img_array = image.img_to_array(img)

img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)[0]

print("\nSONUÇLAR\n")

for i in range(len(class_names)):
    print(
        f"{class_names[i]} : %{prediction[i]*100:.2f}"
    )

best_class = class_names[np.argmax(prediction)]

print("\nTahmin edilen stil:")
print(best_class)
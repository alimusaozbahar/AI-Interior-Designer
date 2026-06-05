import streamlit as st
import tensorflow as tf
import numpy as np
import os
import random

from PIL import Image
from tensorflow.keras.preprocessing import image

st.set_page_config(
    page_title="AI Interior Designer",
    layout="wide"
)

st.title("🏠 AI Interior Designer")

# =========================
# MODEL YÜKLE
# =========================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

classes = [
    "industrial",
    "modern",
    "scandinavian",
    "rustic",
    "traditional"
]

# =========================
# FOTOĞRAF YÜKLE
# =========================

uploaded_file = st.file_uploader(
    "Oda fotoğrafı yükle",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    img = Image.open(uploaded_file)
    
    # MÜHENDİSLİK DÜZELTMESİ 1: RGBA (4 kanal) görseli RGB (3 kanal) moduna güvenle çeviriyoruz
    if img.mode != "RGB":
        img = img.convert("RGB")

    st.image(
        img,
        caption="Yüklenen Oda",
        use_container_width=True
    )

    # =========================
    # TAHMİN
    # =========================

    img_resized = img.resize((224, 224))

    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]

    st.subheader("📊 Stil Analizi")

    for cls, score in zip(classes, prediction):
        st.write(f"{cls} : %{score * 100:.2f}")

    best_index = np.argmax(prediction)
    predicted_style = classes[best_index]

    st.success(
        f"En Baskın Stil: {predicted_style}"
    )

    # =========================
    # GÜVEN ORANI
    # =========================

    confidence = float(prediction[best_index]) * 100

    st.info(
        f"Model bu tahmini %{confidence:.2f} güvenle yaptı."
    )

    st.progress(
        float(prediction[best_index])
    )

    # =========================
    # STİL SEÇİMİ
    # =========================

    selected_style = st.selectbox(
        "Hangi stile dönüştürmek istiyorsunuz?",
        classes
    )

    # =========================
    # TASARIM ÖNERİLERİ
    # =========================

    if st.button("Tasarım Önerileri Getir"):

        folder = f"dataset/{selected_style}"
        images = []

        # MÜHENDİSLİK DÜZELTMESİ 2: Sunucuda dataset klasörü yoksa uygulamanın çökmesini engelliyoruz
        if os.path.exists(folder):
            images = [
                os.path.join(folder, file)
                for file in os.listdir(folder)
                if file.lower().endswith(
                    (".jpg", ".jpeg", ".png")
                )
            ]
            random.shuffle(images)
        else:
            st.warning(f"⚠️ Sunucuda '{folder}' klasörü bulunamadı. Lütfen GitHub deponuzda bu klasörün ve örnek resimlerin olduğundan emin olun.")

        if len(images) > 0:
            st.subheader(
                f"🎨 {selected_style} Tasarım Önerileri"
            )

            cols = st.columns(3)

            for i in range(min(3, len(images))):
                with cols[i]:
                    st.markdown(
                        f"### Tasarım {i+1}"
                    )
                    st.image(
                        images[i],
                        use_container_width=True
                    )

            # =========================
            # ÖNCE / SONRA
            # =========================

            example_image = images[0]

            st.markdown("---")

            st.header(
                "🔄 Öncesi - Sonrası Karşılaştırması"
            )

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📸 Önce")
                st.image(
                    img,
                    use_container_width=True
                )
                st.caption(
                    "Kullanıcının yüklediği oda"
                )

            with col2:
                st.subheader("✨ Sonra")
                st.image(
                    example_image,
                    use_container_width=True
                )
                st.caption(
                    f"AI önerisi: {selected_style} tasarım"
                )
import streamlit as st
import cv2
import numpy as np
from utils import analyze_cake_image
from PIL import Image

st.set_page_config(page_title="Small Cake Analyzer", layout="wide")
st.title("🍰 IEC 60350-1 Small Cake Shade Analyzer")

uploaded_file = st.file_uploader("📤 Küçük kek test görselini yükle (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.subheader("🔍 Yüklenen Görsel:")
    st.image(image, channels="BGR", use_column_width=True)

    if st.button("🚀 Analizi Başlat"):
        with st.spinner("Analiz yapılıyor..."):
            output, mean_ry_values, ry_plot_fig, heatmap_fig = analyze_cake_image(image)

        st.success("✅ Analiz tamamlandı!")

        st.subheader("🖼️ İşlenmiş Görsel:")
        st.image(output, channels="BGR", use_column_width=True)

        st.subheader("📊 Ortalama Ry Dağılımı:")
        st.pyplot(ry_plot_fig)

        st.subheader("🌡️ Ry Isı Haritası:")
        st.pyplot(heatmap_fig)

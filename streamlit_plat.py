import streamlit as st
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
import os
import requests
from streamlit_lottie import st_lottie

# Supaya tidak ada error terkait library
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Inisialisasi PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # Ubah ke 'id' jika ingin menggunakan bahasa Indonesia

# Fungsi untuk memproses gambar dengan PaddleOCR
def process_image(image):
    # Konversi gambar dari PIL ke OpenCV
    image_np = np.array(image)
    image_cv2 = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Deteksi teks dengan PaddleOCR
    result = ocr.ocr(image_cv2, cls=True)

    # Buat bounding box di sekitar karakter yang terdeteksi
    for line in result:
        for text_info in line:
            box = np.array(text_info[0]).astype(np.int32)
            image_cv2 = cv2.polylines(image_cv2, [box], True, (0, 255, 0), 2)

    # Konversi kembali ke format yang dapat ditampilkan Streamlit
    image_with_boxes = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_with_boxes), result





# Fungsi untuk memuat animasi Lottie dari URL
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Streamlit app
st.set_page_config(page_title="Product Detection with AI", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")

# Tambahkan animasi Lottie di bagian header
lottie_ai = load_lottie_url("https://lottie.host/41057b06-2b9e-4f57-a52a-2c44badf538c/J9mPLA66u4.json")

# Pastikan animasi Lottie dimuat dengan benar
if lottie_ai:
    st_lottie(lottie_ai, speed=0.5, width=700, height=300, key="lottie_animation")  # Ukuran dikurangi
else:
    st.error("Failed to load animation.")


# Streamlit UI
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Number Plate Detection with OCR</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>project by : Richo</h3>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey;'>Upload an image of a vehicle's license plate to recognize the characters.</p>", unsafe_allow_html=True)



# Upload gambar
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Tampilkan gambar input di sebelah kiri dan hasil di sebelah kanan
    col1, col2 = st.columns(2)

    with col1:
        st.image(Image.open(uploaded_file), caption='Uploaded Image', use_column_width=True)

    if st.button('Process'):
        with col2:
            # Proses gambar dengan OCR
            output_image, ocr_result = process_image(Image.open(uploaded_file))

            # Tampilkan gambar hasil dengan bounding box
            st.image(output_image, caption='Processed Image with Bounding Boxes', use_column_width=True)

        # Tampilkan hasil OCR dengan tampilan yang rapi
        if ocr_result:
            # st.write("<div style='background-color: #f9f9f9; padding: 10px; border-radius: 5px; border: 2px solid #4CAF50;'>", unsafe_allow_html=True)
            for line_idx, line in enumerate(ocr_result):
                st.markdown(f"<h4 style='color: #4CAF50; text-align: center;'>Hasil Deteksi OCR :</h4>", unsafe_allow_html=True)
                for text_info in line:
                    st.markdown(f"<p style='font-size: 24px; color: black; font-weight: bold; text-align: center; background-color: #e0f7fa; padding: 5px; border-radius: 5px;'>{text_info[1][0]}</p>", unsafe_allow_html=True)
            st.write("</div>", unsafe_allow_html=True)
        else:
            st.write("No text detected.")
else:
    st.write("<p style='color: red;'>Please upload an image to start the OCR process.</p>", unsafe_allow_html=True)

# Footer
st.markdown("""<hr><p style='text-align: center;'>Developed My Application ❤️ by [Richo]</p>""", unsafe_allow_html=True)

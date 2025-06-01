
from pathlib import Path 
import PIL 
import streamlit as st 

import settings 
import helper 

st.set_page_config(
    page_title="Waste Classifier",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.title("ỨNG DỤNG NHẬN DIỆN RÁC THẢI")


st.sidebar.header("Chọn độ chính xác")

confidence = float(st.sidebar.slider("", settings.MIN_CONFIDENCE, settings.MAX_CONFIDENCE, settings.DEFAULT_CONFIDENCE))


model_path = Path(settings.BEST_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Chọn ảnh để nhận diện vật thể")
source_radio = settings.IMAGE


source_img = None

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:  
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_container_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_container_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")  
            st.error(ex)

    with col2:
        if source_img is None: 
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Nhận diện ảnh',
                     use_container_width=True)
        else:
            if st.sidebar.button('Nhận diện ảnh'):
                res = model.predict(uploaded_image,conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_container_width=True)




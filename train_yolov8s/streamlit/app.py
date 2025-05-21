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

st.title("DETECT RUBBISH YOLO")

st.sidebar.header("Set Confidence")

confidence = float(st.sidebar.slider("", settings.MIN_CONFIDENCE, settings.MAX_CONFIDENCE, settings.DEFAULT_CONFIDENCE, step = 0.01)) 

model_path = Path(settings.DETECTION_MODEL)

try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


st.sidebar.header("Choose Image to Detect Objects")
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
            st.image(default_detected_image_path, caption='Detected Image',
                     use_container_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                res = model.predict(source= uploaded_image, conf=0.2, iou =0.6, save=True)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_container_width=True)




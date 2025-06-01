from ultralytics import YOLO 
import streamlit as st 
import cv2 
import settings

def load_model(model_path):
    
    model = YOLO(model_path) 
    return model


def display_tracker_options(): 
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None): 
    
    image = cv2.resize(image, (720, int(720*(9/16)))) 

    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        res = model.predict(image, conf=conf)

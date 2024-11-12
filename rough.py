import streamlit as st
import sys
import os

# from openai import OpenAI

from PIL import Image
import gpt4all
import re
import json
import cv2
from ultralytics import YOLO
import numpy as np
import torch
from utils_cc import markdown_css, t2t_template, segmate



# from ultralytics import YOLO
# CSS to fully customize the text input, text area fields, headers, and write sections
st.markdown(
   markdown_css,
    unsafe_allow_html=True
)

# Check if model is already in session state
if 'model_phi' not in st.session_state:
    print("**************icme aaya")
    modelphi_path=  "C:\\Users\\Aditya\\.cache\\gpt4all\\Phi-3-mini-4k-instruct-q4.gguf" #"/Users/admin/Documents/FL/Phi-3-mini-4k-instruct-q4.gguf"
    st.session_state.model_phi = gpt4all.GPT4All(modelphi_path)
    st.sidebar.success("Text model successfully loaded and cached.")



# Show title and description in the main page
st.title("📄 Consistent Comic Generation")
# st.write("Upload an image below and provide a short scenario")

# Move input fields to sidebar
st.sidebar.title("ChronoCrafters")

# default_sce = st.sidebar.button("Load default scenario")

# # if default_sce:
# default_img_pth = "assets\input\harry1.jpg"
# face_image = Image.open(default_img_pth)
# resized_image = face_image.resize((200, 200))
# st.image(resized_image, caption="Uploaded Image (Resized to 200x200)")

# mask_image = segmate(face_image).convert('RGB')#Image.open("examples/ldh_mask.png").convert('RGB')
# st.image(mask_image, caption="Uploaded Image (Resized to 200x200)")

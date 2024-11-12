import streamlit as st
import sys
import os

# from openai import OpenAI
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
from PIL import Image
import gpt4all
import re
import json
import cv2
from ultralytics import YOLO
import numpy as np

from utils_cc import markdown_css, t2t_template, segmate, default_prompt_list



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

import torch

# Show title and description in the main page
st.title("📄 Consistent Comic Generation")
# st.write("Upload an image below and provide a short scenario")

# Move input fields to sidebar
st.sidebar.title("ChronoCrafters")

default_sce = st.sidebar.button("Load default scenario")

# if default_sce:
default_img_pth = "assets\input\harry1.jpg"
face_image = Image.open(default_img_pth)
resized_image = face_image.resize((200, 200))
st.image(resized_image, caption="Uploaded Image (Resized to 200x200)")
mask_image = segmate(face_image).convert('RGB')#Image.open("examples/ldh_mask.png").convert('RGB')
st.image(mask_image, caption="Uploaded Image (Resized to 200x200)")


sys.path.append('StoryMaker') # for ip_adapter StoryMaker\ip_adapter
sys.path.append('StoryMaker\ip_adapter') # for ip_adapter 

from StoryMaker.pipeline_sdxl_storymaker import StableDiffusionXLStoryMakerPipeline
import diffusers
from insightface.app import FaceAnalysis
from diffusers import UniPCMultistepScheduler

# prepare 'buffalo_l' under ./models
if 'app' not in st.session_state:
    app = FaceAnalysis(name='buffalo_l', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    st.session_state.app = app
    print("app loaded")

# prepare models under ./checkpoints
face_adapter = f'checkpoints/mask.bin'#f'/content/checkpoints/mask.bin' #f'./checkpoints/mask.bin'
image_encoder_path = 'laion/CLIP-ViT-H-14-laion2B-s32B-b79K'  #  from https://huggingface.co/laion/CLIP-ViT-H-14-laion2B-s32B-b79K

base_model = 'huaquan/YamerMIX_v11'  # from https://huggingface.co/huaquan/YamerMIX_v11

if 'pipe' not in st.session_state:
    pipe = StableDiffusionXLStoryMakerPipeline.from_pretrained(
        base_model,
        torch_dtype=torch.float16
    )
    pipe.cuda()

    # load adapter
    pipe.load_storymaker_adapter(image_encoder_path, face_adapter, scale=0.8, lora_scale=0.8)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    st.session_state.pipe = pipe
    print("pipe loaded")

# # Check if model is already in session state
# if 'pipe' not in st.session_state:
#     print("************** image gen model load")
    
#     pipe =  "C:\\Users\\Aditya\\.cache\\gpt4all\\Phi-3-mini-4k-instruct-q4.gguf" #"/Users/admin/Documents/FL/Phi-3-mini-4k-instruct-q4.gguf"
#     st.session_state.model_phi = gpt4all.GPT4All(modelphi_path)
#     st.sidebar.success("Text model successfully loaded and cached.")

model_phi = st.session_state.model_phi
app = st.session_state.app
pipe = st.session_state.pipe



# Input fields for character name and scenario in the sidebar
characters = st.sidebar.text_input("Enter Your Character Name")
scenario = st.sidebar.text_area(
    "Provide a scenario for the comic.",
    placeholder="Adventures of Ravi in his office...",
)

# Image upload in the sidebar
uploaded_image = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
 # Resize and display the uploaded image on the main page
if uploaded_image is not None:
    image = Image.open(uploaded_image)
    resized_image = image.resize((200, 200))
    st.sidebar.image(resized_image, caption="Uploaded Image (Resized to 200x200)")
ready_button = st.sidebar.button("Ready to See the Storyline")
word_count = len(scenario.split())

prompts_list = []
texts_list = []
if ready_button:
        # Only display the "Ready to See the Comic" button if the scenario has sufficient words
    if scenario.strip():  # Check if text has been entered in the scenario
        if word_count < 20:
            # Display custom error message with red color if word count is less than 20
            st.sidebar.markdown(
                '<div class="custom-error">The scenario must contain at least 20-25 words. Please add more details.</div>',
                unsafe_allow_html=True
        )
        else:
        
            # Displaying user input on the main page
            st.write(f"**Character**: {characters}")
            st.write(f"**Scenario**: {scenario}")
            
           
            template = t2t_template(characters,scenario)

            if scenario:
                print("----generating prompts--------")
                result =  model_phi.generate(template,temp=0.2,max_tokens=1000)
                # print(result)
                # st.write(result)
            
                print("------------------")
        # Define pattern to match each panel's details
                
                # Define pattern to match each panel's details - updated to handle multiple different formats
                pattern_1 = r'# Panel (\d+)\n[\s]*description: (.*?)\n[\s]*text: (.*?)(?=\n# end|\Z|# Panel)'
                pattern_2 = r'# Panel (\d+)\ndescription: (.*?)\ntext: (.*?)(?=\n# end|\Z|# Panel)'
                pattern_3 = r'# Panel (\d+)\ndescription: (.*?)\ntext: (.*?)\n# end'

                # Find all matches using the first regex pattern
                matches = re.findall(pattern_1, result, re.DOTALL)[:6]

                # If no matches found using the first pattern, try the alternate pattern
                if not matches:
                    matches = re.findall(pattern_2, result, re.DOTALL)[:6]

                # If still no matches found, try the third pattern
                if not matches:
                    matches = re.findall(pattern_3, result, re.DOTALL)[:6]

                # Create the desired dictionary structure
                panels = []
                for match in matches:
                    panel_number = f"Panel {match[0]}"
                    description = match[1].strip() if match[1] else ""
                    text = match[2].strip() if match[2] else ""
                    prompts_list = prompts_list + description
                    texts_list = texts_list + text
                    panels.append({panel_number: {'description': description, 'text': text}})

                # Print the result
                import json
                output = json.dumps(panels, indent=2)
                print(output)
                print(prompts_list)
                print(texts_list)
                # Parse the JSON string into a Python list of dictionaries
                panels_json = json.loads(output)

                # Streamlit Title
                st.title("Comic Panels")

                # Loop through each panel in the JSON and display in separate boxes (2 per row)

                for i in range(0, len(panels_json), 2):
                    # Create two columns for two panels per row
                    cols = st.columns(2)

                    # Loop over the two columns and populate with content if available
                    for idx, col in enumerate(cols):
                        if i + idx < len(panels_json):
                            panel = panels_json[i + idx]
                            panel_name = list(panel.keys())[0]  # Extract the panel name (e.g., "Panel 1")
                            panel_content = panel[panel_name]   # Extract the panel content dictionary

                            # Using the `col` to create a box-like display for each panel
                            with col:
                                st.markdown(f"### {panel_name}")  # Panel title
                                st.markdown(f"**Description**: {panel_content['description']}")  # Panel description
                                st.markdown(f"**Text**: *{panel_content['text']}*")  # Panel text in italics


img_ready_button = st.sidebar.button("Ready to See the Comic")
if img_ready_button:
    prompts_list = default_prompt_list
    print(prompts_list)
    print("-----------generating comic-------")
    face_image = Image.open(uploaded_image)
    # load an image and mask
    #face_image = Image.open("/content/drive/MyDrive/FMGAI/Project/input/adit.jpg").convert('RGB')#Image.open("/content/drive/MyDrive/FMGAI/Project/input/thor1.jpg").convert('RGB')
    mask_image = segmate(face_image).convert('RGB')#Image.open("examples/ldh_mask.png").convert('RGB')
    torch.cuda.empty_cache()
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face

    n_prompt = "bad quality, NSFW, cartoonish, low quality, ugly, disfigured, deformed"

    generator = torch.Generator(device='cuda').manual_seed(63)
    outputs = []
    for i,prompt in enumerate(prompts_list):
        output = pipe(
            image=face_image, mask_image=mask_image, face_info=face_info,
            prompt=prompt,
            negative_prompt=n_prompt,
            ip_adapter_scale=0.8, lora_scale=0.8,
            num_inference_steps=25,
            guidance_scale=7.5,
            height=640, width=320,
            generator=generator,
        ).images[0]
        # cv2_imshow(output)
        outputs.append(output)
        st.image(output)
        output.save(f'outputs/prompt_gpt_1_hry_{i}.jpg')

    # st.sidebar.image(resized_image, caption="Uploaded Image (Resized to 200x200)")
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np

markdown_css =  """
    <style>
    /* Set background color for the whole page */
    html, body {
        background-color: #f4f8e2; /* Dark blue color */
    }

    /* Set background color for the main content area */
    .stApp {
        background-color:#d8dec6; /* Dark blue color */
    }

    /* Set custom styles for text input and text area */
    input[type="text"], textarea {
        background-color: #030056 !important; /* Dark blue to match page */
        color: white !important; /* White text color */
        border: 2px solid white !important; /* White border */
        border-radius: 5px; /* Optional rounded corners */
    }

    /* Modify the input label text color to make it visible against the dark background */
    .stTextInput > label, .stTextArea > label {
        color: white !important; /* White text for the labels */
    }

    /* Adjust the focus effect for input and text area fields */
    input[type="text"]:focus, textarea:focus {
        outline: none; /* Remove default focus outline */
        border: 2px solid #FFFFFF; /* Keep white border on focus */
    }
    
    /* To ensure Streamlit specific classes are styled properly for textarea */
    .stTextArea textarea {
        background-color: #030056!important; /* Match background color */
        color: white !important; /* White text color */
        border: 2px solid white !important; /* White border */
    }

    /* Set the header (h1, h2, h3, etc.) text color to white */
    h1, h2, h3, h4, h5, h6 {
        color: #030056 !important; /* Make all headers white */
    }

    /* Set the paragraph (st.write) text color to white */
    p {
        color: #030056 !important; /* Make all paragraph text white */
    }
     /* Set sidebar background color and styling */
    [data-testid="stSidebar"] {
        background-color: #030056; /* Dark blue color for sidebar */
       
    }

    /* Centering sidebar content */
    [data-testid="stSidebar"] .css-1lcbmhc {
        display: flex;
        flex-direction: column;
        align-items: center; /* Center elements horizontally */
    }

    /* Set the text color for all elements in the sidebar */
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
        color: white !important; /* Make sidebar text white */
    }

    /* Style the sidebar title "ChronoCrafters" */
    [data-testid="stSidebar"] h1 {
        color: #65e8b4 !important; /* Light green color */
        font-style: italic;
        font-size: 40px;
        text-align: center;
        
       
    }

    /* Adding a vertical line between sidebar and main page */
    .css-1lcbmhc {
        border-right: 2px solid #65e8b4; /* Light green vertical line to divide sidebar and main content */
    }
    /* Styling the 'Ready to See the Comic' button */
    div.stButton > button {
        background-color: #65e8b4 !important; /* Green background */
        color: #030056 !important; /* Blue foreground (text) */
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: bold;
        cursor: pointer;
    }

    div.stButton > button:hover {
        background-color: #52b69a !important; /* Slightly darker green for hover effect */
        color: #ffffff !important; /* White text on hover */
    }
    .custom-error {
        color: red;
        font-size: 16px;
        font-weight: bold;
    }
    </style>
    """

def t2t_template(characters,scenario):
    
    return f"""


            You will be given a short scenario, and your task is to split it into 6 parts.
            Each part will represent a different cartoon panel.

            For each cartoon panel, provide the following details:

            1. Panel Description:

            Write a detailed description of the panel, consisting of words or short phrases separated by commas.

            Include:

            Characters: Describe all characters precisely in each panel (use detailed character descriptions instead of their names).

            Background: Describe the setting or backdrop for the panel.

            Character Postures and Expressions: Include details about character positioning, body language, and facial expressions.

            Important Notes:

            No Full Sentences: Use only words or short phrases, separated by commas.

            Avoid Repetition: Descriptions should be unique for each panel. Do not use the same description twice.

            2. Panel Text:

            Write the dialogue or text for the panel.

            Limit the text to two short sentences at most.

            Each sentence must start with the character's name.



            You have to follow below Example Output format:


                    # Panel 1
                    description: 
                    text:
                    # end

            Special Note:
            Make Sure only description and text should be present in each panel avoid any extra details.

            Use following characters amd short scenario for output:

            Character: {characters}
            Short Scenario: {scenario}

            Split the Scenario into 6 Panels:

            Create a description and text for each of the 6 panels based on the given scenario.

            Your output will include Exact 6 panels in total, each with a detailed description and dialogue.

            """

def segmate(image):
    #https://stackoverflow.com/questions/76168470/how-to-create-a-binary-mask-from-a-yolo8-segmentation-result
    print("Mask entered")
    model = YOLO('yolov8m-seg.pt')
    results = model.predict(source=image.copy(), save=True, save_txt=False, stream=True)
    for result in results:
        # get array results
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes.data.cpu().numpy()
        # extract classes
        clss = boxes[:, 5]
        # get indices of results where class is 0 (people in COCO)
        # people_indices = torch.where(clss == 0)
        people_indices = np.where(clss == 0)
        # use these indices to extract the relevant masks
        people_masks = masks[people_indices]
        # scale for visualizing results
        # people_mask = torch.any(people_masks, dim=0).int() * 255
        people_mask = np.any(people_masks, axis=0).astype(np.uint8) * 255
        # save to file
        # mask_image = Image.fromarray(people_mask.cpu().numpy())
        mask_image = Image.fromarray(people_mask)
        del model
        # torch.cuda.empty_cache()
        print("Mask generated")
    return mask_image
    
default_prompt_list = [
        "Billy standing at zoo entrance, excited expression, holding ticket, sunny day background",
        "Billy entering a colorful animal enclosure, wide-eyed wonderment, vibrant flora and fauna backdrop",
        "Closeup of Billy's face showing confusion, surrounded by various animal sounds in the background",
        "A group of animals looking curiously at Billy from behind bars or fences; some chirping birds visible above her head",
        "Billy attempting to mimic animal sounds with a comical expression of effort and slight frustration",
        "Animals playfully interacting around Billy, who is still trying but clearly not understanding; the sun sets in soft light behind her"
    ]
import streamlit as st
from openai import OpenAI
from PIL import Image

# model_phi = gpt4all.GPT4All("/assets/Phi-3-mini-4k-instruct-q4.gguf")
scenario = "Spending Rainy Day"
characters = "Sina"
template = f"""


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
```

```
# end

Use following characters amd short scenario for output:

Characters: {characters}
Short Scenario: {scenario}

Split the Scenario into 6 Panels:

Create a description and text for each of the 6 panels based on the given scenario.

Your output will include Exact 6 panels in total, each with a detailed description and dialogue.

"""



# Show title and description.
st.title("📄 Consistent Comic Generation")
st.write(
    "Upload an image below and provide a short scenario "
    #"To use this app, you need to provide an OpenAI API key, which you can get [here](https://platform.openai.com/account/api-keys). "
)

# Ask the user for a question via `st.text_area`.
scenario = st.text_area(
    "Provide a scenario for the comic.",
    placeholder="Adventures of Ravi in his office...",
    # disabled=not uploaded_file,
)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.info("Please add your OpenAI API key to continue.", icon="🗝️")
else:

    # Create an OpenAI client.
    client = OpenAI(api_key=openai_api_key)

    # Let the user upload a file via `st.file_uploader`.
    uploaded_file = st.file_uploader(
        "Upload a document (.txt or .md)", type=("txt", "md")
    )

    # Ask the user for a question via `st.text_area`.
    question = st.text_area(
        "Now ask a question about the document!",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )

    if uploaded_file and question:

        # Process the uploaded file and question.
        document = uploaded_file.read().decode()
        messages = [
            {
                "role": "user",
                "content": f"Here's a document: {document} \n\n---\n\n {question}",
            }
        ]

        # Generate an answer using the OpenAI API.
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            stream=True,
        )

        # Stream the response to the app using `st.write_stream`.
        st.write_stream(stream)

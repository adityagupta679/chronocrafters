
# Clone the repository to your google drive. 

#### Download the model files from https://drive.google.com/drive/folders/19ytYfIRX3PQkCXkaPiX3QKBLnzCFy3r9?usp=sharing and place in the same location as cloned repository. 

#### To run the model, run FINAL_CHRONOCRAFTERS.ipynb . This file runs best in Colab GPU T4. Change the working directory according to your drive location. Add your open api key where it is required, or modify code to use with your own LLM model for prompts.  Generates a gradio interface that allows you to enter character name, scenario and upload image of your character to get an output comic strip, which also gets saved in your drive. 

#### To give feedback and see the outputs of already generated 30 comics, run Feedback.ipynb. Uses normal cpu based colab environment. Generates a gradio interface you can share with others to view the already run examples using FINAL_CHRONOCRAFTERS.ipynb.  

#### To run evaluations on the model, after obtaining feedback, run Evaluation.ipynb . This gives us various scores. 

#### The outputs from Evaluation and Feedback, save the outputws in csv file in your drive folder. 


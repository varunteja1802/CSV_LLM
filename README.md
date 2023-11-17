#Student Details 

1.Kalakoti Varun Teja 
2.Woxsen University 
3.ID: 20U0101001 
4.email:varunteja.kalakoti_2024@woxsen.edu.in 
5.M:+918333890029


# CSV_LLM
tensorgo_assignment 

# Assignment Repository
Welcome to the assignment repository! This repository contains two different approaches for answering questions based on a given CSV file. Please follow the instructions below to understand and utilize each approach.

## Approach 1(app.py): Pretrained Llama 2 7b Model
In this approach, a pretrained Llama 2 7b model is used to answer questions based on the provided CSV file. Due to the lack of computational support, accuracy may not be optimal. To use this approach, follow these steps:

Upload your CSV file using the Streamlit web application.
Ask questions related to the data.
Note that the answers may not be as accurate due to the limitations of the pretrained model.
## Approach 2(main.py): OpenAI API Key
In the second approach, the OpenAI API key is utilized to generate answers to questions based on the uploaded CSV file. This approach may provide more accurate results compared to the pretrained model. Follow these steps to use this approach:

Upload your CSV file using the Streamlit web application.
Ask questions related to the data.
Note the improved accuracy compared to the pretrained model.
### Important Note:
The current repository does not include the third approach, which involves converting the CSV to text format using df2str and fine-tuning the Llama 2 model with custom text data. As this approach would consume lots of time and need lots of computational support I have not used this but a deep literature review has been done and if this approach would be completed then the accuracy would be enhanced.

# Getting Started
To run the Streamlit web application locally, make sure you have the necessary dependencies installed. You can do this by running: 
first git clone this repository and then perform the below tasks 
pip install -r requirements.txt

streamlit run app.py (for approach 1)
streamlit run main.py (for approach 2)


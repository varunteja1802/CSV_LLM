import streamlit as st 
from streamlit_chat import message
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DB_FAISS_PATH = 'vectorstore/db_faiss'

# Function to load the language model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="E:/CSV_LLM/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

# Function to perform basic statistical analysis
def perform_statistical_analysis(data):
    st.write("### Basic Statistical Analysis")
    
    # Display basic statistics
    st.write("#### Descriptive Statistics:")
    st.write(data.describe())
    
    # Calculate correlation matrix
    st.write("#### Correlation Matrix:")
    correlation_matrix = data.corr()
    st.write(correlation_matrix)
    
    # Generate correlation heatmap
    st.write("#### Correlation Heatmap:")
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    st.pyplot()

# Function to generate plots
def generate_plots(data):
    st.write("### Data Plots")
    
    # Histograms for numeric columns
    st.write("#### Histograms:")
    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            st.write(f"**{column} Histogram**")
            st.hist_chart(data[column])

    # Scatter plot for selected numeric columns
    st.write("#### Scatter Plot:")
    x_column = st.selectbox("Select X-axis:", data.columns)
    y_column = st.selectbox("Select Y-axis:", data.columns)
    if pd.api.types.is_numeric_dtype(data[x_column]) and pd.api.types.is_numeric_dtype(data[y_column]):
        st.write(f"**Scatter Plot between {x_column} and {y_column}**")
        st.scatter_chart(data[[x_column, y_column]])


# Streamlit App
st.title("TensorGo Assignment")
uploaded_file = st.sidebar.file_uploader("Upload your Data", type="csv")


if uploaded_file:
    # use tempfile because CSVLoader only accepts a file_path
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()

    # Check if data is loaded
    if 'data' not in st.session_state:
        st.session_state['data'] = data

    # Statistical analysis and plotting
    if 'user_input' in st.session_state and st.session_state['user_input'].lower() == "statistical analysis":
        perform_statistical_analysis(st.session_state['data'])
    elif 'user_input' in st.session_state and st.session_state['user_input'].lower() == "generate plots":
        generate_plots(st.session_state['data'])
    else:
        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
        db = FAISS.from_documents(st.session_state['data'], embeddings)
        db.save_local(DB_FAISS_PATH)
        llm = load_llm()
        chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())

        def conversational_chat(query):
            result = chain({"question": query, "chat_history": st.session_state['history']})
            st.session_state['history'].append((query, result["answer"]))
            return result["answer"]

        if 'history' not in st.session_state:
            st.session_state['history'] = []

        if 'generated' not in st.session_state:
            st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

        if 'past' not in st.session_state:
            st.session_state['past'] = ["Hey ! 👋"]

        # container for the chat history
        response_container = st.container()
        # container for the user's text input
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                st.session_state['user_input'] = st.text_input("Query:", placeholder="Talk to your csv data here (:", key='input')
                submit_button = st.form_submit_button(label='Send')
                
            if submit_button and st.session_state['user_input']:
                output = conversational_chat(st.session_state['user_input'])
                st.session_state['past'].append(st.session_state['user_input'])
                st.session_state['generated'].append(output)

        if st.session_state['generated']:
            with response_container:
                for i in range(len(st.session_state['generated'])):
                    message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                    message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

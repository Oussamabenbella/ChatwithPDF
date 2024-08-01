import streamlit as st
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import pipeline
import torch
import textwrap
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from constants import CHROMA_SETTINGS
from streamlit_chat import message
import io
import pandas as pd

st.set_page_config(layout="wide")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

checkpoint = "MBZUAI/LaMini-T5-738M"
print(f"Checkpoint path: {checkpoint}")  # Add this line for debugging
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
base_model.to(device)

persist_directory = "db"

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
                texts = text_splitter.split_documents(documents)
                # Create embeddings here
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                # Create vector store here
                db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
                db.persist()
                db = None

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        device=device
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']

    return answer

def get_file_size(file):
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    return file_size

@st.cache_data
# Function to display the PDF of a given file
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

# Generate an Excel file from the provided data
def generate_excel(data):
    output = io.BytesIO()

    # Print data for debugging
    print("Data received for Excel generation:", data)

    try:
        if isinstance(data, str):
            # Split the string into rows and columns assuming a simple structure
            rows = [row.split() for row in data.split('\n')]
            df = pd.DataFrame(rows[1:], columns=rows[0])
        elif isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError("Unsupported data format for Excel generation")

    except ValueError as e:
        st.error(f"Error creating DataFrame: {e}")
        return None

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    
    # Get the Excel data
    excel_data = output.getvalue()
    return excel_data

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i], key=str(i))

def main():
    st.markdown("<h1 style='text-align: center; color: blue;'>Chat with your PDF 📄 </h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; color: grey;'>By Oussama</h3>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details = {
            "Filename": uploaded_file.name,
            "File size": get_file_size(uploaded_file)
        }
        filepath = os.path.join("docs", uploaded_file.name)
        
        # Ensure the docs directory exists
        os.makedirs("docs", exist_ok=True)
        
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("<h4 style color:black;'>File details</h4>", unsafe_allow_html=True)
            st.json(file_details)
            st.markdown("<h4 style color:black;'>File preview</h4>", unsafe_allow_html=True)
            displayPDF(filepath)

        with col2:
            with st.spinner('Embeddings are in process...'):
                data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h4 style color:black;'>Chat Here</h4>", unsafe_allow_html=True)

            user_input = st.text_input("", key="input")

            # Initialize session state for generated responses and past messages
            if "generated" not in st.session_state:
                st.session_state["generated"] = ["I am ready to help you"]
            if "past" not in st.session_state:
                st.session_state["past"] = ["Hey there!"]

            # Search the database for a response based on user input and update session state
            if user_input:
                if "excel" in user_input.lower() or "tableau" in user_input.lower():
                    answer = process_answer(user_input)
                    st.session_state["past"].append(user_input)
                    st.session_state["generated"].append("Here's your response in Excel format:")

                    excel_data = generate_excel(answer)
                    if excel_data:
                        st.download_button(
                            label="Download Excel",
                            data=excel_data,
                            file_name="response.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                else:
                    answer = process_answer(user_input)
                    st.session_state["past"].append(user_input)
                    st.session_state["generated"].append(answer)

            # Display conversation history using Streamlit messages
            if st.session_state["generated"]:
                display_conversation(st.session_state)

if __name__ == "__main__":
    main()

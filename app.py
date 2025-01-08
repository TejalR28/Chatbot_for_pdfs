#import libraries
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    # Initialize an empty string to hold the extracted text
    text = ""
    # Loop through each uploaded PDF file
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        # Extract text from each page of the PDF
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Initialize text splitter to divide text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Initialize embeddings using a specific model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create FAISS index from text chunks
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the FAISS index locally
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    # Define a prompt template for answering questions based on the provided context
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize the conversational model with specific parameters
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0.3)

    # Create a prompt object using the template and input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    # Load the question-answering chain with the conversational model and prompt
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question, folder_path="faiss_index", model="models/embedding-001", allow_dangerous_deserialization=True):
    try:
        # Initialize embeddings
        embeddings = GoogleGenerativeAIEmbeddings(model=model)

        # Load FAISS index with the specified parameters
        new_db = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=allow_dangerous_deserialization  # Pass the new parameter
        )

        # Perform similarity search with the user's question
        docs = new_db.similarity_search(user_question)

        # Retrieve the conversational chain
        chain = get_conversational_chain()

        # Get the response from the chain
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )

        # Output the response
        print(response)
        st.write("Reply: ", response["output_text"])

    except ValueError as e:
        print(f"Error loading FAISS index: {e}")
        st.write(f"Error loading FAISS index: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        st.write(f"An unexpected error occurred: {e}")

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini")

    # Get user input for the question
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    # Sidebar for uploading PDF files and processing them
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from the uploaded PDF files
                raw_text = get_pdf_text(pdf_docs)
                # Split the extracted text into chunks
                text_chunks = get_text_chunks(raw_text)
                # Create and save the vector store from the text chunks
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()

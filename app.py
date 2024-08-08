import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page_num, page in enumerate(pdf_reader.pages):
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text
            else:
                print(f"No text found on page {page_num} of {pdf.name}")
    if not text:
        print("No text extracted from the PDF files.")
    else:
        print(f"Extracted text (first 1000 characters): {text[:1000]}")
    return text

def get_text_chunks(text):
    if not text.strip():  # Check if the text is empty or contains only whitespace
        print("The text is empty or not valid.")
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    if not chunks:
        print("No text chunks created. The text might be too short or the splitter configuration might need adjustment.")
    else:
        print(f"Number of chunks created: {len(chunks)}")
        print(f"Sample chunk (first 500 characters): {chunks[0][:500]}")
    return chunks

def get_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No text chunks available for embedding.")
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    except Exception as e:
        print(f"Error generating FAISS index: {e}")
        if text_chunks:
            print(f"Sample text chunk: {text_chunks[0]}")
        raise e
    
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.6)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    allow_dangerous_deserialization = True
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat PDF", layout="wide")
    st.header("ChatPDF: Chat with your PDF-Files📁🗃️")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract and debug text from PDFs
                raw_text = get_pdf_text(pdf_docs)
                
                # Split text into chunks and debug
                text_chunks = get_text_chunks(raw_text)
                
                # Create vector store and handle potential issues
                try:
                    get_vector_store(text_chunks)
                    st.success("Done")
                except ValueError as e:
                    st.error(f"Error: {e}")
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()

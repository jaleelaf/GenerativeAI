import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS


OPENAI_API_KEY = "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"
#upload PDF files

st.header("My First Chatbot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a PDF file and start asking questions",type="pdf")

#Extract the text
if file is not None:
    with st.spinner("Loading PDF..."):
        reader = PdfReader(file)
        text=""
        for page in reader.pages:
            text+=page.extract_text()
            #st.write(text)
            #Break into chunks
        text_splitter = RecursiveCharacterTextSplitter(
                separators ="\n",
                chunk_size=1000,
                chunk_overlap=150,
                length_function=len
            )
        chunks = text_splitter.split_text(text)
            #st.write(chunks)

        #generate embedding
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        #initialize and create a vector store using Faiss
        vector_stores= FAISS.from_texts(chunks,embeddings)

        #get user questions
        user_question = st.text_input("Type your question")
        #do similarity search
        if user_question:
            match = vector_stores.similarity_search(user_question)
            st.write(match)





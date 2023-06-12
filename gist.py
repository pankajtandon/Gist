
# from scipy import spatial
# import ast  # for converting embeddings saved as strings back to arrays
# import openai  # for calling the OpenAI API
# import pandas as pd  # for storing text and embeddings data
# import tiktoken  # for counting tokens
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

PAGE_CONFIG = {"page_title": "Hello baby!", "page_icon": "smiley", "layout": "centered"}
st.set_page_config(**PAGE_CONFIG)
st.title("Welcome to our world!")
st.subheader("We are head over heels!")
pdf = st.file_uploader("Upload your PDF", type = "PDF")

if pdf is not None:
    pdf_reader = PdfReader(pdf)
    content = ""
    for page in pdf_reader.pages:
        content += page.extract_text()
    st.write("====Content====")
    st.write(content)

    # Chunk out the file
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size= 160, chunk_overlap = 15, length_function= len)
    chunks = text_splitter.split_text(content)

    st.write("====Chunks====")
    st.write(chunks)

    #Ask the question
    question = st.text_input("Ask me something about the PDF that you just uploaded:")
    if question:
        embeddings = OpenAIEmbeddings()

        # These are the vectorized chunks:
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # Docs are those vectors that are similar to the vectors in the knowledge base.
        docs = knowledge_base.similarity_search(question)

        if docs is not None:
            st.write("These are the related chunks:")
            for doc in docs:
                st.write(doc)
            
            # Forward the related chunks to the LLM with the query as a prompt
            llm = OpenAI()
        
            chain = load_qa_chain(llm, chain_type = "stuff")
            with get_openai_callback() as cb:
                response = chain.run(question = question, input_documents = docs)
                st.write("Cost of query:")
                st.write(cb)

            st.write(response)
        else:
            st.write("No match on the chunks!")


# EMBEDDING_MODEL = "text-embedding-ada-002"
# GPT_MODEL = "gpt-3.5-turbo"

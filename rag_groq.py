from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.chains import RetrievalQA
import os
import streamlit as st
from dotenv import load_dotenv
import time
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")

llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY,temperature=1,model="gpt-3.5-turbo")

prompt_template = """
You are a helpful assistant that helps people find information. 
Use the following context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Context:
{context}
Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

def create_vector_store():
    embeddings=OpenAIEmbeddings()
    loader=PyPDFDirectoryLoader("./") # data ingestion
    doc=loader.load() # document loading
    if not doc:
        st.error("No PDF files found in the current folder.")
        st.stop()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_doc=text_splitter.split_documents(doc)
    vector_store=FAISS.from_documents(final_doc,embeddings) 
    return vector_store
if "vector" not in st.session_state:
    try:
        st.session_state.vector = create_vector_store()
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
st.title("📄 PDF Research Paper QA")    
user_prompt=st.text_input("enter your query here from the research paper")
if st.button("Search") and user_prompt:
    with st.spinner("Fetching answer from the research papers..."):
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vector.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            chain_type="stuff",  
            chain_type_kwargs={"prompt": prompt},
        )
        start_time = time.process_time()
        response = qa_chain.run(user_prompt)
        elapsed = time.process_time() - start_time

        st.success(f"Answer fetched in {elapsed:.2f} seconds:")
        st.write(response)
        docs=retriever.get_relevant_documents(user_prompt)
        with st.expander("Document similarity search"):
            for i, doc in enumerate(docs):
                st.write(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.write("------------------------------------------------")
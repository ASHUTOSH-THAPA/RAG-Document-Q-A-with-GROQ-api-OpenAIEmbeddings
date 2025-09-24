from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
 # load groq api key from .env file
os.environ["OPENAI_API_KEY"]=os.getenv("openai_api_key")
os.environ["GROQ_API_KEY"]=os.getenv("groq_api_key")
groq_api_key=os.getenv("groq_api_key")

llm=ChatGroq(groq_api_key=groq_api_key,temperature=1,model="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    template="""You are a helpful assistant that helps people find information. 
    Use the following context to answer the question at the end. If you don't know the answer, 
    just say that you don't know, don't try to make up an answer. " \
    Context: {context} Question: {question}""",
    input_variables=["context","question"]
)
def create_vector_store():
    if "vector" not in st.session_state:
        st.session_state.embeddings=OpenAIEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("/research_papers") # data ingestion
        st.session_state.doc=st.session_state.loader.load() # document loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        st.session_state.final_doc=st.session_state.text_splitter.split_documents(st.session_state.doc)
        st.session_state.vector=FAISS.from_documents(st.session_state.final_doc,st.session_state.embeddings) 
user_prompt=st.text_input("enter your query here from the research paper")
if st.button("search"):
    create_vector_store()
    st.write("searching...")
import time
if user_prompt:
    with st.spinner("fetching answer from the research paper..."):
        time.sleep(2)
        document_chain=create_stuff_documents_chain(llm,prompt)
        retriever=st.session_state.vector.as_retriever()
        retriever_chain=create_retrieval_chain(retriever,document_chain)
        start=time.process_time()
        response=retriever_chain.invoke({"question":user_prompt})
        print(f"time taken to fetch answer: {time.process_time()-start}")
        st.write(response['answer'])

        with st.expander("document similarity search"):
            for i,doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("------------------------------------------------")



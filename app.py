import streamlit as st
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="Agentic AI Project")
st.title("Web Page Question Answering bot")


def get_docs_from_url(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    loader = WebBaseLoader(url, requests_kwargs={"headers": headers})
    docs = loader.load()
    return docs

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def create_vector_db(documents):
    embeddings = OpenAIEmbeddings()
    vectorstoredb = FAISS.from_documents(documents, embeddings)
    return vectorstoredb

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_chain(vectorstoredb):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    # Define the Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the following question based only on the provided context:
        <context>
        {context}
        </context>
        Question: {input}
        """
    )
    
    # Define the Retriever
    retriever = vectorstoredb.as_retriever()

    # MANUAL CHAIN CONSTRUCTION 
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


url_input = st.text_input("Hello, Enter the URL to process")

if st.button("Process URL"):
    if url_input:
        # Using st.status to show dynamic step-by-step progress
        with st.status("Processing Webpage...", expanded=True) as status:
            try:
                # Step 1: Scraping
                st.write("Fetching content from URL...")
                raw_docs = get_docs_from_url(url_input)
                
                if not raw_docs:
                    status.update(label="Error: No text found on webpage.", state="error")
                    st.stop()
                
                # Step 2: Splitting
                st.write("Splitting text into chunks...")
                chunks = chunk_documents(raw_docs)
                num_chunks = len(chunks) 
                st.write(f"Created {num_chunks} chunks.")

                # Step 3: Embedding
                st.write("Creating vector embeddings...")
                vector_db = create_vector_db(chunks)
                
                # Step 4: Building Chain
                st.write("Building RAG chain...")
                st.session_state.chain = build_chain(vector_db)
                
                # Finalize Status
                status.update(label="Processing Complete!", state="complete", expanded=False)
                
                # Display Success and Chunk Count
                st.success(f"Webpage processed successfully! Total Chunks Created: {num_chunks}")
                
            except Exception as e:
                status.update(label="Error occurred", state="error")
                st.error(f"An error occurred: {e}")

question_input = st.text_input("Ask a question about the website")

if question_input:
    if "chain" in st.session_state:
        try:
            response = st.session_state.chain.invoke(question_input)
            st.write("### Answer")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
    else:
        st.warning("Please process a URL first.")
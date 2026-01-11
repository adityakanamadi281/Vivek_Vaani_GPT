import os
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser



st.set_page_config(
    page_title="Vivek Vaani",
    page_icon="📘",
    layout="centered"
)

st.title("Vivek Vaani")
st.caption("Ask questions grounded in Swami Vivekananda’s lectures")




load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error(" GEMINI_API_KEY not found in .env file")
    st.stop()



@st.cache_resource
def load_rag_pipeline():
    # LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        api_key=api_key
    )

    # Embeddings
    embeddings = FastEmbedEmbeddings(
        model_name="thenlper/gte-large"
    )

    # Check if vector DB already exists
    persist_directory = "chroma_db"
    
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        # Load existing vector DB
        st.info("Loading existing vector database...")
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # Create new vector DB from PDF
        st.info("Creating vector database from PDF (this may take a while)...")
        file_path = r"C:\Users\adity\Vivek_Vaani_GPT\data\Lectures_from_Colombo_To_Almora.pdf"
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # Text Splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024,
            chunk_overlap=50
        )
        chunks = text_splitter.split_documents(docs)

        # Vector DB
        vectordb = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vectordb.persist()

    retriever = vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Swami Vivekananda assistant. Answer using only the context provided."),
        ("system", "Context: {context}"),
        ("user", "{query}")
    ])

    # Helper function to format retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Helper function to retrieve and format context
    def retrieve_context(input_dict):
        query = input_dict["query"]
        docs = retriever.invoke(query)
        return format_docs(docs)

    # Chain
    rag_chain = (
        {"context": retrieve_context, "query": lambda x: x["query"]}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Initialize RAG pipeline
try:
    rag_chain = load_rag_pipeline()
except Exception as e:
    st.error(f"Error loading RAG pipeline: {str(e)}")
    st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_query = st.chat_input("Ask a question about Swami Vivekananda...")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append(
        {"role": "user", "content": user_query}
    )
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"query": user_query})
                st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )
            except Exception as e:
                error_msg = f"Sorry, an error occurred: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

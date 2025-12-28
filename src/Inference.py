import os
from dotenv import load_dotenv 
from langchain_google_genai import ChatGoogleGenerativeAI 
#from langchain_community.embeddings import GoogleGenerativeAIEmbeddings 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings 
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=api_key
)

file_path = r"C:\Users\adity\Vivek_Vaani_GPT\data\Lectures_from_Colombo_To_Almora.pdf"

loader = PyPDFLoader(file_path)   # path to your PDF
docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1024, chunk_overlap=50
)
chunks = text_splitter.split_documents(docs)
embeddings = FastEmbedEmbeddings(model_name="thenlper/gte-large")


vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="chroma_db"   # folder to store DB
)

vectordb.persist()

retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}    # fetch top 3 relevant chunks
)


RAG_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", "You are a Swami Vivekanandaassistant. Answer using only the context provided."),
    ("system", "Context: {context}"),
    ("user", "{query}")
])


rag_chain = (
    {"context": retriever, "query": lambda x: x["query"]} 
    | RAG_TEMPLATE 
    | llm 
    | StrOutputParser()
)


# response = rag_chain.invoke({"query": "what is the role of Aditya kanamadi"})
# print(response)




while True:
    user_query = input("You: ")

    if user_query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    response = rag_chain.invoke({"query": user_query})
    print("Assistant:", response, "\n")
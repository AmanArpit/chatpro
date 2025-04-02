from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from operator import itemgetter
import streamlit as st
import tempfile
import os
import yaml

# Load API Key
with open('gemini_api_credentials.yml', 'r') as file:
    api_creds = yaml.safe_load(file)

os.environ['GOOGLE_API_KEY'] = api_creds['GOOGLE_API_KEY']

# Streamlit UI Setup
st.set_page_config(page_title="MULTIPDF QA Chatbot", page_icon="🤖")
st.title("Welcome to MULTIPLE PDF QA RAG Chatbot 🤖")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    try:
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
            loader = PyMuPDFLoader(temp_filepath)
            docs.extend(loader.load())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        doc_chunks = text_splitter.split_documents(docs)
        
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Ensure the directory exists
        persist_dir = "./chroma_db"
        os.makedirs(persist_dir, exist_ok=True)
        
        vectordb = Chroma.from_documents(doc_chunks, embeddings_model, persist_directory=persist_dir)
        vectordb.persist()
        
        return vectordb.as_retriever()
    except Exception as e:
        st.error(f"An error occurred while configuring the retriever: {e}")
        st.stop()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

uploaded_files = st.sidebar.file_uploader(label="Upload PDF files", type=["pdf"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

try:
    retriever = configure_retriever(uploaded_files)
except Exception as e:
    st.error(f"Failed to configure retriever: {e}")
    st.stop()

gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1, streaming=True)

qa_template = """
Use only the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know,
don't try to make up an answer. Keep the answer as concise as possible.

{context}

Question: {question}
"""
qa_prompt = ChatPromptTemplate.from_template(qa_template)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

qa_rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question")
    }
    | qa_prompt
    | gemini
)

streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question?")

for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    with st.chat_message("ai"):
        stream_handler = StreamHandler(st.empty())
        response = qa_rag_chain.invoke({"question": user_prompt}, callbacks=[stream_handler])
        st.write(response.content)


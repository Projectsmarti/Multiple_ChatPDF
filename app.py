import os
import streamlit as st
from dotenv import load_dotenv
import tempfile
import json

# Advanced Document Processing Libraries
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.docstore.document import Document

# Additional Libraries for Enhanced Processing
import requests
from transformers import pipeline
import torch
import google  # For web searching
from newspaper import Article
from googlesearch import search

# Configuration and Setup
load_dotenv()


class AdvancedDocumentIntelligence:
    def __init__(self):
        # API and Configuration Setup
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

        # Initialize Advanced Components
        self.summarization_model = self._load_summarization_model()
        self.embedding_model = HuggingFaceEmbeddings()

        # Streamlit Configuration
        st.set_page_config(
            page_title="Advanced Document Intelligence",
            page_icon="📚",
            layout="wide"
        )

        # Initialize Session States
        self._initialize_session_states()

    def _load_summarization_model(self):
        """Load advanced summarization model"""
        try:
            summarizer = pipeline(
                "summarization",
                model="facebook/bart-large-cnn",
                device=0 if torch.cuda.is_available() else -1
            )
            return summarizer
        except Exception as e:
            st.error(f"Summarization Model Error: {e}")
            return None

    def _initialize_session_states(self):
        """Initialize comprehensive session states"""
        session_state_configs = {
            'uploaded_documents': [],
            'document_summaries': {},
            'page_extractions': {},
            'chat_history': [],
            'vectorstore': None,
            'conversation_chain': None,
            'research_results': {}
        }

        for key, default_value in session_state_configs.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

    def load_document(self, uploaded_file):
        """Advanced document loading with multiple format support"""
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                temp_file.write(uploaded_file.getbuffer())
                temp_path = temp_file.name

            # Select loader based on file extension
            ext = os.path.splitext(uploaded_file.name)[1].lower()
            if ext == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(temp_path)
            else:
                raise ValueError(f"Unsupported file type: {ext}")

            # Load documents
            documents = loader.load()

            # Remove temporary file
            os.unlink(temp_path)

            return documents
        except Exception as e:
            st.error(f"Document Loading Error: {e}")
            return []

    def generate_comprehensive_summary(self, documents):
        """Generate advanced multi-level summary"""
        try:
            # Combine all text
            full_text = " ".join([doc.page_content for doc in documents])

            # Generate summary
            summary = self.summarization_model(
                full_text,
                max_length=500,
                min_length=100,
                do_sample=False
            )[0]['summary_text']

            # Page-wise extraction
            page_summaries = {}
            for i, doc in enumerate(documents):
                page_summary = self.summarization_model(
                    doc.page_content,
                    max_length=200,
                    min_length=50,
                    do_sample=False
                )[0]['summary_text']
                page_summaries[i + 1] = page_summary

            return {
                'full_summary': summary,
                'page_summaries': page_summaries
            }
        except Exception as e:
            st.error(f"Summarization Error: {e}")
            return None

    def create_vectorstore(self, documents):
        """Create advanced vector store for semantic search"""
        try:
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory='./document_vectors'
            )
            return vectorstore
        except Exception as e:
            st.error(f"Vectorstore Creation Error: {e}")
            return None

    def setup_conversational_chain(self, vectorstore):
        """Setup advanced conversational retrieval"""
        try:
            llm = ChatGroq(
                model="llama-3.1-70b-versatile",
                temperature=0.3,
                max_tokens=1024
            )

            memory = ConversationBufferMemory(
                memory_key='chat_history',
                return_messages=True,
                output_key='answer'
            )

            conversation_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                memory=memory,
                return_source_documents=True
            )

            return conversation_chain
        except Exception as e:
            st.error(f"Conversation Chain Setup Error: {e}")
            return None

    def web_research(self, query):
        """Perform advanced web research"""
        try:
            # Use Google search API (requires additional setup)
            search_results = []
            for result in google.search(query, num_results=5):
                try:
                    article = Article(result)
                    article.download()
                    article.parse()
                    search_results.append({
                        'url': result,
                        'title': article.title,
                        'summary': article.summary
                    })
                except Exception:
                    continue

            return search_results
        except Exception as e:
            st.error(f"Web Research Error: {e}")
            return []

    def document_processing_pipeline(self, uploaded_files):
        """Comprehensive document processing pipeline"""
        all_documents = []

        for uploaded_file in uploaded_files:
            # Load documents
            documents = self.load_document(uploaded_file)
            all_documents.extend(documents)

            # Generate summaries
            summaries = self.generate_comprehensive_summary(documents)
            st.session_state.document_summaries[uploaded_file.name] = summaries

        # Create vectorstore
        vectorstore = self.create_vectorstore(all_documents)
        st.session_state.vectorstore = vectorstore

        # Setup conversation chain
        conversation_chain = self.setup_conversational_chain(vectorstore)
        st.session_state.conversation_chain = conversation_chain

        st.success("Documents processed successfully!")

    def intelligent_chat_interface(self):
        """Advanced chat interface with multiple capabilities"""
        st.title("🚀 Intelligent Document Intelligence System")

        # Document Upload
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )

        if uploaded_files and uploaded_files != st.session_state.uploaded_documents:
            st.session_state.uploaded_documents = uploaded_files
            self.document_processing_pipeline(uploaded_files)

        # Chat and Research Tabs
        tab1, tab2 = st.tabs(["Document Chat", "Web Research"])

        with tab1:
            # Document Chat Interface
            st.subheader("Document Intelligence Chat")

            # Chat Input
            user_query = st.chat_input("Ask questions about your documents...")

            if user_query and st.session_state.conversation_chain:
                # Process Query
                response = st.session_state.conversation_chain({
                    "question": user_query
                })

                # Display Response
                st.chat_message("assistant").write(response['answer'])

                # Source Documents
                with st.expander("Reference Sources"):
                    for doc in response.get('source_documents', []):
                        st.write(doc.page_content)

        with tab2:
            # Web Research Interface
            st.subheader("Contextual Web Research")

            research_query = st.text_input("Enter research query...")

            if st.button("Search Web") and research_query:
                research_results = self.web_research(research_query)

                for result in research_results:
                    st.markdown(f"### {result['title']}")
                    st.write(result['summary'])
                    st.write(f"Source: {result['url']}")
                    st.divider()

    def run(self):
        """Main application runner"""
        self.intelligent_chat_interface()


def main():
    system = AdvancedDocumentIntelligence()
    system.run()


if __name__ == "__main__":
    main()

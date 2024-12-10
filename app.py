import os
import json
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,  # Changed from DocxLoader
    TextLoader,
    UnstructuredFileLoader
)
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from typing import List, Dict, Any

# Load environment variables
load_dotenv()


class MultiDocumentChatbot:
    def __init__(self):
        # Configuration and API Keys
        self.GROQ_API_KEY = os.getenv('GROQ_API_KEY')
        self.HUGGINGFACE_API_KEY = os.getenv('HUGGINGFACE_API_KEY', '')

        # Streamlit page configuration
        st.set_page_config(
            page_title="Multi-Document Intelligence",
            page_icon="📚",
            layout="wide"
        )

        # Initialize session state
        self._initialize_session_state()

    def _initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        # Consistent initialization of session state variables
        session_state_keys = [
            'chat_history', 'vectorstore',
            'conversation_chain', 'uploaded_files'
        ]
        for key in session_state_keys:
            if key not in st.session_state:
                st.session_state[key] = [] if key == 'chat_history' or key == 'uploaded_files' else None

    def load_document(self, uploaded_file):
        """
        Load document based on file type
        Supports PDF, DOCX, TXT, and other unstructured files
        """
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        try:
            # Temporarily save the uploaded file
            temp_path = os.path.join(os.getcwd(), uploaded_file.name)
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())

            # Select appropriate loader based on file type
            if file_extension == '.pdf':
                loader = PyPDFLoader(temp_path)
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(temp_path)
            elif file_extension == '.txt':
                loader = TextLoader(temp_path)
            else:
                loader = UnstructuredFileLoader(temp_path)

            # Load documents
            documents = loader.load()

            # Optional: Remove temporary file
            os.remove(temp_path)

            return documents
        except Exception as e:
            st.error(f"Error loading document {uploaded_file.name}: {str(e)}")
            return []

    def split_documents(self, documents, chunk_size=1000, chunk_overlap=200):
        """Split documents into manageable chunks."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def create_vectorstore(self, documents):
        """Create a Chroma vectorstore from documents."""
        try:
            embeddings = HuggingFaceEmbeddings()
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory='./vector_db'
            )
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vectorstore: {str(e)}")
            return None

    def setup_conversational_chain(self):
        """Setup conversational retrieval chain with Groq LLM."""
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
                retriever=st.session_state.vectorstore.as_retriever(search_kwargs={'k': 3}),
                memory=memory,
                return_source_documents=True
            )

            return conversation_chain
        except Exception as e:
            st.error(f"Error setting up conversation chain: {str(e)}")
            return None

    def process_uploaded_files(self, uploaded_files):
        """Process uploaded files and create vectorstore."""
        all_documents = []

        for uploaded_file in uploaded_files:
            # Load and process documents
            documents = self.load_document(uploaded_file)
            split_docs = self.split_documents(documents)
            all_documents.extend(split_docs)

        # Create vectorstore
        st.session_state.vectorstore = self.create_vectorstore(all_documents)

        # Setup conversation chain
        if st.session_state.vectorstore:
            st.session_state.conversation_chain = self.setup_conversational_chain()
            st.success(f"Processed {len(uploaded_files)} documents successfully!")
        else:
            st.error("Failed to create vectorstore. Please check your documents.")

    def chat_interface(self):
        """Main Streamlit chat interface."""
        st.title("📚 Multi-Document Intelligence Chatbot")

        # File upload section
        uploaded_files = st.file_uploader(
            "Upload Documents",
            type=['pdf', 'docx', 'txt'],
            accept_multiple_files=True
        )

        if uploaded_files and uploaded_files != st.session_state.uploaded_files:
            st.session_state.uploaded_files = uploaded_files
            self.process_uploaded_files(uploaded_files)

        # Chat history display
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if user_prompt := st.chat_input("Ask about your documents..."):
            # User message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_prompt
            })

            # AI Response
            if st.session_state.conversation_chain:
                try:
                    response = st.session_state.conversation_chain({
                        "question": user_prompt
                    })

                    ai_response = response['answer']
                    source_docs = response.get('source_documents', [])

                    # Display AI response
                    with st.chat_message("assistant"):
                        st.markdown(ai_response)

                        # Optional: Show source document highlights
                        with st.expander("Source Documents"):
                            for doc in source_docs:
                                st.write(doc.page_content)

                    # Update chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": ai_response
                    })
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
            else:
                st.warning("Please upload documents first.")

    def run(self):
        """Run the Streamlit application."""
        self.chat_interface()


# Entrypoint
def main():
    chatbot = MultiDocumentChatbot()
    chatbot.run()


if __name__ == "__main__":
    main()
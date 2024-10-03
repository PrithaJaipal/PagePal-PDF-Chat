import streamlit as st
import warnings
import os
import certifi
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import tempfile
import time

# Wrapper to measure time taken
def time_taken(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time() - t1
        st.write(f"{func.__name__} ran in {t2} seconds")
        return result
    return wrapper

# Set up environment variables
def setup_environment():
    os.environ['CURL_CA_BUNDLE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = ''

# Upload PDF
@time_taken
def upload_pdf():
    return st.file_uploader("Choose a PDF file", type="pdf")

# Create temp path
@time_taken
def create_tmp_path(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# Split text into chunks
@time_taken
def split_texts(pages, chunk_size, chunk_overlap, separators=['\n\n', '\n', '(?=>\. )', ' ', '']):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators
    )
    return text_splitter.split_documents(pages)

# Extract text content
@time_taken
def extract_text_content(texts):
    return [text.page_content for text in texts]

# Initialize HuggingFaceEmbeddings
@time_taken
def get_embeddings():
    return HuggingFaceEmbeddings()

# Initialize HuggingFaceEndpoint for LLM
@time_taken
def initialize_llm(api_token, repo_id, temperature, max_new_tokens, top_p, top_k):
    return HuggingFaceEndpoint(
        huggingfacehub_api_token=api_token,
        repo_id=repo_id,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
    )

# Set up Chroma database for each document and store in session state
@time_taken
def setup_chroma_for_document(texts, embeddings, doc_name):
    if 'chroma_stores' not in st.session_state:
        st.session_state['chroma_stores'] = {}

    # If this document doesn't have a vector store yet, create it
    if doc_name not in st.session_state['chroma_stores']:
        db = Chroma.from_documents(texts, embedding=embeddings, persist_directory=f"/chroma_{doc_name}.sqlite3")
        st.session_state['chroma_stores'][doc_name] = db
        st.write(f"Created Chroma database for {doc_name}")
    else:
        st.write(f"Using existing Chroma database for {doc_name}")

    return st.session_state['chroma_stores'][doc_name]

# Create a prompt template
def create_prompt_template():
    template = """
    You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. Below is some information.
    {context}

    Based on the above information only, answer the below question. If the information is not present in the provided text, respond with "The document does not contain the information needed to answer this question."
    {question}
    """
    return PromptTemplate.from_template(template)

# Perform a similarity search
@time_taken
def perform_similarity_search(db, query, k=1):
    similar_docs = db.similarity_search(query, k=k)
    if len(similar_docs) >= 2:
        return " ".join([doc.page_content for doc in similar_docs])
    else:
        return similar_docs[0].page_content

# Main function to run the Streamlit app
@time_taken
def main():
    st.title("PagePal - PDF Chat")
    warnings.filterwarnings('ignore')
    setup_environment()

    # Sidebar sliders for text splitting parameters
    st.sidebar.title("Text Splitting Parameters")
    chunk_size = st.sidebar.slider("Chunk Size", min_value=500, max_value=10000, value=5000, step=100)
    chunk_overlap = st.sidebar.slider("Chunk Overlap", min_value=0, max_value=500, value=0)

    # Sidebar sliders for LLM parameters
    st.sidebar.title("LLM Initialization Parameters")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    max_new_tokens = st.sidebar.slider("Max New Tokens", min_value=1, max_value=1000, value=500)
    top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.1, step=0.01)
    top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=5)

    # Initialize LLM
    huggingfacehub_api_token = "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"
    repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
    llm = initialize_llm(huggingfacehub_api_token, repo_id, temperature, max_new_tokens, top_p, top_k)

    # Check if there are already stored Chroma documents
    if 'chroma_stores' in st.session_state:
        doc_options = list(st.session_state['chroma_stores'].keys()) + ['Add New']
    else:
        doc_options = ['Add New']

    # Allow the user to select which document to query
    selected_doc = st.selectbox("Select document to query or upload a new one:", doc_options)

    # If the user selects 'Add New', upload a PDF and create the vector store
    if selected_doc == 'Add New':
        uploaded_file = upload_pdf()
        if uploaded_file is None:
            st.write("Please upload a PDF")
            return

        doc_name = uploaded_file.name  # Use document name as identifier
        tmp_file_path = create_tmp_path(uploaded_file)
        loader = PyPDFLoader(tmp_file_path)
        pages = loader.load_and_split()
        texts = split_texts(pages, chunk_size, chunk_overlap)

        st.write(f"Number of documents in {doc_name}: {len(texts)}")

        # Initialize embeddings
        embeddings = get_embeddings()

        # Set up Chroma database for this specific document
        db = setup_chroma_for_document(texts, embeddings, doc_name)

    else:
        db = st.session_state['chroma_stores'][selected_doc]

    # Query input from the user
    query = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if query:
            # Create prompt template and LLM chain
            prompt = create_prompt_template()
            llm_chain = prompt | llm

            # Perform similarity search on the selected document
            context = perform_similarity_search(db, query)
            st.write("Context:", context)

            # Generate and display the response
            response = llm_chain.invoke({"context": context, "question": query})
            if response:
                st.write("LLM Response:", response)
            else:
                st.write("No response generated. Please try a different query.")
        else:
            st.write("No query entered. Please enter a question.")

# Run the Streamlit app
if __name__ == "__main__":
    main()

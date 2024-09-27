import streamlit as st
import os
import base64
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import subprocess
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Configuration
pdf_folder_path = "reportstest"

# Function to interact with the locally downloaded Ollama LLaMA model
def llama_summarize(text, summary_length="500 words"):
    """Generate a summary of the text using a locally downloaded LLaMA model via Ollama."""    
    length_map = {
        "500 words": "Summarize the following text and include all the key points in less than 500 words.",
        "less than 1000 words": "Summarize the following text and include all the key points in less than 1000 words.",
        "more than 1000 words": "Provide a detailed summary and include all the key points of more than 1000 words."
    }

    length_instruction = length_map.get(summary_length, "Summarize the following text.")
    
    prompt = f"{length_instruction} Include all key points and important information:\n{text}\nProvide a single summary."
    
    # Use absolute path to Ollama
    ollama_path = "D:/Summary/ollama/ollama.exe"  # Replace with the actual path to Ollama executable
    model_name = "llama3:8b"  # Replace with your specific model name
    command = f'"{ollama_path}" run {model_name}'
    
    # Pass the prompt through stdin
    result = subprocess.run(command, input=prompt.encode('utf-8'), shell=True, capture_output=True, text=False)
    
    if result.returncode != 0:
        raise RuntimeError(f"Error running Ollama: {result.stderr.decode('utf-8')}")
    
    return result.stdout.decode('utf-8')

# Initialize session state variables
if 'retrieved_files' not in st.session_state:
    st.session_state['retrieved_files'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
if 'file_summaries' not in st.session_state:
    st.session_state['file_summaries'] = {}
if 'full_summary' not in st.session_state:
    st.session_state['full_summary'] = None

# Split the extracted text into chunks for summarization
def get_text_chunks(text):
    """Split text into chunks for summarization."""    
    splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = splitter.split_text(text)
    return chunks

# Extract text from PDF
def get_pdf_text(pdf_path):
    """Extract text from a PDF file."""    
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to search for PDFs by keywords
def search_pdfs_by_keywords(keywords, folder_path):
    """Search for PDFs in the folder that contain any of the keywords in their content."""
    matching_files = []
    
    if not os.path.exists(folder_path):
        st.write(f"Folder path '{folder_path}' does not exist.")
        return []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            
            # Extract text from the PDF file
            pdf_text = get_pdf_text(file_path)
            
            # Check if any keyword is in the extracted text
            if any(keyword.lower() in pdf_text.lower() for keyword in keywords):
                matching_files.append(filename)
    
    return matching_files

def get_base64_download_link(file_path, filename):
    """Generate a base64-encoded link to download a file."""    
    with open(file_path, "rb") as f:
        file_data = f.read()

    b64 = base64.b64encode(file_data).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">{filename}</a>'
    return href

# Provide file links and summaries for matching PDFs
def provide_pdf_links_and_summaries(keywords, summary_length):
    """Generate downloadable links and summaries for PDFs matching the keyword in the content."""
    
    matching_files_count = 0  # Keep track of the number of matching files
    individual_summaries = []  # Store individual summaries for consolidated summary

    if not os.path.exists(pdf_folder_path):
        st.write(f"Folder path '{pdf_folder_path}' does not exist.")
        return
    
    # Iterate through each file in the folder one by one
    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            
            # Extract text from the PDF file
            pdf_text = get_pdf_text(file_path)
            
            # Check if any keyword is in the extracted text
            if any(keyword.lower() in pdf_text.lower() for keyword in keywords):
                # Increment the count for matched files
                matching_files_count += 1
                
                # Generate the download link for the file
                download_link = get_base64_download_link(file_path, filename)
                text_chunks = get_text_chunks(pdf_text)
                
                if not text_chunks:
                    st.write(f"No text chunks found for {filename}.")
                    continue
                
                # Combine all chunks into a single text string
                combined_text = " ".join(text_chunks)
                
                # Generate summary for the combined text
                full_summary = llama_summarize(combined_text, summary_length)
                
                # Store summary and link in session state
                st.session_state.file_summaries[filename] = {'link': download_link, 'summary': full_summary}
                individual_summaries.append(full_summary)  # Store the summary for consolidation
                
                # Display the file's download link and summary immediately
                st.markdown(f'- **Summary of {download_link}:**\n{full_summary}\n\n', unsafe_allow_html=True)

    # After processing all files, display the total count of matched files
    if matching_files_count > 0:
        st.write(f"\nTotal files matched: {matching_files_count}")
        
        # Generate and display consolidated summary
        consolidated_summary = llama_summarize(" ".join(individual_summaries), "less than 1000 words")  # Customize as needed
        st.write("### Consolidated Summary:")
        st.write(consolidated_summary)
    else:
        st.write("No files found related to the provided keywords.")

# Handle user-uploaded files
def handle_uploaded_file(uploaded_file, summary_length):
    """Process and summarize the user-uploaded file."""
    if uploaded_file is not None:
        file_path = os.path.join(pdf_folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())
        
        pdf_text = get_pdf_text(file_path)
        text_chunks = get_text_chunks(pdf_text)
        
        if not text_chunks:
            st.write(f"No text chunks found for the uploaded file.")
            return
        
        # Combine all chunks into a single text string
        combined_text = " ".join(text_chunks)
        
        # Generate summary for the combined text
        full_summary = llama_summarize(combined_text, summary_length)
        
        # Generate download link for the uploaded file
        download_link = get_base64_download_link(file_path, uploaded_file.name)
        
        # Display the summary and download link for the uploaded file
        st.markdown(f"**Summary of uploaded file ({uploaded_file.name}):**\n{full_summary}\n\n", unsafe_allow_html=True)
        st.markdown(f'<a href="data:application/pdf;base64,{base64.b64encode(open(file_path, "rb").read()).decode()}" download="{uploaded_file.name}">Download {uploaded_file.name}</a>', unsafe_allow_html=True)

# Streamlit UI
st.set_page_config(page_title="PDF Finder & Summarizer", page_icon="📄")
st.markdown('<div style="background-color:#C4E4F7;padding:10px;border-radius:10px;">'
            '<h2 style="color:#214761;text-align:center;font-size:28px;"><b>PDF Finder and Summarizer 📄</b></h2>'
            '</div>', unsafe_allow_html=True)

# Sidebar for keyword input and summary options
st.sidebar.title("Search PDF Files")
search_keywords = st.sidebar.text_input("Enter keywords to search in PDFs (comma-separated)")
summary_length = st.sidebar.selectbox("Summary Length:", ["500 words", "less than 1000 words", "more than 1000 words"])

# Button to search and summarize PDFs
if st.sidebar.button("Search and Summarize"):
    if search_keywords:
        keyword_list = [keyword.strip() for keyword in search_keywords.split(',')]
        provide_pdf_links_and_summaries(keyword_list, summary_length)
    else:
        st.sidebar.write("Please enter keywords to search.")

# File uploader for user-uploaded files
st.sidebar.title("Upload PDF File")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file to upload", type="pdf")

# Handle user-uploaded file
if st.sidebar.button("Summarize Uploaded File") and uploaded_file:
    handle_uploaded_file(uploaded_file, summary_length)

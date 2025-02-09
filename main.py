# import os
# import streamlit as st
# from dotenv import load_dotenv
# import pickle
# from PyPDF2 import PdfReader
# from langchain_openai import OpenAI
# from langchain_community.vectorstores import FAISS
# from langchain.chains.question_answering import load_qa_chain
# from langchain.callbacks import get_openai_callback
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# load_dotenv()

# def main():
#     st.header("Chat with PDF")
#     st.sidebar.title("LLM ChatApp with LangChain")
    
#     st.sidebar.markdown('''
#     This is an LLM powered RAG chatbot using streamlit and langchain 
#     ''')


#     pdf = st.file_uploader("Upload your file here...", type = "pdf")
#     raw_text = ""
        
#     if pdf is not None:
#         reader = PdfReader(pdf)
#         for page in reader.pages:
#             text = page.extract_text()
#             if text:
#                 raw_text += text
#         # st.write(text)

#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size = 1000,
#         chunk_overlap = 200,
#         length_function = len
#     )
#     chunks = text_splitter.split_text(raw_text)
#     store_name = "first"
#     st.write(store_name)
    
#     if os.path.exists(f"{store_name}.pkl"):
#         with open(f"{store_name}.pkl", "rb") as fd:
#             vector_store = pickle.load(fd)
#         st.write("Embeddings loaded from the disk")
#     else:
#         embeddings = OpenAIEmbeddings()
#         vector_store = FAISS.from_texts(chunks, embeddings)
#         with open("f{store_name}.pkl", "wb"):
#             pickle.dump(vector_store, fd)
#         st.write("Embeddings created")


#     query = st.text_input("Ask question from your pdf file")
#     if query:
#         docs = vector_store.similarity_search(query = query, k = 3)
#         llm = OpenAI()
#         chain = load_qa_chain(llm, chain_type = "stuff")
#         with get_openai_callback() as cb:
#             response = chain.run(input_documents = docs, question = query)
#             print(cb)
#         st.write(response)



# if __name__ == "__main__":
#     main()






import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI
from langchain.memory import ConversationBufferMemory
import gradio as gr

# Load environment variables
load_dotenv()

# Load OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_pdf_text(pdf_path):
    """Extracts and combines text from all pages of a PDF file."""
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            raw_text += text
    return raw_text


def split_text_into_chunks(raw_text, chunk_size=1000, chunk_overlap=200):
    """Splits raw text into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_text(raw_text)


def initialize_vector_store(texts):
    """Creates a FAISS vector store from the given texts."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_texts(texts, embeddings)


def fetch_answer(query, chain, docsearch):
    """Fetches an answer to the query using the QA chain and document search."""
    docs = docsearch.similarity_search(query)
    response = chain.run(input_documents=docs, question=query)
    return response


def process_pdf(pdf_file):
    """Process the uploaded PDF and prepare it for querying."""
    raw_text = load_pdf_text(pdf_file.name)
    texts = split_text_into_chunks(raw_text)
    docsearch = initialize_vector_store(texts)
    # memory = Conversation
    chain = load_qa_chain(OpenAI(), chain_type='stuff')
    return docsearch, chain


def chatbot(query, chat_history, docsearch, chain):
    """Handles the chatbot interactions with the PDF content."""
    # Fetch the answer using the QA chain
    response = fetch_answer(query, chain, docsearch)
    
    # Append to the chat history
    chat_history.append((query, response))
    return chat_history, ""  # Returning empty string as output for the textbox


# Gradio Interface
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# 🟢 Chat with PDF")
        
        with gr.Row():
            with gr.Column():
                pdf_file = gr.File(label="Upload PDF", type="filepath")  # Use `type="file"` for file inputs
                chatbot_widget = gr.Chatbot(label="Chat with PDF")
                user_input = gr.Textbox(label="Your message", placeholder="Type your query...")
                send_button = gr.Button("Send")
        
        # Initialize chat history and states for docsearch and chain
        chat_history = gr.State([])
        docsearch_state = gr.State(None)
        chain_state = gr.State(None)

        def on_pdf_upload(pdf_file):
            docsearch, chain = process_pdf(pdf_file)
            return docsearch, chain, "PDF processed and ready to chat!"

        def on_send(user_message, chat_history, docsearch, chain):
            return chatbot(user_message, chat_history, docsearch, chain)

        # Set up Gradio events
        pdf_file.upload(
            on_pdf_upload,
            inputs=[pdf_file],
            outputs=[docsearch_state, chain_state, gr.Textbox(label="Status")]
        )
        send_button.click(
            on_send,
            inputs=[user_input, chat_history, docsearch_state, chain_state],
            outputs=[chatbot_widget, user_input]
        )

    return interface


# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(debug=True, share = True)

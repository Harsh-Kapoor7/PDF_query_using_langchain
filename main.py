import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import OpenAI

load_dotenv()


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



def main():
    pdf_path = "/home/harsh/Documents/PDF_query_using_langchain/IJCRT2301086.pdf"
    
    # Step 1: Load and process the PDF
    raw_text = load_pdf_text(pdf_path)
    texts = split_text_into_chunks(raw_text)
    
    # Step 2: Initialize vector store and QA chain
    docsearch = initialize_vector_store(texts)
    chain = load_qa_chain(OpenAI(), chain_type='stuff')
    
    # Step 3: Interactive query loop
    print()
    print("-"*100)
    print("PDF Query System: Type your query below (type 'exit' to quit).")
    while True:
        print("-"*100)
        query = input("Enter your query: ")
        if query.lower() == "exit":
            print("Exiting the model. Thank you!")
            break
        response = fetch_answer(query, chain, docsearch)
        print(f"Answer: {response}")



if __name__ == "__main__":
    main()

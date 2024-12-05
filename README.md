# PDF_query_using_langchain
Chat with your PDF files effortlessly using LangChain and OpenAI!

This project allows you to interact with PDF documents as if you're chatting with them. Ask questions, extract information, and get precise answers using the power of LangChain and OpenAI's natural language processing capabilities.

## Features
Extracts text from PDF documents.
Splits content into manageable chunks for efficient querying.
Uses OpenAI and LangChain for question answering.
Quick and easy setup.


## Getting Started
1. git clone git@github.com:Harsh-Kapoor7/PDF_query_using_langchain.git
2. cd PDF_query_using_langchain

3. pip install -r requirements.txt

4. Create a .env file in the project directory and add your OpenAI API key: OPENAI_API_KEY=your_openai_api_key_here

5. python3 main.py

6. Place the PDF file you want to query in the specified path (modify main.py if needed).
Enter your questions in the terminal, and get instant responses!
To exit, type exit.

PDF_query_using_langchain/
│
├── main.py             # Main script for the project
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .env                # Environment variables (add your OpenAI key here)

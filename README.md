LangChain RAG Application with Gemini & ChromaDB

This project is a complete Retrieval-Augmented Generation (RAG) application built using Python. It leverages the LangChain framework to connect a Google Gemini language model with a local ChromaDB vector store. The application can ingest and process both Markdown and PDF documents to answer questions based on their content.
Features

    LLM Integration: Uses Google's Gemini Pro model for powerful and coherent text generation.

    Vector Store: Employs ChromaDB for efficient, local storage and retrieval of document embeddings.

    Document Processing: Capable of loading and processing both .pdf and .md files.

    Secure API Key Management: Uses a .env file to keep your Google API key secure.

    Modular Code: Separated logic for data ingestion (ingest.py) and the main application (app.py).

Project Structure

.
├── app.py              # Main application to run the RAG chain
├── ingest.py           # Script to process documents and build the vector store
├── requirements.txt    # Python dependencies
├── .env                # For storing your API key
├── sample_data/        # Directory for your source documents
│   ├── sample.md
│   └── sample.pdf      # (You need to create this file)
└── README.md

How to Set Up and Run
1. Prerequisites

    Python 3.8 or higher

    An active Google API key with the Gemini API enabled. You can get one from Google AI Studio.

2. Clone the Repository

Clone this project to your local machine.
3. Install Dependencies

Navigate to the project directory and install the required Python packages:

pip install -r requirements.txt

4. Set Up Your API Key

    Rename the provided .env file if necessary.

    Open the .env file and replace "YOUR_API_KEY" with your actual Google API Key.

GOOGLE_API_KEY="AIzaSy...your...key"

5. Add Your Documents

    Place any Markdown (.md) or PDF (.pdf) files you want to query inside the sample_data directory.

    A sample.md file is included.

    Important: You'll need to add your own PDF file, for example, sample.pdf, to this directory to test PDF processing.

6. Ingest Your Data

Before running the main application, you need to process your documents and build the vector store. Run the ingestion script:

python ingest.py

This will create a chroma_db directory in your project folder containing the document embeddings. You only need to run this script whenever you add, remove, or change the documents in the sample_data directory.
7. Run the Application

Now you're ready to start asking questions! Run the main application:

python app.py

The application will initialize, and you can start typing your questions into the command line. Type exit to quit the application.
Example Interaction

$ python app.py
Initializing RAG application...
RAG application ready. Ask a question or type 'exit' to quit.

Your question: What are the benefits of RAG?

Searching for relevant documents and generating an answer...

--- Answer ---
Retrieval-Augmented Generation (RAG) offers several key benefits:

* **Reduces Hallucinations**: It minimizes the chances of the model generating incorrect or fabricated "facts" by grounding the LLM's response in specific, retrieved information.
* **Access to Current Information**: RAG models can overcome the knowledge cutoff limitation of statically trained LLMs by retrieving the latest information.
* **Increased Trust and Transparency**: Users can verify the information as they can be shown the source documents that were used to generate an answer.

--- Sources ---
- sample_data/sample.md



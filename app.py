
import os
import sys
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load environment variables
load_dotenv()

# --- Configuration ---
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Same as in ingest.py
# --- IMPORTANT: UPDATE THIS PATH ---
# This is a much smaller model (~1.6 GB) suitable for laptops.
# Download the model file from: https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q4_K_M.gguf
# and place it in a 'models' directory.
MODEL_PATH = "models/phi-2.Q4_K_M.gguf" 

# --- Validation ---
if not os.path.exists(CHROMA_DB_PATH):
    print(f"Error: ChromaDB not found at '{CHROMA_DB_PATH}'.")
    print("Please run `ingest.py` first to create the vector store.")
    sys.exit(1)

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at '{MODEL_PATH}'.")
    print("Please download the model and place it in the correct directory.")
    print("Download from: https://huggingface.co/TheBloke/phi-2-GGUF/blob/main/phi-2.Q4_K_M.gguf")
    sys.exit(1)


def main():
    """
    Main function to run the RAG application locally.
    Initializes the local models, vector store, and QA chain, then enters a loop
    to answer user questions.
    """
    print("Initializing local RAG application...")
    
    # Initialize the local embedding model
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"Error initializing local embeddings model: {e}")
        sys.exit(1)

    # Load the existing vector store
    vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

    # Initialize the Local LLM
    try:
        print("Loading local LLM... (This may take a moment)")
        llm = LlamaCpp(
            model_path=MODEL_PATH,
            n_gpu_layers=-1, # Offload all layers to GPU if available
            n_batch=512,
            n_ctx=2048,      # Context window for Phi-2
            f16_kv=True,     # Must be True for RAM-only inference
            verbose=False,
        )
    except Exception as e:
        print(f"Error initializing language model: {e}")
        sys.exit(1)

    # Create a custom prompt template
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer from the context, just say that you don't know, don't try to make up an answer.
    Provide a detailed and comprehensive answer based on the provided context.

    Context:
    {context}

    Question:
    {question}

    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Set up the RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )

    print("Local RAG application ready. Ask a question or type 'exit' to quit.\n")

    # Start the interactive question-answering loop
    while True:
        user_query = input("Your question: ")
        if user_query.lower() == 'exit':
            break

        print("Searching for relevant documents and generating an answer...")
        try:
            result = qa_chain.invoke({"query": user_query})
            print("\n--- Answer ---")
            print(result["result"])
            print("\n--- Sources ---")
            for doc in result["source_documents"]:
                print(f"- {doc.metadata.get('source', 'Unknown')}")
            print("\n" + "="*50 + "\n")
        except Exception as e:
            print(f"\nAn error occurred while processing your query: {e}\n")

if __name__ == "__main__":
    main()



import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables from .env file (for app.py)
load_dotenv()

# --- Configuration ---
DATA_PATH = "sample_data"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

def load_documents():
    """
    Loads all documents from the specified data path using different loaders.
    """
    loader_configs = {
        "*.pdf": {"loader_cls": PyPDFLoader},
        "*.md": {"loader_cls": UnstructuredMarkdownLoader},
    }
    
    loaders = []
    for glob_pattern, config in loader_configs.items():
        loaders.append(
            DirectoryLoader(
                DATA_PATH,
                glob=f"**/{glob_pattern}",
                loader_cls=config["loader_cls"],
                show_progress=True,
                use_multithreading=True
            )
        )

    all_documents = []
    for loader in loaders:
        print(f"Loading documents with {loader.glob}...")
        all_documents.extend(loader.load())
        
    return all_documents

def main():
    """
    Main function to ingest data locally. It checks for new documents and adds them
    to a persistent ChromaDB vector store using a local embedding model.
    """
    print("Starting local data ingestion process...")

    # Step 1: Initialize local embeddings model and text splitter
    print(f"Initializing local embeddings model: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # Step 2: Load all documents from the file system
    all_fs_documents = load_documents()
    if not all_fs_documents:
        print("No documents found in the 'sample_data' directory.")
        return
    print(f"Found {len(all_fs_documents)} documents in total on file system.")

    # Step 3: Check if the DB exists and is populated
    if os.path.exists(CHROMA_DB_PATH):
        print("Existing vector store found. Loading and checking for new documents...")
        vector_store = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # Keep uniqueness at (source, page) level
        existing_sources = set(
            (meta.get("source"), meta.get("page"))
            for meta in vector_store.get().get("metadatas", [])
        )

        new_documents = [
            doc for doc in all_fs_documents
            if (doc.metadata.get("source"), doc.metadata.get("page")) not in existing_sources
        ]

        if not new_documents:
            print("\nNo new documents to add. Database is up-to-date.")
            return
            
        print(f"\nFound {len(new_documents)} new documents to ingest.")
        texts_to_add = text_splitter.split_documents(new_documents)

        # Ensure metadata includes page
        for t in texts_to_add:
            if "page" not in t.metadata:
                t.metadata["page"] = t.metadata.get("page", None)

        print(f"Adding {len(texts_to_add)} new chunks to the vector store...")
        vector_store.add_documents(texts_to_add)

    else:
        # --- LOGIC FOR A NEW DATABASE ---
        print("No existing vector store found. Creating a new one...")
        texts = text_splitter.split_documents(all_fs_documents)

        for t in texts:
            if "page" not in t.metadata:
                t.metadata["page"] = t.metadata.get("page", None)

        if not texts:
            print("No text chunks to process. Exiting.")
            return

        print(f"Creating vector store with {len(texts)} chunks...")
        Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )

    print("\n-----------------------------------------")
    print("Local data ingestion complete!")
    print(f"Vector store at '{CHROMA_DB_PATH}' is up-to-date.")
    print("You can now run `app.py` to ask questions.")
    print("-----------------------------------------")

if __name__ == "__main__":
    main()

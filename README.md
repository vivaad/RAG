
# Fully Local RAG Application with LangChain & LlamaCpp

This project is a **self-contained Retrieval-Augmented Generation (RAG) application** that runs **entirely on your local machine**.
It uses **LangChain** to orchestrate the pipeline, **ChromaDB** for embeddings storage, and a **local LLM (Phi-2 via LlamaCpp)** to generate answers from your private PDF and Markdown documents.

✅ **No API keys, no external services, no costs — 100% local & private.**

---

## ✨ Key Features

* 🔒 **100% Local & Private** – Your data never leaves your machine.
* ⚡ **LangChain-Powered** – Flexible orchestration for retrieval + generation.
* 🧠 **Local Embeddings** – Uses [SentenceTransformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`) on CPU/GPU.
* 🤖 **Local LLM (Phi-2)** – Runs with [LlamaCpp](https://github.com/abetlen/llama-cpp-python), optimized for laptops.
* 💾 **Persistent Vector Store** – Stores embeddings in **ChromaDB**, only reprocesses new/changed files.
* 📄 **Multi-Format Support** – Works with both **PDF** and **Markdown** files.

---

## 📂 Project Structure

```
rag_project/
├── models/
│   └── phi-2.Q4_K_M.gguf      # (download & place here)
├── sample_data/
│   ├── sample.md
│   └── sample.pdf
├── chroma_db/                 # (auto-generated after ingestion)
├── main.py                     # Main application (RAG pipeline)
├── ingest.py                  # Document ingestion & vector store builder
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone git@github.com:vivaad/RAG.git
cd RAG
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
# Activate it:
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Download the Local LLM (Phi-2)

* Create a `models/` directory inside the project folder.
* Download the **Phi-2 (quantized) model file**:
  👉 [phi-2.Q4_K_M.gguf](https://huggingface.co/microsoft/phi-2) (~1.6 GB).
* Place it inside `models/`.

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

⚠️ Note: `llama-cpp-python` may take time to install as it compiles for your system.

---

## ▶️ How to Use

### Step 1: Add Your Documents

* Place `.pdf` and `.md` files inside `sample_data/`.
* Example:

```
sample_data/
├── sample.md
└── sample.pdf
```

### Step 2: Ingest the Documents

Run:

```bash
python ingest.py
```

* This will process your documents, create embeddings, and store them in `chroma_db/`.
* On the **first run**, it downloads the SentenceTransformers model (~230 MB).

### Step 3: Run the Application

```bash
python main.py
```

* The first run may take a while to load the Phi-2 model into RAM.
* Then you’ll see:

```
RAG application ready. Ask a question or type 'exit' to quit.
```

---

## 💡 Example Interaction

```
Your question: What is RAG?

--- Answer ---
Retrieval-Augmented Generation (RAG) is a technique that enhances language models
by retrieving relevant context from external documents before generating a response.

--- Sources ---
- sample_data/sample.md
```

---

## 🛠️ Tech Stack

* [LangChain](https://www.langchain.com/)
* [ChromaDB](https://www.trychroma.com/)
* [LlamaCpp](https://github.com/abetlen/llama-cpp-python)
* [Microsoft Phi-2](https://huggingface.co/microsoft/phi-2)
* [SentenceTransformers](https://www.sbert.net/)

---

## ⚡ Performance Tips

Running local models can be demanding. Here’s how to optimize:

### 🔹 1. Use Quantized Models

* You’re already using `phi-2.Q4_K_M.gguf` (quantized), which reduces memory usage.
* If you have more RAM/GPU power, try higher-precision versions (e.g., `Q5`, `Q8`) for better accuracy.

### 🔹 2. Run on GPU (if available)

* Install llama-cpp-python with GPU support:

```bash
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
```

* This enables CUDA acceleration (NVIDIA GPUs).

### 🔹 3. Control Context Length

* Large context windows increase memory usage.
* Adjust parameters in `app.py`, e.g.:

```python
n_ctx = 2048   # reduce if running out of RAM
```

### 🔹 4. Batch Ingestion

* If you have many PDFs, ingest them in smaller batches to avoid memory overload.

### 🔹 5. Monitor System Resources

* On Linux/macOS:

```bash
htop
```

* On Windows: use Task Manager.

---

## ✅ Advantages of Local RAG

* No API keys or hidden costs
* Total data privacy
* Works offline
* Lightweight enough for standard laptops

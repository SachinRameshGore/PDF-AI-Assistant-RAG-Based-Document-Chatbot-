
An AI-powered document assistant that allows users to upload PDF files and interact with them through a conversational interface.  Built using Retrieval-Augmented Generation (RAG) architecture to provide context-grounded responses.

Features

📄 Multi-PDF Upload Support
✂ Semantic Text Chunking
🧠 Local Embeddings (HuggingFace)
🔎 FAISS Vector Similarity Search
🤖 Gemini 2.5 Flash Integration
💬 ChatGPT-style UI (Streamlit)
⚡ Context-Grounded Responses
🛠 Modular Architecture

Architecture
PDF Upload
   ↓
Text Extraction (PyPDF)
   ↓
Chunking (RecursiveCharacterTextSplitter)
   ↓
Embeddings (HuggingFace)
   ↓
FAISS Vector Store
   ↓
Retriever
   ↓
Prompt Template
   ↓
Gemini LLM
   ↓
Chat UI

🧠 How It Works

1.PDFs are parsed and converted into raw text.

2.Text is split into overlapping chunks to handle LLM token limits.

3.Each chunk is converted into vector embeddings.

4.FAISS stores embeddings for fast similarity search.

5.When user asks a question:

      a.Relevant chunks are retrieved

      b.Injected into prompt

      c.Gemini generates grounded response

🛠 Tech Stack

			Python

			Streamlit

			LangChain

			FAISS

			HuggingFace Embeddings

			Gemini 2.5 Flash

⚙ Installation

	pip install -r requirements.txt

Add your .env file:

GOOGLE_API_KEY=your_api_key

Run the app:

	streamlit run app.py

Screenshot 

<img width="1910" height="962" alt="image" src="https://github.com/user-attachments/assets/99bea7ba-f0cc-46c4-8fd9-79ec1c2e7f0b" />


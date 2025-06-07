# Chatbot_Using_RAG_Model

Document-Based Chatbot using LangChain and OpenAI GPT-4
This project is a chatbot that answers questions based on the content of your PDF or TXT
documents. It uses LangChain for document processing, OpenAI embeddings for semantic search,
FAISS for vector similarity search, and GPT-4 for generating responses.
---
Features
- Supports PDF and plain text (.txt) documents
- Automatically splits large documents into manageable chunks
- Creates semantic embeddings for fast similarity-based retrieval
- Uses GPT-4 language model for accurate answers
- Displays source document excerpts along with answers
---
Requirements
- Python 3.8 or higher
- An OpenAI API key with access to GPT-4 and embeddings
---
Installation
1. Clone or download this repository.
2. Create and activate a Python virtual environment (optional but recommended):
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
3. Install required Python packages:
pip install langchain langchain_openai langchain_community faiss-cpu python-dotenv
---
Setup
1. Create a .env file in the root directory of your project with the following content:
OPENAI_API_KEY=your_openai_api_key_here
Replace your_openai_api_key_here with your actual OpenAI API key.
2. Make sure your document files (PDF or TXT) are accessible on your system.
---
Usage
Run the chatbot script:
python catbot.py
When prompted, enter the full file path to your PDF or TXT document. For example:
C:\Users\Darshan\Documents\example.pdf
Then, wait for the chatbot to load, chunk, and embed the document.
Ask your questions about the document. Type exit or quit to stop.
---
Troubleshooting
- OpenAI API quota errors: Check your usage limits on OpenAI dashboard.
- Environment variable not found: Confirm .env file exists and is properly formatted.
- Unsupported file format: Use only .pdf or .txt files.
---
License
This project is open source and free to use.
---
Feel free to contribute or open issues if you encounter any problems!

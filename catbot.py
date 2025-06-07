import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def load_documents(file_path):
    # Support PDF or TXT files
    if file_path.lower().endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or TXT.")
    documents = loader.load()
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(documents)
    return chunks

def main():
    # Load environment variables from .env
    load_dotenv()
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("OpenAI API key not found. Please set it in .env")

    # Ask user for document path
    file_path = input("Enter path to your PDF or TXT knowledge file: ").strip()
    print("Loading documents...")
    documents = load_documents(file_path)
    
    print(f"Loaded {len(documents)} documents. Splitting into chunks...")
    chunks = chunk_documents(documents)

    print(f"Created {len(chunks)} chunks. Generating embeddings...")
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectorstore = FAISS.from_documents(chunks, embeddings)

    print("Initializing language model...")
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0, model_name="gpt-4")

    # Create a RetrievalQA chain to combine retriever + LLM
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=True)

    print("Chatbot is ready! Ask questions about your document (type 'exit' to quit).")

    while True:
        query = input("\nYour question: ")
        if query.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        result = qa_chain(query)
        answer = result['result']
        sources = result['source_documents']
        
        print("\nAnswer:\n", answer)
        print("\nSource chunks:")
        for i, doc in enumerate(sources):
            print(f"[{i+1}] {doc.page_content[:300]}...\n")  # Print first 300 chars

if __name__ == "__main__":
    main()

import os
import tempfile
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def load_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    os.unlink(tmp_path)
    return pages

def chunk_documents(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=80,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(pages)

def build_vectorstore(uploaded_file):
    try:
        pages = load_pdf(uploaded_file)
        if not pages:
            return None
        chunks = chunk_documents(pages)
        if not chunks:
            return None
        embeddings = get_embeddings()
        vectorstore = FAISS.from_documents(chunks, embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error: {e}")
        return None

def get_answer(query, vectorstore):
    try:
        docs = vectorstore.similarity_search(query, k=4)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        sources = [doc.page_content for doc in docs]
        prompt = f"""You are Study Buddy, a helpful academic assistant.
Answer ONLY from the document context below.

RULES:
- If answer is NOT in context, reply exactly: OUT_OF_SCOPE
- Give clear, student-friendly answers
- Use bullet points for multi-part answers

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        response = gemini_model.generate_content(prompt)
        answer = response.text.strip()
        if "OUT_OF_SCOPE" in answer:
            return {
                "answer": "This question is not in your document. Please ask something from your uploaded notes.",
                "sources": [],
                "refused": True
            }
        return {"answer": answer, "sources": sources, "refused": False}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": [], "refused": True}

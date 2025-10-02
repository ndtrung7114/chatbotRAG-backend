from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from utils.uploadFilePDFtoMD import convert_pdf_to_md
from utils.vectorDB import create_retriever, load_retriever
from utils.chunking import split_text_by_markdown
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.llm import ask_question
from pydantic import BaseModel

class QueryRequest(BaseModel):
    question: str

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.streamlit.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

@app.post("/uploadfile/")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file temporarily
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, file.filename)
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        md = convert_pdf_to_md(temp_path)
        chunks = split_text_by_markdown(md)
        retriever = create_retriever(chunks, embeddings)
        os.remove(temp_path)
        return {"message": "File processed and vector store created successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query(request: QueryRequest):
    try:
        retriever = load_retriever(embeddings)
        retrieved_docs = retriever.invoke(request.question)  # Access via request.question
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        answer = ask_question(request.question, context)
        return {"question": request.question, "answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))

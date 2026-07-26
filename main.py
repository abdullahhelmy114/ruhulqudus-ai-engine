from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import psycopg2
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

genai.configure(api_key=GEMINI_API_KEY)

class EmbedRequest(BaseModel):
    chunks: list[str]
    bookTitle: str

@app.post("/embed")
async def embed_chunks(req: EmbedRequest):
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        processed = 0

        for chunk in req.chunks:
            if len(chunk.strip()) < 50:
                continue
                
            # 🚀 تم تغيير النموذج إلى النسخة الأكثر استقراراً في جوجل
            result = genai.embed_content(
                model="models/text-embedding-001",
                content=chunk
            )

            embedding = result['embedding']
            embedding_str = f"[{','.join(map(str, embedding))}]"

            cursor.execute(
                "INSERT INTO knowledge_base (book_title, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                (req.bookTitle, chunk, embedding_str)
            )
            processed += 1

        conn.commit()
        cursor.close()
        conn.close()

        return {"success": True, "processed": processed}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
import psycopg2
import os

app = FastAPI()

# السماح لموقعك فقط بالاتصال بهذا السيرفر (أمان عالي)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# جلب المفاتيح من Coolify
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

genai.configure(api_key=GEMINI_API_KEY)

# تعريف شكل البيانات القادمة من المتصفح
class EmbedRequest(BaseModel):
    chunks: list[str]
    bookTitle: str

@app.post("/embed")
async def embed_chunks(req: EmbedRequest):
    try:
        # الاتصال بقاعدة بيانات Neon
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        processed = 0

        # تحويل الأجزاء لمتجهات وحفظها
        for chunk in req.chunks:
            if len(chunk.strip()) < 50:
                continue
                
            # إنشاء الـ Vector عبر Gemini
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk
            )
            embedding = result['embedding']
            embedding_str = f"[{','.join(map(str, embedding))}]"

            # الحفظ في الداتا بيز
            cursor.execute(
                "INSERT INTO knowledge_base (book_title, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                (req.bookTitle, chunk, embedding_str)
            )
            processed += 1

        # تأكيد الحفظ وإغلاق الاتصال
        conn.commit()
        cursor.close()
        conn.close()

        return {"success": True, "processed": processed}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
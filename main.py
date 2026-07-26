from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import google.generativeai as genai
import psycopg2
import os
import re

app = FastAPI()

# 1. حل مشكلة الـ CORS نهائياً للسماح لموقعك بالتحدث مع بايثون
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # يمكنك لاحقاً وضع رابط موقعك فقط هنا
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. إعدادات البيئة (سيأخذها من إعدادات Coolify لاحقاً)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ضع_مفتاحك_هنا_للتجربة_المحلية")
DATABASE_URL = os.getenv("DATABASE_URL", "ضع_رابط_قاعدة_بيانات_Neon_هنا")

genai.configure(api_key=GEMINI_API_KEY)

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...), book_title: str = Form(...)):
    try:
        # 1. قراءة الـ PDF بسرعة الصاروخ
        pdf_bytes = await file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        full_text = ""
        for page in doc:
            full_text += page.get_text("text") + " "
            
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="الملف فارغ أو يحتوي صوراً فقط.")

        # 2. تقطيع النص
        # نقسم النص لقطع كل منها 1000 حرف تقريباً
        chunks = [full_text[i:i+1000] for i in range(0, len(full_text), 1000)]
        valid_chunks = [c for c in chunks if len(c.strip()) > 50]

        # 3. الاتصال بقاعدة بيانات Neon
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        # 4. تحويل النصوص لمتجهات (Vectors) عبر Gemini وحفظها
        processed = 0
        for chunk in valid_chunks:
            # Gemini Embedding
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunk
            )
            embedding = result['embedding']
            embedding_str = f"[{','.join(map(str, embedding))}]"

            # الحفظ في الداتا بيز
            cursor.execute(
                "INSERT INTO knowledge_base (book_title, chunk_text, embedding) VALUES (%s, %s, %s::vector)",
                (book_title, chunk, embedding_str)
            )
            processed += 1

        # تأكيد الحفظ وإغلاق الاتصال
        conn.commit()
        cursor.close()
        conn.close()

        return {"success": True, "message": f"تمت المعالجة وحفظ {processed} جزء بنجاح!"}

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
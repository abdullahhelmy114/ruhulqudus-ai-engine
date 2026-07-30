import os, io, re, json
from typing import TypedDict, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pdfplumber
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
import traceback  

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-free-key")
BASE_URL = "https://openrouter.ai/api/v1"
client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)

AGENT_MODELS = {
    "strategist": "google/gemma-4-26b-a4b-it:free",
    "titler": "google/gemma-4-26b-a4b-it:free",
    "explainer": "google/gemma-4-31b-it:free",
    "assessor": "google/gemma-4-31b-it:free",
}

class CurriculumState(TypedDict):
    book_text: str
    level: str
    instructions: str
    parts: List[str]
    titles: List[str]
    explanations: List[str]
    assessments: List[str]
    final_markdown: str

class CurriculumRequest(BaseModel):
    book_text: str
    level: str
    instructions: str = ""

app = FastAPI(title="AI Curriculum Generator", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ruhulqudus.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except:
        cleaned = re.sub(r'```json|```', '', text).strip()
        try:
            return json.loads(cleaned)
        except:
            start_idx = min(
                (cleaned.find('{') if cleaned.find('{') != -1 else float('inf')),
                (cleaned.find('[') if cleaned.find('[') != -1 else float('inf'))
            )
            if start_idx != float('inf'):
                end_idx = max(cleaned.rfind('}'), cleaned.rfind(']'))
                if end_idx != -1:
                    return json.loads(cleaned[start_idx:end_idx+1])
            raise

async def strategist_agent(state: CurriculumState) -> CurriculumState:
    prompt = f"""..."""  # نفس الـ prompt السابق
    response = await client.chat.completions.create(
        model=AGENT_MODELS["strategist"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    try:
        data = extract_json(raw)
        state["parts"] = data["parts"]
    except:
        state["parts"] = [f"الجزء {i+1}" for i in range(10)]
    return state

# ... (بقية الوكلاء كما كانت بالضبط في الملف القديم)

async def formatter_agent(state: CurriculumState) -> CurriculumState:
    # كما سابقاً
    pass

def create_curriculum_graph():
    # كما سابقاً
    pass

curriculum_graph = create_curriculum_graph()

# نقطة النهاية الجديدة التي تستقبل ملف PDF

@app.post("/generate-from-pdf")
async def generate_from_pdf(
    file: UploadFile = File(...),
    level: str = Form(...),
    instructions: str = Form("")
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب رفع ملف PDF")
    
    # 1. استخراج النص من PDF
try:
    pdf_bytes = await file.read()
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        full_text = ""
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
    # تسجيل النص المستخرج للتشخيص
    print(f"[PDF Extract] Total chars: {len(full_text.strip())}")
    print(f"[PDF Extract] First 200 chars: {full_text.strip()[:200]}")
    if len(full_text.strip()) < 100:
        raise HTTPException(status_code=400, detail="النص المستخرج قصير جداً (أقل من 100 حرف)")
except Exception as e:
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=f"فشل استخراج النص: {str(e)}")
    
    # 2. تشغيل وكلاء LangGraph
    initial_state: CurriculumState = {
        "book_text": full_text,
        "level": level,
        "instructions": instructions,
        "parts": [],
        "titles": [],
        "explanations": [],
        "assessments": [],
        "final_markdown": ""
    }
    try:
        final_state = await curriculum_graph.ainvoke(initial_state)
        return {
            "success": True,
            "markdown": final_state["final_markdown"],
            "titles": final_state["titles"]
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"فشل توليد المنهج: {str(e)}")

@app.get("/")
async def root():
    return {"status": "AI Curriculum Generator Running", "version": "3.0"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
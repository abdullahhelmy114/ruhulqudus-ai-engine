"""
خدمة وكلاء الذكاء الاصطناعي لتوليد المناهج التعليمية (LangGraph + FastAPI)
- استخراج النص من PDF محلياً عبر pdfplumber
- توليد المنهج عبر وكلاء متسلسلين (OpenRouter - نماذج مجانية)
- معالجة خلفية مع استعلام عن الحالة
- إعادة المحاولة التلقائية عند تجاوز الحد (429)
"""

import os, re, io, json, traceback, uuid, asyncio
from typing import TypedDict, List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pdfplumber
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

# ---------- إعداد OpenRouter ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
BASE_URL = "https://openrouter.ai/api/v1"
client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)

# ---------- النماذج المتخصصة ----------
AGENT_MODELS = {
    "strategist": "google/gemma-4-26b-a4b-it:free",
    "titler": "google/gemma-4-26b-a4b-it:free",
    "explainer": "google/gemma-4-31b-it:free",
    "assessor": "google/gemma-4-31b-it:free",
}

# ---------- الحالة ----------
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

# ---------- تطبيق FastAPI ----------
app = FastAPI(title="AI Curriculum Generator", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ruhulqudus.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تخزين المهام الخلفية
tasks_store: dict[str, dict] = {}

# ---------- دوال مساعدة ----------
def extract_json(text: str) -> dict:
    """تنظيف ردود النماذج لاستخراج JSON."""
    try:
        return json.loads(text)
    except:
        cleaned = re.sub(r'```json|```', '', text).strip()
        try:
            return json.loads(cleaned)
        except:
            start = min(
                (cleaned.find('{') if cleaned.find('{') != -1 else float('inf')),
                (cleaned.find('[') if cleaned.find('[') != -1 else float('inf'))
            )
            if start != float('inf'):
                end = max(cleaned.rfind('}'), cleaned.rfind(']'))
                if end != -1:
                    return json.loads(cleaned[start:end+1])
            raise

async def call_with_retry(
    model_name: str,
    messages: list,
    temperature: float,
    max_tokens: Optional[int] = None,
    max_retries: int = 5
) -> str:
    """استدعاء OpenRouter مع إعادة المحاولة عند 429."""
    last_error = None
    for attempt in range(max_retries):
        try:
            kwargs = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
            }
            if max_tokens:
                kwargs["max_tokens"] = max_tokens
            resp = await client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            last_error = e
            err_str = str(e)
            if "429" in err_str:
                # محاولة استخراج وقت الانتظار من الرسالة
                retry_after = 60
                match = re.search(r"retry in (\d+)s", err_str)
                if match:
                    retry_after = int(match.group(1))
                print(f"⚠️ Rate limited on {model_name}. Retry in {retry_after}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(retry_after + 5)
            else:
                raise  # خطأ غير 429، نوقفه فورًا
    raise last_error

# ---------- الوكلاء ----------
async def strategist_agent(state: CurriculumState) -> CurriculumState:
    prompt = f"""
أنت خبير تعليمي في تصميم المناهج. الكتاب التالي لتعليم العربية والعلوم الإسلامية.
المستوى التعليمي: {state['level']}.
التعليمات الإضافية: {state.get('instructions', 'لا يوجد')}.

مهمتك:
- تحليل محتوى الكتاب.
- تقسيمه إلى 10 أجزاء متسلسلة منطقية لتكوين دورة تعليمية.
- لكل جزء، اذكر نطاق الصفحات أو الفصول التي يغطيها، مع وصف مقتضب لما يتعلمه الطالب فيه.

أعد المخرجات بتنسيق JSON صارم بهذا الشكل:
{{
  "parts": [
    "الجزء 1: وصف ونطاق الصفحات",
    "الجزء 2: ...",
    ...
  ]
}}
نص الكتاب (أول 15000 حرف):
{state['book_text'][:15000]}
"""
    raw = await call_with_retry(AGENT_MODELS["strategist"], [{"role": "user", "content": prompt}], 0.3)
    try:
        data = extract_json(raw)
        state["parts"] = data["parts"]
    except:
        state["parts"] = [f"الجزء {i+1}" for i in range(10)]
    return state

async def titler_agent(state: CurriculumState) -> CurriculumState:
    parts = state["parts"]
    prompt = f"""
أنت مؤلف مناهج مبدع. الأجزاء العشرة للدورة هي:
{chr(10).join(f'{i+1}. {p}' for i, p in enumerate(parts))}
المستوى التعليمي: {state['level']}.

مهمتك:
- لكل جزء، اكتب عنوان درس واضح وجذاب ومناسب للمستوى، يعكس فكرته الرئيسية.
- استخدم لغة عربية فصيحة بسيطة.
أعد المخرجات بصيغة JSON فقط:
{{
  "titles": ["العنوان 1", "العنوان 2", ...]
}}
"""
    raw = await call_with_retry(AGENT_MODELS["titler"], [{"role": "user", "content": prompt}], 0.7)
    try:
        data = extract_json(raw)
        state["titles"] = data["titles"]
    except:
        state["titles"] = [f"الدرس {i+1}" for i in range(10)]
    return state

async def explainer_agent(state: CurriculumState) -> CurriculumState:
    titles = state["titles"]
    parts = state["parts"]
    explanations = []
    for i in range(10):
        prompt = f"""
أنت معلم خبير في تعليم العربية والعلوم الإسلامية. ستشرح الدرس رقم {i+1} من دورة تعليمية.
عنوان الدرس: {titles[i]}
محتوى الجزء المرتبط من الكتاب (مختصر): {parts[i]}
المستوى: {state['level']}.
تعليمات إضافية: {state.get('instructions', '')}.

المطلوب:
- كتابة شرح كامل ومبسط للدرس، يناسب المستوى {state['level']}.
- استخدم لغة واضحة، أمثلة حية، واستعارات تربوية.
- اجعل الشرح شاملاً بحيث يمكن للطالب فهم الموضوع دون الرجوع للكتاب.
- استخدم تنسيق Markdown خفيف (عناوين فرعية، تعداد نقطي) ليكون جاهزاً للعرض.

أعد الشرح فقط، بدون مقدمات.
"""
        text = await call_with_retry(AGENT_MODELS["explainer"], [{"role": "user", "content": prompt}], 0.7, max_tokens=2000)
        explanations.append(text)
    state["explanations"] = explanations
    return state

async def assessor_agent(state: CurriculumState) -> CurriculumState:
    titles = state["titles"]
    explanations = state["explanations"]
    assessments = []
    for i in range(10):
        prompt = f"""
أنت مختص في القياس التربوي. بناءً على شرح الدرس التالي، قم بإنشاء مجموعة أسئلة تقييمية.
عنوان الدرس: {titles[i]}
الشرح: {explanations[i][:2000]}
المستوى: {state['level']}.

أنشئ 3-4 أسئلة متنوعة (اختيار من متعدد، صح وخطأ، سؤال مقالي قصير) مع الإجابات الصحيحة ونقاط التقييم.
قدم الأسئلة بتنسيق Markdown واضح.
"""
        text = await call_with_retry(AGENT_MODELS["assessor"], [{"role": "user", "content": prompt}], 0.8, max_tokens=1500)
        assessments.append(text)
    state["assessments"] = assessments
    return state

async def formatter_agent(state: CurriculumState) -> CurriculumState:
    titles = state["titles"]
    explanations = state["explanations"]
    assessments = state["assessments"]
    md = f"# المنهج التعليمي (المستوى: {state['level']})\n\n"
    for i in range(10):
        md += f"## الدرس {i+1}: {titles[i]}\n\n"
        md += f"### الشرح\n{explanations[i]}\n\n"
        md += f"### التقييم\n{assessments[i]}\n\n---\n\n"
    state["final_markdown"] = md
    return state

# ---------- بناء الرسم البياني ----------
def create_curriculum_graph():
    workflow = StateGraph(CurriculumState)
    workflow.add_node("strategist", strategist_agent)
    workflow.add_node("titler", titler_agent)
    workflow.add_node("explainer", explainer_agent)
    workflow.add_node("assessor", assessor_agent)
    workflow.add_node("formatter", formatter_agent)

    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "titler")
    workflow.add_edge("titler", "explainer")
    workflow.add_edge("explainer", "assessor")
    workflow.add_edge("assessor", "formatter")
    workflow.add_edge("formatter", END)
    return workflow.compile()

curriculum_graph = create_curriculum_graph()

# ---------- مهمة خلفية ----------
async def process_curriculum_background(task_id: str, full_text: str, level: str, instructions: str):
    state: CurriculumState = {
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
        final_state = await curriculum_graph.ainvoke(state)
        tasks_store[task_id] = {
            "status": "completed",
            "result": {
                "success": True,
                "markdown": final_state["final_markdown"],
                "titles": final_state["titles"]
            }
        }
    except Exception as e:
        print(f"Background task {task_id} failed: {e}")
        traceback.print_exc()
        tasks_store[task_id] = {
            "status": "failed",
            "result": {"detail": str(e)}
        }

# ---------- نقاط النهاية ----------
@app.post("/generate-from-pdf")
async def generate_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    level: str = Form(...),
    instructions: str = Form("")
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب رفع ملف PDF")

    # 1. استخراج النص
    try:
        pdf_bytes = await file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "".join(page.extract_text() or "" for page in pdf.pages)
        clean_text = full_text.strip()
        if len(clean_text) < 100:
            raise HTTPException(status_code=400, detail="النص المستخرج قصير جداً (أقل من 100 حرف)")
    except HTTPException:
        raise
    except Exception as e:
        print("PDF extraction error:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"فشل استخراج النص: {str(e)}")

    # 2. بدء مهمة خلفية
    task_id = str(uuid.uuid4())
    tasks_store[task_id] = {"status": "processing", "result": None}
    background_tasks.add_task(process_curriculum_background, task_id, clean_text, level, instructions)

    return {"success": True, "task_id": task_id}

@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return task

@app.post("/generate-curriculum")
async def generate_curriculum(req: CurriculumRequest):
    if not req.book_text or not req.level:
        raise HTTPException(status_code=400, detail="يجب توفير نص الكتاب والمستوى التعليمي")
    if len(req.book_text) < 500:
        raise HTTPException(status_code=400, detail="نص الكتاب قصير جداً")

    state: CurriculumState = {
        "book_text": req.book_text,
        "level": req.level,
        "instructions": req.instructions,
        "parts": [],
        "titles": [],
        "explanations": [],
        "assessments": [],
        "final_markdown": ""
    }
    try:
        final_state = await curriculum_graph.ainvoke(state)
        return {"success": True, "markdown": final_state["final_markdown"], "titles": final_state["titles"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل توليد المنهج: {str(e)}")

@app.get("/")
async def root():
    return {"status": "AI Curriculum Generator Running", "version": "4.0"}

# ---------- تشغيل ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
"""
خدمة وكلاء الذكاء الاصطناعي لتوليد المناهج التعليمية (LangGraph + FastAPI)
النماذج المستخدمة مجانية بالكامل عبر OpenRouter:
- Google Gemma 4 26B (A4B) للتخطيط والعناوين
- Google Gemma 4 31B للشرح والتقييم
"""

import os
import re
import io
import json
import traceback
import uuid
import asyncio
from typing import TypedDict, List, Dict, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import pdfplumber
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI

# ---------- إعداد OpenRouter (مفتاح واحد لجميع النماذج) ----------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your-free-key")
BASE_URL = "https://openrouter.ai/api/v1"

# إعداد عميل OpenAI متوافق مع OpenRouter
client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=BASE_URL)

# ---------- النماذج المتخصصة لكل وكيل (جميعها مجانية) ----------
AGENT_MODELS = {
    "strategist": "google/gemma-4-26b-a4b-it:free",  # تحليل هيكلي وتخطيط JSON دقيق
    "titler": "google/gemma-4-26b-a4b-it:free",      # صياغة عناوين دقيقة ومنظمة
    "explainer": "google/gemma-4-31b-it:free",       # شرح عربي غني ومبسط
    "assessor": "google/gemma-4-31b-it:free",        # أسئلة تقييمية إبداعية
}

# ---------- تعريف الحالة (State) المتنقلة بين الوكلاء ----------
class CurriculumState(TypedDict):
    book_text: str
    level: str
    instructions: str
    parts: List[str]
    titles: List[str]
    explanations: List[str]
    assessments: List[str]
    final_markdown: str

# ---------- نموذج الطلب للواجهة القديمة ----------
class CurriculumRequest(BaseModel):
    book_text: str
    level: str
    instructions: str = ""

# ---------- هيكل تخزين المهام ----------
class TaskInfo(BaseModel):
    status: str  # "processing", "completed", "failed"
    result: Optional[dict] = None

# ---------- إعداد تطبيق FastAPI ----------
app = FastAPI(title="AI Curriculum Generator", version="4.0")

# حماية CORS: السماح فقط لتطبيق Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ruhulqudus.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# تخزين المهام المؤقت (في الذاكرة)
tasks_store: Dict[str, TaskInfo] = {}

# ---------- دالة مساعدة لتنظيف استخراج JSON ----------
def extract_json(text: str) -> dict:
    """محاولة استخراج JSON من رد النموذج، مع تنظيف الشوائب."""
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
                    sliced = cleaned[start_idx:end_idx+1]
                    return json.loads(sliced)
            raise

# ---------- دالة ذكية لإعادة المحاولة عند خطأ 429 ----------
async def safe_openrouter_call(model: str, messages: list, max_tokens: int = 1500, temperature: float = 0.7):
    """تنفيذ استدعاء OpenRouter مع إعادة المحاولة عند تجاوز الحد."""
    for attempt in range(5):  # حتى 5 محاولات
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                # استخراج مدة الانتظار المقترحة من رسالة الخطأ إن وجدت
                wait_time = 30  # افتراضي 30 ثانية
                # محاولة قراءة retry-after من النص
                # يمكن تحسينها لاحقًا
                print(f"Rate limited for {model}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
                continue
            else:
                raise
    raise Exception(f"فشل الاتصال بالنموذج {model} بعد عدة محاولات")

# ---------- دوال الوكلاء (Agents) ----------

async def strategist_agent(state: CurriculumState) -> CurriculumState:
    """الوكيل المخطط: يقسم الكتاب إلى 10 أجزاء تعليمية منطقية"""
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
    raw = await safe_openrouter_call(
        model=AGENT_MODELS["strategist"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    try:
        data = extract_json(raw)
        state["parts"] = data["parts"]
    except:
        state["parts"] = [f"الجزء {i+1}" for i in range(10)]
    return state

async def titler_agent(state: CurriculumState) -> CurriculumState:
    """وكيل العناوين: صياغة عناوين جذابة ومناسبة تربوياً"""
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
    raw = await safe_openrouter_call(
        model=AGENT_MODELS["titler"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1000,
    )
    try:
        data = extract_json(raw)
        state["titles"] = data["titles"]
    except:
        state["titles"] = [f"الدرس {i+1}" for i in range(10)]
    return state

async def explainer_agent(state: CurriculumState) -> CurriculumState:
    """وكيل الشرح: كتابة شروحات تفصيلية بأسلوب تربوي"""
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
        resp_text = await safe_openrouter_call(
            model=AGENT_MODELS["explainer"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,
        )
        explanations.append(resp_text)
        # إضافة تأخير بسيط لتخفيف الضغط على OpenRouter المجاني
        await asyncio.sleep(1)
    state["explanations"] = explanations
    return state

async def assessor_agent(state: CurriculumState) -> CurriculumState:
    """وكيل التقييم: توليد أسئلة تقييمية لكل درس"""
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
        resp_text = await safe_openrouter_call(
            model=AGENT_MODELS["assessor"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500,
        )
        assessments.append(resp_text)
        await asyncio.sleep(1)
    state["assessments"] = assessments
    return state

async def formatter_agent(state: CurriculumState) -> CurriculumState:
    """وكيل التنسيق: تجميع المنهج النهائي في Markdown (بدون LLM)"""
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

# ---------- بناء الرسم البياني (Graph) ----------
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

# تهيئة الرسم البياني
curriculum_graph = create_curriculum_graph()

# ---------- معالجة المهمة في الخلفية ----------
async def process_curriculum_background(task_id: str, full_text: str, level: str, instructions: str):
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
        tasks_store[task_id] = TaskInfo(
            status="completed",
            result={
                "success": True,
                "markdown": final_state["final_markdown"],
                "titles": final_state["titles"]
            }
        )
    except Exception as e:
        traceback.print_exc()
        tasks_store[task_id] = TaskInfo(
            status="failed",
            result={"success": False, "detail": str(e)}
        )

# ---------- نقطة النهاية لاستقبال PDF مباشرة ----------
@app.post("/generate-from-pdf")
async def generate_from_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    level: str = Form(...),
    instructions: str = Form("")
):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="يجب رفع ملف PDF")
    
    # استخراج النص
    try:
        pdf_bytes = await file.read()
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            full_text = "".join([page.extract_text() or "" for page in pdf.pages])
        if len(full_text.strip()) < 100:
            raise HTTPException(status_code=400, detail="النص المستخرج قصير جداً (أقل من 100 حرف)")
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"فشل استخراج النص: {str(e)}")
    
    # إنشاء task_id
    task_id = str(uuid.uuid4())
    tasks_store[task_id] = TaskInfo(status="processing", result=None)

    # تشغيل المهمة في الخلفية
    background_tasks.add_task(process_curriculum_background, task_id, full_text, level, instructions)

    return {"success": True, "task_id": task_id, "message": "جاري معالجة المنهج في الخلفية"}

# ---------- نقطة نهاية لفحص حالة المهمة ----------
@app.get("/task-status/{task_id}")
async def get_task_status(task_id: str):
    task = tasks_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="المهمة غير موجودة")
    return task

# ---------- نقطة النهاية القديمة (للتوافق) ----------
@app.post("/generate-curriculum")
async def generate_curriculum(req: CurriculumRequest):
    if not req.book_text or not req.level:
        raise HTTPException(status_code=400, detail="يجب توفير نص الكتاب والمستوى التعليمي")
    
    if len(req.book_text) < 500:
        raise HTTPException(status_code=400, detail="نص الكتاب قصير جداً (أقل من 500 حرف)")

    initial_state: CurriculumState = {
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
        final_state = await curriculum_graph.ainvoke(initial_state)
        return {
            "success": True,
            "markdown": final_state["final_markdown"],
            "titles": final_state["titles"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"فشل توليد المنهج: {str(e)}")

@app.get("/")
async def root():
    return {"status": "AI Curriculum Generator Running", "version": "4.0"}

# ---------- التشغيل ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
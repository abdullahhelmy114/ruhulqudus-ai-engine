"""
خدمة وكلاء الذكاء الاصطناعي لتوليد المناهج التعليمية (LangGraph + FastAPI)
النماذج المستخدمة مجانية بالكامل عبر OpenRouter:
- Google Gemma 4 26B (A4B) للتخطيط والعناوين
- Google Gemma 4 31B للشرح والتقييم
"""

import os
import re
from typing import TypedDict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
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

# ---------- نموذج الطلب ----------
class CurriculumRequest(BaseModel):
    book_text: str
    level: str
    instructions: str = ""

# ---------- إعداد تطبيق FastAPI ----------
app = FastAPI(title="AI Curriculum Generator", version="2.0")

# حماية CORS: السماح فقط لتطبيق Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ruhulqudus.net"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- دالة مساعدة لتنظيف استخراج JSON ----------
def extract_json(text: str) -> dict:
    """محاولة استخراج JSON من رد النموذج، مع تنظيف الشوائب."""
    try:
        # محاولة مباشرة
        return json.loads(text)
    except:
        # محاولة إزالة علامات Markdown إن وُجدت
        cleaned = re.sub(r'```json|```', '', text).strip()
        try:
            return json.loads(cleaned)
        except:
            # محاولة البحث عن أول { أو [ في النص
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
    response = await client.chat.completions.create(
        model=AGENT_MODELS["strategist"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    raw = response.choices[0].message.content
    try:
        data = extract_json(raw)
        state["parts"] = data["parts"]
    except Exception as e:
        # fallback بسيط
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
    response = await client.chat.completions.create(
        model=AGENT_MODELS["titler"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )
    raw = response.choices[0].message.content
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
        resp = await client.chat.completions.create(
            model=AGENT_MODELS["explainer"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000,  # شرح وافي
        )
        explanations.append(resp.choices[0].message.content)
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
        resp = await client.chat.completions.create(
            model=AGENT_MODELS["assessor"],
            messages=[{"role": "user", "content": prompt}],
            temperature=0.8,
            max_tokens=1500,
        )
        assessments.append(resp.choices[0].message.content)
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

# ---------- نقطة النهاية API ----------
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
    return {"status": "AI Curriculum Generator Running", "version": "2.0"}

# ---------- التشغيل ----------
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
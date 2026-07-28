# استخدام صورة بايثون خفيفة
FROM python:3.12-slim

# تعيين دليل العمل
WORKDIR /app

# نسخ ملف المتطلبات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي المشروع
COPY . .

# تعيين منفذ التشغيل
EXPOSE 8000

# تشغيل الخدمة
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
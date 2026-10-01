import os
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from google import genai
from google.genai import types

app = FastAPI()

# تفعيل CORS للسماح لصفحة GitHub Pages بالتواصل مع السيرفر
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# جلب مفاتيح الـ API من متغيرات البيئة
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

groq_client = Groq(api_key=GROQ_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

@app.get("/")
def home():
    return {"status": "running", "message": "Voice Sheet Assistant Backend is Live!"}

@app.post("/webhook/voice-edit")
async def voice_edit(
    audio: UploadFile = File(...),
    headers: str = Form(...)  # يستقبل أسماء أعمدة الجدول بصيغة JSON string
):
    try:
        # 1. حفظ ملف الصوت مؤقتاً للمعالجة
        audio_bytes = await audio.read()
        temp_audio_path = f"/tmp/{audio.filename}"
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)

        # 2. تحويل الصوت لنص عربي عبر Groq Whisper
        with open(temp_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio.filename, file.read()),
                model="whisper-large-v3",
                language="ar",
                response_format="text"
            )

        spoken_text = str(transcription).strip()

        # 3. توجيه Gemini لتحليل الأمر وإرجاع JSON دقيق
        system_instruction = f"""
        أنت مساعد ذكي مخصص لتعديل جداول البيانات.
        أعمدة الجدول المتاحة هي: {headers}
        
        مهمتك: قراءة كلام المندوب وتحويله إلى أمر تعديل دقيق بصيغة JSON فقط بدون أي نصوص أو markdown إضافية.
        
        القواعد:
        1. إذا كان الأمر لتعديل قيمة موجودة (مثال: "عدل خيمة 12 إلى استلمت"):
        {{
            "action": "update",
            "search_col": "<اسم العمود الذي نبحث به>",
            "search_val": "<القيمة التي نبحث عنها>",
            "target_col": "<اسم العمود المراد تعديله>",
            "new_value": "<القيمة الجديدة>"
        }}
        
        2. إذا كان الأمر لإضافة صف جديد (مثال: "سجل خيمة 40 بحالة جديد"):
        {{
            "action": "append",
            "row_data": {{ "<اسم العمود>": "<القيمة>" }}
        }}
        
        تأكد أن أسماء الأعمدة مطابقة تماماً للموجود في قائمة الأعمدة.
        """

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=spoken_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json"
            )
        )

        result_json = json.loads(response.text)
        result_json["transcribed_text"] = spoken_text
        return result_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
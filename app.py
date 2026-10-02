import os
import json
import traceback
import requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/")
def home():
    return {"status": "online", "model": "gemini-3.5-flash-lite"}

@app.post("/webhook/voice-edit")
async def voice_edit(
    audio: UploadFile = File(...),
    headers: str = Form(...)
):
    print("\n--- NEW VOICE COMMAND RECEIVED ---")
    if not GROQ_API_KEY or not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API Keys in Render Environment Variables!")

    temp_audio_path = f"/tmp/{audio.filename}"
    try:
        # 1. حفظ ملف الصوت
        content = await audio.read()
        with open(temp_audio_path, "wb") as f:
            f.write(content)

        # 2. تحويل الصوت لنص عبر Groq Whisper
        groq_client = Groq(api_key=GROQ_API_KEY)
        with open(temp_audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(audio.filename, file.read()),
                model="whisper-large-v3",
                language="ar",
                response_format="text"
            )

        spoken_text = str(transcription).strip()
        print(f"Spoken text: '{spoken_text}'")

        if not spoken_text:
            raise HTTPException(status_code=400, detail="لم يتم التقاط أي صوت واضح")

        # 3. توجيه Gemini 3.5 Flash-Lite لاستخراج الـ JSON
        system_prompt = f"""
أنت مساعد ذكي لإدارة وتعديل جداول بيانات المخيم.
أعمدة الجدول المتاحة هي:
{headers}

المستخدم قال: "{spoken_text}"

مهمتك: تحديد ما يريد تعديله ومطابقته بدقة مع أعمدة الجدول.
أرجع فقط كائن JSON خالص بالصيغة التالية دون أي كود Markdown أو نصوص إضافية:
{{
    "action": "update",
    "search_col": "<اسم العمود الأنسب للبحث مثل رقم الهوية أو اسم رب الأسرة رباعي أو No>",
    "search_val": "<القيمة التي نبحث عنها>",
    "target_col": "<اسم العمود المراد تعديل قيمته من القائمة بالضبط>",
    "new_value": "<القيمة الجديدة>"
}}
"""

        gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.5-flash-lite:generateContent?key={GEMINI_API_KEY}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": system_prompt}]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json"
            }
        }

        gemini_res = requests.post(gemini_url, json=payload, timeout=20)
        if gemini_res.status_code != 200:
            print(f"Gemini API Error: {gemini_res.text}")
            raise HTTPException(status_code=500, detail=f"Gemini Error: {gemini_res.text}")

        res_data = gemini_res.json()
        raw_text = res_data["candidates"][0]["content"]["parts"][0]["text"]
        print(f"Gemini JSON: {raw_text}")

        result_json = json.loads(raw_text)
        result_json["transcribed_text"] = spoken_text
        return result_json

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

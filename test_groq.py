import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    print("❌ GROQ_API_KEY is not set in .env")
    exit(1)

client = Groq(api_key=api_key)

print("🔍 Fetching active models available for your API key...\n")
models_list = client.models.list()

chat_models = [m.id for m in models_list.data if not "whisper" in m.id and not "guard" in m.id]

print(f"📋 Found {len(chat_models)} chat models:")
for m in chat_models:
    print(f"  - {m}")

print("\n🧪 Testing each model with a sample prompt...")
working_models = []

for model_id in chat_models:
    try:
        res = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": "Write 5 words"}],
            max_tokens=20
        )
        print(f"  ✅ WORKING: {model_id}")
        working_models.append(model_id)
    except Exception as e:
        print(f"  ❌ FAILED ({model_id}): {e}")

print("\n" + "="*50)
if working_models:
    print(f"🏆 Pick one of these working models: {working_models}")
else:
    print("⚠️ No chat models responded successfully.")
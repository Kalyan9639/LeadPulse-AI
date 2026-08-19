import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

audio_file = "test_call.mp4"

with open(audio_file, "rb") as file:
    transcription = client.audio.transcriptions.create(
        file=(audio_file, file.read()),
        model="whisper-large-v3-turbo",
        response_format="text"
    )

print("\n===== TRANSCRIPT =====\n")
print(transcription)
import asyncio
import base64
import io
import os
import json
import pyaudio
import sounddevice as sd
import soundfile as sf
import httpx
from dotenv import load_dotenv
from sarvamai import AsyncSarvamAI, SarvamAI

# =============================
# CONFIG
# =============================
load_dotenv()
SARVAM_API_KEY = os.getenv('SARVAM_API_KEY')
CHAT_URL = "http://127.0.0.1:8000/chat"

RATE = 16000
CHUNK = 1024

# =============================
# CLIENTS
# =============================
client_stt = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)
client_tts = SarvamAI(api_subscription_key=SARVAM_API_KEY)


# =============================
# TTS (Runs in Thread)
# =============================
def speak_gujarati(text):
    try:
        response = client_tts.text_to_speech.convert(
            text=text,
            target_language_code="gu-IN",
            model="bulbul:v3",
            speaker="simran",
            pace=1.0
        )

        audio_bytes = base64.b64decode(response.audios[0])
        data, fs = sf.read(io.BytesIO(audio_bytes))

        sd.play(data, fs)
        sd.wait()

    except Exception as e:
        print(f"TTS Error: {e}")


# =============================
# MAIN VOICE LOOP
# =============================
async def run_voice_agent():
    session_id = "user_mokshank_007"
    transcript_buffer = ""

    #1️⃣ PLAY GREETING FIRST
    greeting_text = "નમસ્તે! હું સમયસેતુ AI છું. હું તમારી કેવી રીતે મદદ કરી શકું?"
    print(f"🤖 AI Greeting: {greeting_text}")
    try:
        # We use to_thread so the code doesn't freeze while playing audio
        await asyncio.to_thread(speak_gujarati, greeting_text)
    except Exception as e:
        print(f"Initial Greeting Error: {e}")

    async with client_stt.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="gu-IN",
        sample_rate=RATE,
        high_vad_sensitivity=True
    ) as ws:

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )

        print("\n--- SamaySetu AI Listening (Speak Gujarati) ---")

        try:
            while True:

                # 1️⃣ Send mic audio to STT
                data = stream.read(CHUNK, exception_on_overflow=False)
                await ws.transcribe(
                    audio=base64.b64encode(data).decode("utf-8")
                )

                try:
                    response = await asyncio.wait_for(ws.recv(), timeout=0.05)

                    if not response:
                        continue
                    
                    print(response)

                    # Only process transcription data events
                    if hasattr(response, "type") and response.type == "data":
                        print("entered 1st if")
                        if hasattr(response, "data") and response.data.transcript:
                            print("entered 2nd if")
                            user_text = response.data.transcript.strip()

                            if not user_text:
                                continue
                            
                            print(f"\n🗣 Final Sentence: {user_text}")

                            # Send to FastAPI
                            print("calling the server")

                            async with httpx.AsyncClient(timeout=30.0) as client:
                                brain_res = await client.post(
                                    CHAT_URL,
                                    json={
                                        "session_id": session_id,
                                        "text": user_text
                                    }
                                )

                            await asyncio.to_thread(speak_gujarati, "કૃપા કરીને બે ક્ષણ રાહ જો જો...")

                            if brain_res.status_code == 200:
                                ai_reply = brain_res.json().get("reply")
                                print(f"🤖 AI: {ai_reply}")

                                await asyncio.to_thread(speak_gujarati, ai_reply)

                                print("\n--- Listening again... ---\n")

                            else:
                                print("Server Error:", brain_res.text)

                except asyncio.TimeoutError:
                    continue

        except KeyboardInterrupt:
            print("Stopping...")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()


# =============================
# ENTRY POINT
# =============================
if __name__ == "__main__":
    asyncio.run(run_voice_agent())
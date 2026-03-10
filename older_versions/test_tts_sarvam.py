import io
import soundfile as sf
import sounddevice as sd
import base64
from sarvamai import SarvamAI
import os
from dotenv import load_dotenv

# Load environment variables From project root
env_path = '.env'
load_dotenv(dotenv_path=env_path)

sarvam_api_key = os.getenv('SARVAM_API_KEY')

# 1. Initialize
client = SarvamAI(api_subscription_key=sarvam_api_key)

def test_gujarati_tts(text_to_speak):
    print(f"Generating voice for: {text_to_speak}")
    
    try:
        # 2. Convert Text to Speech
        response = client.text_to_speech.convert(
            text=text_to_speak,
            target_language_code="gu-IN",
            model="bulbul:v3",
            speaker="ritu",
            pace=1.0
        )
        
        # 3. Decode the Base64 audio string from Sarvam
        # Sarvam returns a list of audio strings; we take the first one
        audio_base64 = response.audios[0] 
        audio_bytes = base64.b64decode(audio_base64)
        
        # 4. Load the bytes into an audio object and play
        # We use io.BytesIO to treat the bytes like a file
        data, fs = sf.read(io.BytesIO(audio_bytes))
        
        print("Speaking now...")
        sd.play(data, fs)
        sd.wait()  # Wait until the audio finishes playing
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    guj_text = "નમસ્તે મોક્ષ આંક, તમારી ૧૧ વાગ્યાની અપોઇન્ટમેન્ટ બુક થઈ ગઈ છે."
    test_gujarati_tts(guj_text)


import time
import io
import os
import speech_recognition as sr
from faster_whisper import WhisperModel
import datetime

# 1. Configuration for your 1650 GPU
# We use 'int8_float16' to keep memory usage around 1.5GB - 2GB.
model_size = "medium" 
device = "cuda"
compute_type = "int8_float16"

print(f"Loading {model_size} model on {device}...")
model = WhisperModel(model_size, device=device, compute_type=compute_type)

# 2. Setup Microphone
recognizer = sr.Recognizer()
mic = sr.Microphone()

guj_prompt = "મારું નામ મોક્ષાંક્ષ છે. તમે કેમ છો?"

print("\n--- Model Ready! ---")
print("બોલવાનું શરૂ કરો (Start speaking in Gujarati)...")

try:
    with mic as source:
        # Adjust for background noise once
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
        while True:
            print("\nસાંભળી રહ્યો છું... (Listening...)")
            audio = recognizer.listen(source)
            
            # Convert audio to a format Whisper understands (WAV)
            audio_data = io.BytesIO(audio.get_wav_data())
            
            # 3. Transcribe with Gujarati force-detection
            segments, info = model.transcribe(
                audio_data, 
                beam_size=5, 
                language="gu", 
                task="transcribe",
                initial_prompt=guj_prompt
            )

            # 4. Display Output
            for segment in segments:
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                print(f"[{timestamp}] {segment.text}")

except KeyboardInterrupt:
    print("\nSystem stopped by user.")
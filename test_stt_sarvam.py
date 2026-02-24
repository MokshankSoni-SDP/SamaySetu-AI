import asyncio
import pyaudio
import base64
import json
from sarvamai import AsyncSarvamAI
import os
from dotenv import load_dotenv

# Load environment variables From project root
env_path = '.env'
load_dotenv(dotenv_path=env_path)

sarvam_api_key = os.getenv('SARVAM_API_KEY')

# Configuration
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
# IMPORTANT: Use 16000 for local mic tests.
RATE = 16000 

async def debug_sarvam_stream():
    # Initialize with your key
    client = AsyncSarvamAI(api_subscription_key=sarvam_api_key)
    
    # Use 'transcribe' mode and high sensitivity for better local testing
    async with client.speech_to_text_streaming.connect(
        model="saaras:v3",
        mode="transcribe",
        language_code="gu-IN",
        sample_rate=RATE,
        high_vad_sensitivity=True  # Helps detect quiet speech
    ) as ws:
        
        p = pyaudio.PyAudio()
        
        # DEBUG: Check if the default mic is actually there
        try:
            device_info = p.get_default_input_device_info()
            print(f"--- Using Microphone: {device_info['name']} ---")
        except Exception as e:
            print(f"--- Error finding microphone: {e} ---")
            return

        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, 
                        input=True, frames_per_buffer=CHUNK_SIZE)
        
        print("Listening... Speak in Gujarati now.")
        
        try:
            while True:
                data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                audio_base64 = base64.b64encode(data).decode("utf-8")
                
                # Send the chunk
                await ws.transcribe(audio=audio_base64)
                
                # Try to receive a response
                try:
                    # Use a short timeout so the loop doesn't freeze
                    response = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    if response:
                        print(f"Result: {response}")
                except asyncio.TimeoutError:
                    # No text yet, just keep sending audio
                    continue
                    
        except KeyboardInterrupt:
            print("--- Stopped by User ---")
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.run(debug_sarvam_stream())
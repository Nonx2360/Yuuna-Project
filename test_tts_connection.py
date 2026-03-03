import requests
import os

URL = "http://localhost:5000/api/tts"
OUTPUT_FILE = "test_output.wav"

def test_tts():
    print("Testing Yuuna-Project TTS Endpoint...")
    payload = {
        "text": "Hello, this is a test of the new GPT-SoVITS integration.",
        "target_lang": "en"
    }
    
    try:
        response = requests.post(URL, json=payload)
        
        if response.status_code == 200:
            with open(OUTPUT_FILE, "wb") as f:
                f.write(response.content)
            print(f"✅ Success! Audio saved to {OUTPUT_FILE}")
        elif response.status_code == 503:
            print("❌ Failure: GPT-SoVITS API is not running. (Expected if not started yet)")
        else:
            print(f"❌ Failure: Status {response.status_code}, {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    test_tts()

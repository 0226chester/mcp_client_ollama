import requests
import time

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:14b"  # Choose a smaller model for faster response

def test_connection():
    print(f"Testing connection to Ollama API with model: {MODEL_NAME}")
    print("Sending prompt...")
    
    start_time = time.time()
    
    try:
        # Use the simplest possible request
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": MODEL_NAME,
                "prompt": "Hello, what is Ollama?",
                "stream": False
            }
        )
        
        end_time = time.time()
        
        print(f"\nStatus code: {response.status_code}")
        print(f"Response time: {end_time - start_time:.2f} seconds")
        
        if response.status_code == 200:
            print("\nConnection successful! ✅")
            print("\n--- Response Preview ---")
            # Just print the first few characters to avoid parsing issues
            print(response.text[:500] + "..." if len(response.text) > 500 else response.text)
        else:
            print("Connection failed! ❌")
            print(response.text)
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("==================================================")
    print("SIMPLE OLLAMA API TEST")
    print("==================================================")
    test_connection()
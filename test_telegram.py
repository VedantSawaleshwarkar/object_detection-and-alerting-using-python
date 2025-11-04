import requests

def test_telegram():
    token = "8164979830:AAGMPFSvc-yGPfUTxIfFG2AV_70IlrQ9yCk"
    chat_id = "7361910235"
    
    # Test 1: Send a simple message
    print("Sending test message...")
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {
        "chat_id": chat_id,
        "text": "Test message from your bot!"
    }
    
    try:
        response = requests.post(url, data=data)
        print("Status Code:", response.status_code)
        print("Response:", response.json())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_telegram()

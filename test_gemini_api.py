import requests
import time
import hmac
import hashlib
import base64
import json
API_KEY = "your_api_key"
API_SECRET = "your_api_secret"

url = "https://api.gemini.com/v1/balances"
nonce = str(int(time.time() * 1000))
payload = {"request": "/v1/balances", "nonce": nonce}
encoded_payload = base64.b64encode(json.dumps(payload).encode())
signature = hmac.new(API_SECRET.encode(), encoded_payload, hashlib.sha384).hexdigest()

headers = {
    "X-GEMINI-APIKEY": API_KEY,
    "X-GEMINI-PAYLOAD": encoded_payload,
    "X-GEMINI-SIGNATURE": signature
}

response = requests.post(url, headers=headers)
print(response.json())

import requests
import json

prompt = "Panda is a bear in china."
data= { "text_input": prompt }
body = json.dumps(data).encode("utf-8")
headers = {"Content-Type":  "application/json"}

url = "http://localhost:8080/v2/models/model/generate"
response = requests.post(url, data=json.dumps(data), headers=headers)
logits = response.json()['logits']
print(f"response: status_code: {response.status_code} logits: {len(logits)}")
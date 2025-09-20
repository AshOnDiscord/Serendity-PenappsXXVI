import requests

url = "http://localhost:5000/all_data"

response = requests.get(url)

if response.status_code == 200:
    print("Response:")
    print(response.json())  # Assuming the server returns JSON
else:
    print(f"Error: {response.status_code}")
    print(response.text)

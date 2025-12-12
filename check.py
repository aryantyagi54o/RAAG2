from google.genai import Client

# API key configure
client = Client(api_key="AIzaSyBFVFD9-7eIlQxLVp3PKfOoV-2-xML_pTg")

# Chat session create
chat = client.chats.create(model="gemini-2.5-flash")

# Send message
response = chat.send_message("Hello, how are you?")
print(response.text)

# Continue chat
response2 = chat.send_message("Summarize what you just said.")
print(response2.text)


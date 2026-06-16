from gradio_client import Client, handle_file
import sys
try:
    client = Client("http://127.0.0.1:7860/")
    res = client.predict(
        handle_file("data/videos/SENT001_S06_R01_F.mp4"),
        api_name="/predict"
    )
    print("RESULT:", res)
except Exception as e:
    print("CLIENT ERROR:", e)

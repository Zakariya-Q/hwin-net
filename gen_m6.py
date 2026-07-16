import base64
import sys

content_b64 = """
REPLACE_ME

decoded = base64.b64decode(content_b64)

with open(r"C:/Users/lenovo/hwin_net/models/no_leakage.py", "wb") as f2:
    f2.write(decoded)

print("File written successfully")
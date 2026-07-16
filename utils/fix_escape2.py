with open("config.py", "r") as f:
    content = f.read()
content = content.replace('print(f"\\Saved config to', 'print(f"Saved config to')
with open("config.py", "w") as f:
    f.write(content)
print("Fixed escape sequence")

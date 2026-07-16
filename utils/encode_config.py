import base64

# Read the current config.py
with open('config.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Encode and write
encoded = base64.b64encode(content.encode('utf-8')).decode('ascii')
with open('config_b64.txt', 'w') as f:
    f.write(encoded)
print('Encoded length:', len(encoded))

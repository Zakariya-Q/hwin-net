import os, re
for root, dirs, files in os.walk('.'):
    dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.venv', 'build', 'dist']]
    for f in files:
        if f.endswith('.py'):
            p = os.path.join(root, f)
            try:
                txt = open(p, encoding='utf-8', errors='ignore').read()
                if re.search(r'(github|api_?key|secret|password)\s*=\s*[\'"][^\'"]{20,}', txt, re.IGNORECASE):
                    print(p)
            except: pass

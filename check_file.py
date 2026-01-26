import os

try:
    exists = os.path.exists('src/main.py')
    print(f"File exists: {exists}")
    if exists:
        with open('src/main.py', 'r', encoding='utf-8') as f:
            content = f.read()
            has_new = '/api/auth/signup' in content
            has_old = '/auth/register' in content
            
            print(f"Has /api/auth/signup: {has_new}")
            print(f"Has /auth/register: {has_old}")
            print(f"File Size: {len(content)}")
    
except Exception as e:
    print(f"Error: {e}")

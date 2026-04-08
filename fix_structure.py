import os

# 1. Update server/app.py
app_path = 'server/app.py'
if os.path.exists(app_path):
    with open(app_path, 'r') as f:
        content = f.read()
    
    # Update startup logic
    old_startup = 'if __name__ == "__main__":\n    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)'
    new_startup = 'def main():\n    import uvicorn\n    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)\n\nif __name__ == "__main__":\n    main()'
    content = content.replace(old_startup, new_startup)
    
    with open(app_path, 'w') as f:
        f.write(content)
    print("Updated server/app.py startup logic.")

# 2. Update pyproject.toml
toml_path = 'pyproject.toml'
with open(toml_path, 'r') as f:
    toml_content = f.read()

# Add openenv-core
if 'openenv-core' not in toml_content:
    toml_content = toml_content.replace('"scipy"', '"scipy",\n    "openenv-core>=0.2.0"')

# Add [project.scripts]
if '[project.scripts]' not in toml_content:
    toml_content += '\n[project.scripts]\nserver = "server.app:main"\n'

with open(toml_path, 'w') as f:
    f.write(toml_content)
print("Updated pyproject.toml with openenv-core and [project.scripts].")

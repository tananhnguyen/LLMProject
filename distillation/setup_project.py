# setup_project.py

import os

folders = [
    "data",
    "models/student",
    "models/teacher",
    "models/merged",
    "scripts",
    "utils",
    "agent"
]

files = {
    "scripts/train_student.py": "# Training script for student model\n",
    "scripts/train_teacher.py": "# Training script for teacher model\n",
    "scripts/train_merged.py": "# Training script for merged model\n",
    "scripts/evaluate.py": "# Evaluation script\n",
    "scripts/inference.py": "# Inference script\n",
    "utils/data_utils.py": "# Data utility functions\n",
    "utils/model_utils.py": "# Model utility functions\n",
    "agent/rag_agent.py": "# RAG agent code\n",
    "requirements.txt": "# Add your dependencies here\n",
    "README.md": "# LLM NLP Class Project\n"
}

for folder in folders:
    os.makedirs(folder, exist_ok=True)

for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Project structure created successfully.")
import os

# Directory structure
base_dir = "sofmattress_intent_project1"

folders = [
    base_dir,
    f"{base_dir}/data",
    f"{base_dir}/model",
    f"{base_dir}/scripts",
    f"{base_dir}/app"
]

files = {
    f"{base_dir}/README.md": "# SOF Mattress Intent Classification and Response Generation Project\n",
    f"{base_dir}/requirements.txt": """\
transformers
datasets
scikit-learn
pandas
numpy
matplotlib
seaborn
torch
tensorflow
langchain
langchain-community
streamlit
huggingface-hub
nltk
gensim
""",
    f"{base_dir}/scripts/__init__.py": "",
    f"{base_dir}/app/__init__.py": "",
    f"{base_dir}/data/README.md": "Place raw and processed CSV datasets here.\n",
}

# Create directories and files
for folder in folders:
    os.makedirs(folder, exist_ok=True)

for filepath, content in files.items():
    with open(filepath, "w") as f:
        f.write(content)

"Project structure created."

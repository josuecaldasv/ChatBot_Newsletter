import json
import os
import docx2txt

def load_config(config_path="configs/config.json"):
    """
    Load configuration settings from a JSON file.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config

def load_text(file_path):
    """
    Extract text from a .docx file.
    """
    if os.path.splitext(file_path)[1].lower() == ".docx":
        return docx2txt.process(file_path)
    else:
        raise ValueError("Currently, only .docx files are supported.")

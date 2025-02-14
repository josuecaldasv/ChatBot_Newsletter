import os
from utils import base_utils as bu
from utils.retrieval_utils import (
    load_model,
    get_embeddings
)

def main():
    """
    Main function to generate embeddings for all .docx files in the input folder
    based on configurations provided in config.json.
    """
    config = bu.load_config("configs/config.json")

    os.makedirs(config["embeddings"]["output_dir"], exist_ok=True)
    
    model_name = config["embeddings"]["model_name"]
    input_path = config["paths"]["input_path"]
    output_dir = config["embeddings"]["output_dir"]
    splitter_type = config["splitter"]["type"]
    chunk_size = config["splitter"]["chunk_size"]
    chunk_overlap = config["splitter"]["chunk_overlap"]
    retrieval_model = load_model(model_name)

    get_embeddings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_name=model_name,
        input_path=input_path,
        output_dir=output_dir,
        splitter_type=splitter_type,
        retrieval_model=retrieval_model
    )

if __name__ == "__main__":
    main()

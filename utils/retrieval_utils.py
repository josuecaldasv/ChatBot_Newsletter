import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import os

from utils import base_utils as bu

def load_model(model_name):
    """
    Objective:
        Load a SentenceTransformer model with the appropriate device (CUDA if available).

    Parameters:
        model_name (str): The name or path of the SentenceTransformer model.

    Output:
        SentenceTransformer: Loaded model ready for encoding.
    """
    if torch.cuda.is_available():
        device = "cuda"
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("CUDA is not available. Using CPU.")
    
    return SentenceTransformer(model_name, device=device)

def get_text_splitter(splitter_type, chunk_size, chunk_overlap):
    """
    Objective:
        Retrieve the appropriate text splitter based on a specified type.

    Parameters:
        splitter_type (str)  : The type of splitter ('recursive', 'tokens', 'semantic').
        chunk_size (int)     : The size of each text chunk.
        chunk_overlap (int)  : The overlap between consecutive text chunks.

    Output:
        A text splitter object compatible with LangChain document splitting.
    """
    if splitter_type == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
    elif splitter_type == "tokens":
        return CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    elif splitter_type == "semantic":
        embeddings_model = HuggingFaceEmbeddings(model_name=bu.config["embeddings"]["model_name"])
        return SemanticChunker(embeddings=embeddings_model)
    else:
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

def get_embeddings(chunk_size, chunk_overlap, model_name, input_path, output_dir, splitter_type, retrieval_model):
    """
    Objective:
        Generate embeddings for .docx files by splitting them into chunks and encoding them using a specified model.
    
    Parameters:
        chunk_size (int)           : The size of each text chunk.
        chunk_overlap (int)        : The overlap between consecutive text chunks.
        model_name (str)           : The name of the model to use for generating embeddings.
        input_path (str)           : The path pattern for input .docx files.
        output_dir (str)           : The directory to save the generated embeddings.
        splitter_type (str)        : The type of splitter ('recursive', 'tokens', 'semantic').
        retrieval_model            : The SentenceTransformer (or similar) model used to encode text.
    
    Output:
        None: The function saves the embeddings to HDF5 files in the specified output directory.
    """

    text_splitter = get_text_splitter(splitter_type, chunk_size, chunk_overlap)

    doc_files = glob.glob(input_path)
    if not doc_files:
        print(f"No .docx files found in path: {input_path}")
        return

    emb_files = glob.glob(os.path.join(output_dir, "*.h5"))
    for file in emb_files:
        filename_without_ext = os.path.splitext(os.path.basename(file))[0]
        corresponding_doc = os.path.join(os.path.dirname(input_path), filename_without_ext + ".docx")
        if not os.path.exists(corresponding_doc):
            print(f"Embeddings file {file} has no corresponding .docx. Deleting it.")
            os.remove(file)

    for file in tqdm(doc_files, desc="Processing files", total=len(doc_files)):
        file_name = os.path.basename(file)
        output_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.h5")

        if os.path.exists(output_file):
            print(f"Embeddings already exist for {file_name}. Skipping...")
            continue

        print(f"Generating embeddings for: {file_name}")
        text = bu.load_text(file)
        
        splitted_docs = text_splitter.split_text(text)

        embeddings_list = []
        content_list = []

        for segment in splitted_docs:
            embeddings = retrieval_model.encode(segment)
            embeddings_list.append(embeddings)
            content_list.append(segment)
        
        embeddings_df = pd.DataFrame(embeddings_list)
        embeddings_df["segment_content"] = content_list
        embeddings_df["model_name"] = model_name
        embeddings_df["segment_content"] = embeddings_df["segment_content"].astype(str)
        embeddings_df["model_name"] = embeddings_df["model_name"].astype(str)

        embeddings_df.to_hdf(output_file, key="df", mode="w", format="table")
    
def search_query(query, corpus_embeddings, retrieval_model, segment_contents, top_k=5):
    """
    Objective:
        Search for the top-k most similar text segments to a given query using precomputed embeddings.
    
    Parameters:
        query (str)                      : The search query string.
        corpus_embeddings (torch.Tensor) : The tensor containing embeddings of text segments.
        retrieval_model                  : The model used to encode the query into an embedding.
        segment_contents (list)          : The list of text segment contents corresponding to the embeddings.
        top_k (int)                      : The number of top similar segments to return.
    
    Output:
        tuple: A tuple containing:
            - top_segments (list)     : The list of top-k most similar text segments.
            - top_similarities (list) : The list of similarity scores for the top-k segments.
    """

    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarity_scores = retrieval_model.similarity(query_embedding, corpus_embeddings)[0]

    top_similarities, topk_indices = torch.topk(similarity_scores, k=top_k)
    top_segments = [segment_contents[idx] for idx in topk_indices]

    return top_segments, top_similarities

def load_embeddings(embeddings_dir):
    """
    Objective:
        Load embeddings and associated metadata from HDF5 files in a specified directory.
    
    Parameters:
        embeddings_dir (str): The directory containing the HDF5 embedding files.
    
    Output:
        dict: A dictionary containing:
            - embeddings (torch.Tensor) : A tensor of all loaded embeddings.
            - segment_contents (list)   : A list of text segments corresponding to the embeddings.
            - num_documents (int)       : The number of unique documents processed.
            - num_segment_contents (int): The total number of text segments.
    """

    embeddings_list = []
    segment_contents_list = []
    model_names_set = set()

    num_documents = 0
    for file_path in glob.glob(os.path.join(embeddings_dir, "*.h5")):
        num_documents += 1
        embeddings_df = pd.read_hdf(file_path, key="df")
        embeddings = embeddings_df.iloc[:, :-2].values
        segment_contents = embeddings_df["segment_content"].values
        model_name = embeddings_df["model_name"].values[0]

        embeddings_list.extend(embeddings)
        segment_contents_list.extend(segment_contents)
        model_names_set.add(model_name)
    
    if not embeddings_list:
        print(f"No embedding files found in: {embeddings_dir}")
        return {}

    embeddings_array = np.array(embeddings_list)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embeddings_tensor = torch.tensor(embeddings_array, dtype=torch.float32, device=device)

    num_segment_contents = len(segment_contents_list)
    if len(model_names_set) == 1:
        model_name = model_names_set.pop()
    else:
        model_name = "Multiple Models"

    return {
        "embeddings": embeddings_tensor,
        "segment_contents": segment_contents_list,
        "num_documents": num_documents,
        "num_segment_contents": num_segment_contents,
        "model_name": model_name
    }

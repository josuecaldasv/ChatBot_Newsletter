import torch
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm
import pandas as pd
import numpy as np
import glob
import os
import re

from utils import base_utils as bu


def load_model(model_name):
    """
    Load a SentenceTransformer model.
    """
    return SentenceTransformer(model_name, device="cpu")


def get_text_splitter(splitter_type, chunk_size, chunk_overlap):
    """
    Retrieve the appropriate text splitter based on a specified type.
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
    

def is_title_line(line: str) -> bool:
    """
    Checks if a given line is fully uppercase and ends with '(DD/MM/YYYY)'.
    """
    if line != line.upper():
        return False

    pattern = re.compile(r"\(\d{2}\/\d{2}\/\d{4}\)$")
    return bool(pattern.search(line))


def parse_news_blocks(text: str):
    """
    Splits the entire .docx text into multiple news blocks by identifying lines
    that are fully uppercase AND end with a date '(DD/MM/YYYY)' as the block title.
    Everything else until the next title line is considered the block content.
    """
    lines = text.split("\n")
    blocks = []
    current_block = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            if current_block and current_block["content"]:
                current_block["content"] += "\n"
            continue

        if is_title_line(line_stripped):
            if current_block:
                blocks.append(current_block)

            current_block = {
                "title": line_stripped,
                "content": ""
            }
        else:
            if current_block:
                if current_block["content"] == "":
                    current_block["content"] = line_stripped
                else:
                    current_block["content"] += "\n" + line_stripped
            else:
                continue

    if current_block:
        blocks.append(current_block)

    return blocks


def get_embeddings(chunk_size, chunk_overlap, model_name, input_path, output_dir, splitter_type, retrieval_model):
    """
    Generate embeddings for .docx files by splitting them into chunks and encoding them using a specified model.
    Also stores the "source_title" for each chunk.
    """

    text_splitter = get_text_splitter(splitter_type, chunk_size, chunk_overlap)

    doc_files = glob.glob(input_path)
    if not doc_files:
        print(f"No .docx files found in path: {input_path}")
        return
    
    os.makedirs(output_dir, exist_ok=True)

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
        blocks = parse_news_blocks(text)

        embeddings_list = []
        content_list = []
        source_title_list = []

        for block in blocks:
            title = block["title"]
            content = block["content"]

            splitted_docs = text_splitter.split_text(content)
            for chunk in splitted_docs:

                embeddings = retrieval_model.encode(chunk)
                embeddings_list.append(embeddings)
                content_list.append(chunk)
                source_title_list.append(title)

        embeddings_df = pd.DataFrame(embeddings_list)
        embeddings_df["segment_content"] = content_list
        embeddings_df["source_title"] = source_title_list
        embeddings_df["model_name"] = model_name
        embeddings_df["segment_content"] = embeddings_df["segment_content"].astype(str)
        embeddings_df["source_title"] = embeddings_df["source_title"].astype(str)
        embeddings_df["model_name"] = embeddings_df["model_name"].astype(str)

        embeddings_df.to_hdf(output_file, key="df", mode="w", format="table")
    

def search_query(query, corpus_embeddings, retrieval_model, segment_contents, segment_sources, top_k=5):
    """
    Search for the top-k most similar text segments to a given query using precomputed embeddings.
    """

    query_embedding = retrieval_model.encode(query, convert_to_tensor=True)
    similarity_scores = retrieval_model.similarity(query_embedding, corpus_embeddings)[0]

    top_similarities, topk_indices = torch.topk(similarity_scores, k=top_k)
    top_segments = [segment_contents[idx] for idx in topk_indices]
    top_srcs = [segment_sources[idx] for idx in topk_indices]

    return top_segments, top_similarities, top_srcs


def load_embeddings(embeddings_dir):
    """
    Load embeddings and associated metadata from HDF5 files in a specified directory.
    """

    embeddings_list = []
    segment_contents_list = []
    segment_sources_list = []
    model_names_set = set()

    num_documents = 0
    for file_path in glob.glob(os.path.join(embeddings_dir, "*.h5")):
        num_documents += 1
        embeddings_df = pd.read_hdf(file_path, key="df")
        num_embedding_cols = embeddings_df.shape[1] - 3
        embeddings = embeddings_df.iloc[:, :num_embedding_cols].values

        segment_contents = embeddings_df["segment_content"].values
        source_titles = embeddings_df["source_title"].values
        model_name = embeddings_df["model_name"].values[0]

        embeddings_list.extend(embeddings)
        segment_contents_list.extend(segment_contents)
        segment_sources_list.extend(source_titles)
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
        "segment_sources": segment_sources_list,
        "num_documents": num_documents,
        "num_segment_contents": num_segment_contents,
        "model_name": model_name
    }
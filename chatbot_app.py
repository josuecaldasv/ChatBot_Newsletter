import sys
from utils import base_utils as bu
from utils.retrieval_utils import load_embeddings, load_model, search_query
from utils.generation_utils import generate_answer

def main():
    """
    Command-line chatbot app that retrieves relevant text segments
    and generates an answer using an OpenAI model.
    """
    config = bu.load_config("configs/config.json")

    embeddings_data = load_embeddings(config["embeddings"]["output_dir"])
    if not embeddings_data:
        print("No embeddings found. Please run generate_embeddings.py first.")
        sys.exit(1)

    corpus_embeddings = embeddings_data["embeddings"]
    segment_contents = embeddings_data["segment_contents"]
    segment_sources = embeddings_data.get("segment_sources", [])
    model_name = embeddings_data.get("model_name", "Unknown Model")

    retrieval_model = load_model(model_name)

    while True:
        user_query = input("\nDigite sua pergunta (ou 'sair' para encerrar): ")
        if user_query.lower() in ["sair", "encerrar"]:
            print("Encerrando o chatbot...")
            break

        top_k = config["retrieve"]["top_k"]
        top_segments, top_similarities, top_sources = search_query(
            user_query,
            corpus_embeddings,
            retrieval_model,
            segment_contents,
            segment_sources,
            top_k=top_k
        )

        print("\nSegmentos recuperados:\n-------------------")
        for i, (segment, source) in enumerate(zip(top_segments, top_sources)):
            score = float(top_similarities[i])
            print(f"Segmento {i+1} (similaridade={score:.4f}) da fonte '{source}':\n{segment}\n")

        context = "\n".join(top_segments)
        answer = generate_answer(user_query, context, config)

        unique_references = set(top_sources)
        references_text = "\n".join(unique_references)

        print("Resposta do chatbot:\n--------------------")
        print(answer)
        print("\nReferências:\n--------------------")
        print(references_text)


if __name__ == "__main__":
    main()
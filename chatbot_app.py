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
    model_name = embeddings_data.get("model_name", "Unknown Model")

    print(f"Carregou embeddings de {embeddings_data['num_documents']} documentos.")
    print(f"Total de segmentos: {embeddings_data['num_segment_contents']}")
    print(f"Modelo de embedding utilizado: {model_name}")

    retrieval_model = load_model(model_name)

    while True:
        user_query = input("\nDigite sua pergunta (ou 'exit' para sair): ")
        if user_query.lower() in ["exit", "sair", "quit"]:
            print("Encerrando chatbot...")
            break

        top_k = config["retrieve"]["top_k"]
        top_segments, top_similarities = search_query(
            user_query,
            corpus_embeddings,
            retrieval_model,
            segment_contents,
            top_k=top_k
        )

        print("\nTrechos recuperados:\n-------------------")
        for i, segment in enumerate(top_segments):
            print(f"Trecho {i+1} (similaridade={float(top_similarities[i]):.4f}): {segment}\n")

        context = "\n".join(top_segments)
        answer = generate_answer(user_query, context, config)

        print("Resposta do chatbot:\n--------------------")
        print(answer)
        
if __name__ == "__main__":
    main()
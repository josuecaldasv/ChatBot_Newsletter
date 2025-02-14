# Newsletter ChatBot

This repository hosts the project for a chatbot based on Petrobras' newsletter for its employees.

## Example Questions for the Chatbot (in Portuguese)

- **Quais são as vantagens principais das partículas de sílica RESIFA™ SOLESPHERE™ para a captura de CO₂?**
- **Quais foram os principais fatores que contribuíram para a queda nas vendas da Mercedes-Benz em 2024?**
- **Quais são os benefícios do uso de biocarvão pelas empresas Varaha e Charm Industrial?**

## Getting Started

Follow the steps below to set up and run the chatbot locally:

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Install Requirements

Install the necessary Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Create an `.env` File

- Create a `.env` file in the root directory and insert your OpenAI key with the name `OPENAI_API_KEY`.

### 4. Generate Embeddings

Run the following command to generate embeddings. Note that this process may take approximately 15 minutes:

```bash
python generate_embeddings.py
```

### 5. Run the Chatbot

Start the chatbot application by running:

```bash
python app.py
```

You will be able to interact with the chatbot via the terminal.
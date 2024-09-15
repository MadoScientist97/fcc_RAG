import os
import time
from dotenv import load_dotenv
import chromadb
from openai import OpenAI, embeddings
from chromadb.utils.embedding_functions import openai_embedding_function

# Loading env vars from .env
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")


def load_docs_from_dir(path) -> list[dict[str, str]]:
    print("***Loading document from dir***")
    documents = []
    for filename in os.listdir(path):
        if ".txt" in filename:
            with open(
                os.path.join(path, filename), 'r', encoding="utf-8"
            ) as file:
                documents.append({"id": filename, "text": file.read()})
    return documents


def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


def get_openai_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding
    print("**** Generating Embeddings ****")
    return embedding


def query_documents(question, n_results=2):
    results = collection.query(query_texts=question, n_results=n_results)
    relevant_chunks = [doc for sublist in results["documents"] for doc in sublist]
    print("**** Returning Relevant Chunks ****")
    return relevant_chunks


def generate_response(question, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    prompt = (
        "You are an assistant for question-answering tasks. Use the following "
        "pieces of retrieved context to answer the question. If you don't know"
        " the answer, say that you don't know. Use three sentences maximum and"
        " keep the answer concise."
        "\n\nContext:\n" + context + "\n\nQuestion:\n" + question
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": question,
            }
        ],
    )

    answer = response.choices[0].message
    return answer


if __name__ == "__main__":
    openai_ef = openai_embedding_function.OpenAIEmbeddingFunction(
        api_key=openai_key,
        model_name="text-embedding-3-small"
    )

    chroma_client = chromadb.PersistentClient(
        path="chroma_persistent_storageyy"
    )
    collection_name = "document_qa_collection"
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=openai_ef
    )

    client = OpenAI(api_key=openai_key)

#     path = "./news_articles"
#     documents = load_docs_from_dir(path)
#     print(f"Loaded {len(documents)} documents")
#     chunked_docs = []
#     for doc in documents:
#         chunks = split_text(doc["text"])
#         print("**** Splitting Documents into Chunks ****")
#         for i, chunk in enumerate(chunks):
#             chunked_docs.append(
#                 {
#                     "id": f"{doc["id"]}_chunk{i+1}",
#                     "text": chunk
#                 }
#             )

    # Generate embeddings for the document chunks
#     for doc in chunked_docs:
#         print("==== Generating embeddings... ====")
#         doc["embedding"] = get_openai_embedding(doc["text"])

#     for doc in chunked_docs:
#         print("**** Inserting chunks into DB ****")
#         print(doc)
#         collection.upsert(
#             ids=[doc["id"]],
#             documents=[doc["text"]],
#             embeddings=[doc["embedding"]]
#         )

    # Example query
    question = "tell me about databricks"
    relevant_chunks = query_documents(question)
    print(f"Relevant Chunks: \n{relevant_chunks}")
    answer = generate_response(question, relevant_chunks)

    print(answer)

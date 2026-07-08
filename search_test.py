from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

VECTOR_DB_FOLDER = "vector_db"

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vectorstore = Chroma(
    persist_directory=VECTOR_DB_FOLDER,
    embedding_function=embeddings
)

question = "Brief overview of the current detector concepts CLD IDEA ALLEGRO"

rraw_results = vectorstore.similarity_search(
    question,
    k=number_of_results * 8
)

reranker = load_reranker()

pairs = [(question, doc.page_content) for doc in raw_results]
scores = reranker.predict(pairs)

scored_results = list(zip(scores, raw_results))
scored_results.sort(key=lambda x: x[0], reverse=True)

results = [doc for score, doc in scored_results[:number_of_results]]

print("\nQUESTION:")
print(question)

print("\nTOP RESULTS:\n")

for i, doc in enumerate(results, start=1):
    source = doc.metadata.get("source", "Unknown source")
    page = doc.metadata.get("page", "Unknown page")

    print("=" * 80)
    print(f"RESULT {i}")
    print(f"Source: {source}")
    print(f"Page: {page}")
    print("-" * 80)
    print(doc.page_content[:1500])
    print()


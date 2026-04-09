!pip install -q langchain-community

# Start Ollama server
!ollama serve > /dev/null 2>&1 &

import pandas as pd
import numpy as np
from textblob import TextBlob
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.docstore.document import Document
from langchain.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import time

# === STEP 1: Load Excel and Preprocess ===
df = pd.read_excel("newsdata.xlsx")
df = df.rename(columns={'Title': 'headline', 'category_name': 'category'})

# Add sentiment column if missing
def get_sentiment(text):
    try:
        polarity = TextBlob(str(text)).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    except:
        return "neutral"

if 'sentiment' not in df.columns:
    df['sentiment'] = df['headline'].astype(str).apply(get_sentiment)

# Validate columns
required_columns = {"headline", "category", "sentiment"}
if not required_columns.issubset(set(df.columns)):
    raise ValueError(f"Excel must contain the columns: {required_columns}")

# === STEP 2: Create Vector Documents ===
documents = []
for _, row in df.iterrows():
    category = str(row['category']).strip()
    sentiment = str(row['sentiment']).strip()
    headline = str(row['headline']).strip()

    content = (
        f"Category: {category}\n"
        f"Sentiment: {sentiment}\n"
        f"Headline: {headline}\n"
        f"Tags: {category.lower()}, {sentiment.lower()}"
    )

    documents.append(Document(
        page_content=content,
        metadata={"category": category, "sentiment": sentiment}
    ))

# === STEP 3: Build FAISS Vector Store ===
print("🔍 Generating embeddings and building FAISS index...")
embeddings = OllamaEmbeddings(model="mxbai-embed-large")  # or try "nomic-embed-text"
db = FAISS.from_documents(documents, embeddings)

# Inspect Embeddings and Vector Store
print("\n📊 --- FAISS Vector DB Summary ---")
print("📁 Total documents stored:", len(db.docstore._dict))

# Show embedding vector shape and one vector
faiss_index = db.index
embeddings_array = faiss_index.reconstruct_n(0, faiss_index.ntotal)
print("🔢 Shape of embeddings array:", embeddings_array.shape)
print("🧬 First embedding vector (truncated):", embeddings_array[0][:10])

# Show 3 stored documents
print("\n📄 Sample Stored Documents:")
for i, (doc_id, doc) in enumerate(db.docstore._dict.items()):
    print(f"\n--- Document {i+1} ---")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    if i == 2:
        break

# Show similarity results from a sample query
query_text = "positive technology news"
query_embedding = embeddings.embed_query(query_text)
D, I = db.index.search(np.array([query_embedding]), k=3)

print("\n🔍 FAISS Similarity Search for:", query_text)
print("Closest vector indices:", I)
print("Distances:", D)

for idx in I[0]:
    doc = db.docstore._dict[db.index_to_docstore_id[idx]]
    print("\n➡️ Closest Match:")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)

# === STEP 4: Setup LLM and Conversational Chain ===
retriever = db.as_retriever(search_kwargs={"k": 100})
llm = Ollama(model="llama3")
qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)

# === STEP 5: Chat Loop ===
chat_history = []
print("\n🤖 NewsBot is ready. Type your query (or 'exit'):")

while True:
    query = input("You: ")
    if query.strip().lower() in ['exit', 'quit']:
        print("👋 Exiting NewsBot.")
        break

    result = qa_chain({
        "question": query,
        "chat_history": chat_history
    })

    print(f"\n🤖 Bot:\n{result['answer']}\n")
    chat_history.append((query, result["answer"]))

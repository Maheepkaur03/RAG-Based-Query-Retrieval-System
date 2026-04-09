# NewsBot using LLAMA-3 and RAG

A conversational AI assistant that queries a database of news headlines. It utilizes **Retrieval-Augmented Generation (RAG)** built with **LangChain**, **FAISS** vector store, and **Ollama (Llama-3)**.

## 🚀 Features
* **RAG-Powered Chat**: Ask questions about news datasets, and the bot fetches relevant answers from the custom database.
* **Sentiment Analysis**: Automatically evaluates the sentiment (Positive, Negative, Neutral) of news headlines using `TextBlob`.
* **Local Architecture**: Uses locally hosted LLMs (`llama3` and `mxbai-embed-large`) using Ollama, ensuring data privacy and offline capability.
* **Vector Storage**: Employs FAISS to create a fast similarity-search vector database.

## 🛠️ Prerequisites
* Python 3.8+
* [Ollama](https://ollama.com/) installed on your system.
* Required Python libraries:
  ```bash
  pip install pandas numpy textblob langchain faiss-cpu ollama
  ```

## 🧠 Models Needed
Ensure you pull the necessary models via Ollama before running the app:
```bash
ollama pull llama3
ollama pull mxbai-embed-large
```

## 📂 Dataset
The script expects an Excel file named `newsdata.xlsx` in the root directory.
It should contain the following columns:
* `Title` (will be mapped to `headline`)
* `category_name` (will be mapped to `category`)

## 🏃‍♂️ How to Run
1. Start the Ollama server locally (running in background):
   ```bash
   ollama serve
   ```
2. Run the Python script:
   ```bash
   python News_Task.py
   ```
3. Type your query when the `NewsBot is ready` message appears! Type `exit` or `quit` to stop.

## ⚠️ Notes regarding `Main.py`
The code contains Jupyter notebook syntaxes (like `!pip install...` and `!ollama serve...`). If you are running this natively as a Python script (`.py`), you should comment out those lines.

## 💬 Example Interaction

Here is an example of what the AI can do when querying the database!

```text
You: What are the top recurring keywords in negative sentiment news

🤖 Bot:
Based on the provided data, the top recurring keywords in headlines with a negative sentiment are:

1. "recall"
2. "red chilli powder"
3. "urges customers to return product"
4. "claim refund"
5. "negative"

These keywords appear frequently in headlines that have been classified as having a negative sentiment.
```

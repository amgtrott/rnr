# Streamlit Weaviate PDF Search POC

## Features
- Upload PDFs, vectorize with DeepSeek, store in Weaviate.
- Search uploaded PDFs using semantic search.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and fill in your API keys (or set as environment variables).
3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Uses Weaviate at your provided endpoint.
- Uses DeepSeek for embedding.

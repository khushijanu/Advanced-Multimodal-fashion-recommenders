# 👗 Advanced Multimodal Fashion Recommender
This project is a cutting-edge fashion search engine that leverages Multimodal Embeddings, Generative AI (LLMs), and Vector Databases to provide highly accurate, budget-aware product recommendations. By integrating visual features with semantic intent and price constraints, the system handles complex natural language queries like "Elegant teal Anarkali for a wedding under 4000" with sub-second precision.

Image Folder consists of Random 50-100 images, if one wants to refer the original dataset and Image folder, then please refer to this link: https://www.kaggle.com/datasets/djagatiya/myntra-fashion-product-dataset

## 🚀 System Architecture & Procedure
The recommendation engine follows a sophisticated four-stage pipeline to bridge the gap between human language and visual fashion attributes.

### 1. Data Engineering & Semantic Fusion

The foundation of the system is built on a dataset of over 14,000 fashion products.

Semantic Concatenation: To maximize search accuracy, unstructured attributes (brand, color, material, and style) were fused into a single unified semantic string (final_text).

Price Normalization: Since price is a critical search dimension, I implemented a Log-Transformation followed by Min-Max Scaling to ensure budget constraints are mathematically aligned with semantic vectors.

### 2. The 1025-Dimensional "Master Vector"

The core innovation is a custom Master Vector architecture that allows for simultaneous visual, textual, and economic search.

Vector Composition: [Visual Part (512)] + [Text Part (512)] + [Normalized Price (1)].

Weighted Retrieval: During the search, visual signals are weighted (1.5x) to prioritize style and aesthetics, while the text and price components ensure the results match the specific brand and budget intent.

### 3. Real-time Search Intent (LLM Reasoning)

To handle the "gap" between a user's short query and a machine's vector requirements, the system uses Gemini 2.0 Flash.

Visual Expansion: Gemini expands a simple query into a rich, CLIP-optimized description (e.g., "red dress" becomes "A vibrant crimson maxi dress in flowy chiffon, suitable for formal evening events").

Few-Shot Prompting: The system uses in-context learning to extract precise price filters and design attributes in structured JSON format.

### 4. Vector Storage & HNSW Retrieval

ChromaDB: All 14,000+ master vectors are stored in a persistent ChromaDB collection.

Semantic Search: The system uses Cosine Similarity to retrieve the top 6 matches in milliseconds, even on local hardware like a MacBook Air.

## 🛠️ Tech Stack

Multimodal AI: OpenAI CLIP (ViT-B/32)

Generative AI: Google Gemini 2.0 Flash (via google-genai SDK)

Vector DB: ChromaDB (Persistent Storage)

Data Processing: Pandas, NumPy, Scikit-learn (Min-Max Scaling, Log-Transforms)

Frontend: Streamlit (Responsive UI)

Inference: PyTorch (MPS/CUDA support)

## 📦 Project Structure
```
Plaintext
├── data_analysis.ipynb      # EDA, feature engineering, and price scaling logic
├── embeddings.ipynb          # CLIP vector generation & ChromaDB ingestion
├── app.py                    # Streamlit frontend & integrated search logic
├── price_scaler.pkl          # Persisted MinMaxScaler for search alignment
├── .env                      # API Credentials (ignored by git)
└── requirements.txt          # Project dependencies
```

## ⚙️ Setup & Installation
```
Clone & Environment:
Bash
git clone https://github.com/your-username/fashion-recommender.git
cd fashion-recommender
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
API Key: Create a .env file and add: GEMINI_API_KEY=your_google_ai_key.

Run Ingestion: Execute the code in embeddings.ipynb to generate the fashion_vector_db folder.
```
### Launch App:
python -m streamlit run app.py

## 💡Key Challenges Solved
Context Window Limits: Managed CLIP’s 77-token limit by implementing truncation and prompt constraints within the LLM expansion logic.

Attribute Sparsity: Solved the issue of missing descriptions by engineering a "Semantic Proxy" string from available product attributes.

Scale: Optimized the ingestion process into batches of 1000 to handle over 14,000 items efficiently.

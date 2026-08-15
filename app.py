import streamlit as st
import json
import os
import joblib
import torch
import clip
import numpy as np
import pandas as pd
import chromadb
from PIL import Image
from google import genai
from google.genai import types
from dotenv import load_dotenv

# --- 1. CONFIGURATION & INITIALIZATION ---
st.set_page_config(page_title="AI Fashion Recommender", layout="wide", page_icon="👗")
load_dotenv()

# Cache resource-heavy initializations
@st.cache_resource
def load_models_and_db():
    # Hardware check
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Load CLIP
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Load Scaler
    scaler = joblib.load('price_scaler.pkl')
    
    # Connect to ChromaDB
    chroma_client = chromadb.PersistentClient(path="./fashion_vector_db")
    collection = chroma_client.get_or_create_collection(name="multimodal_fashion")
    
    # Gemini Client
    api_key = os.getenv("GEMINI_API_KEY")
    gemini_client = genai.Client(api_key=api_key)
    
    return gemini_client, model, scaler, collection, device

gemini_client, model, scaler, collection, device = load_models_and_db()

# --- 2. CORE FUNCTIONS ---

def get_fashion_search_intent(user_query, default_price=2000):
    # IDENTITY AND RULES: Stays consistent
    sys_instruct = """
    You are an expert fashion stylist. Your task is to expand user queries into 
    highly descriptive visual prompts for CLIP image search. 
    Focus on: materials, patterns, specific design shapes (e.g., Anarkali, Straight), 
    and occasion-appropriate descriptors.
    
    RULES:
    1. Output ONLY valid JSON.
    2. Extract price as an integer. Use the default if not mentioned.
    3. Keep 'visual_expansion' under 50 words to avoid CLIP token limits.
    """

    # FEW-SHOT EXAMPLES: Teach by example
    user_prompt = f"""
    EXAMPLES:
    User: "I want something in teal"
    Output: {{"visual_expansion": "Teal solid Kurti with Palazzos and dupatta. Floral solid Anarkali shape, Empire style, Round neck.", "price": 2000}}
    
    User: "Green traditional wear under 3500"
    Output: {{"visual_expansion": "Green embroidered Kurta with Pyjamas and dupatta. Ethnic motifs, Straight shape, Mandarin collar.", "price": 3500}}
    
    NOW PROCESS THIS:
    User Query: "{user_query}"
    Default Price: {default_price}
    """

    config = types.GenerateContentConfig(
        temperature=0.2, # Low temperature for precision in data extraction
        system_instruction=sys_instruct, # New parameter for v2 SDK
        response_mime_type="application/json"
    )

    response = gemini_client.models.generate_content(
        model="gemini-2.0-flash",
        contents=user_prompt,
        config=config
    )
    return json.loads(response.text)
def execute_smart_search(user_query):
    intent = get_fashion_search_intent(user_query)
    expanded_text = intent['visual_expansion']
    raw_price = intent['price']

    # Normalize Price
    log_p = np.log1p(raw_price)
    price_df = pd.DataFrame([[log_p]], columns=['log_price'])
    norm_price = scaler.transform(price_df)[0][0]

    with torch.no_grad():
        text_tokens = clip.tokenize([expanded_text], truncate=True).to(device)
        text_emb = model.encode_text(text_tokens)
        text_emb /= text_emb.norm(dim=-1, keepdim=True)
        text_emb = text_emb.cpu().numpy().flatten()

    query_vector = np.concatenate([text_emb * 1.5, text_emb * 1.0, [norm_price]])
    
    # Query ChromaDB
    results = collection.query(
        query_embeddings=[query_vector.tolist()],
        n_results=6,
        include=['metadatas', 'distances']
    )
  # --- TERMINAL LOGGING ---
    print(f"\n🔍 Search Results for: {user_query}")
    print("-" * 85)
    print(f"{'Rank':<5} | {'Product ID':<12} | {'Score':<8} | {'Description'}")
    print("-" * 85)
    
    distances = results['distances'][0]
    ids = results['ids'][0]
    metadatas = results['metadatas'][0]
    
    for i, (p_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
        # Calculate Similarity Score (1 - Cosine Distance)
        score = 1 - dist 
        
        # Get description from metadata, fallback to 'No Description' if key missing
        # We use 'final_text' or 'product_name' depending on your dataset
        desc = meta.get('final_text', meta.get('product_name', 'No description available'))
        
        # Clean description for terminal display (remove newlines and truncate)
        clean_desc = desc.replace('\n', ' ')[:120] + "..."
        
        print(f"{i+1:<5} | {p_id:<12} | {score:.4f} | {clean_desc}")
        
    print("-" * 85 + "\n")
    
    return results, intent
# --- 3. STREAMLIT UI ---
st.title("👗 Adv Fashion Product Recommender")
st.markdown("Find your perfect style using AI-powered visual and semantic search.")

# Search interface
with st.container():
    query = st.text_input("Describe what you are looking for:", placeholder="e.g., Blue floral top for a beach wedding under 3000")
    search_button = st.button("Search")

if search_button and query:
    with st.spinner("AI is styling your recommendations..."):
        results, intent = execute_smart_search(query)
        
        st.info(f"**AI Interpretation:** {intent['visual_expansion']}")
        
        ids = results['ids'][0]
        metadatas = results['metadatas'][0]
        
        cols = st.columns(6)
        for i in range(len(ids)):
            with cols[i % 6]:
                p_id = ids[i]
                meta = metadatas[i]
                
                img_path = f"images/{p_id}.jpg"
                if os.path.exists(img_path):
                    # FIX: Changed use_container_width to width='stretch' for 2026 compatibility
                    st.image(img_path, width='stretch') 
                
                # ADDED: Product Name/Description from metadata
                # Note: 'final_text' or 'product_name' must be in your ChromaDB metadata
                st.write(f"**{meta.get('brand', 'Brand')}**")
                
                # Displaying a snippet of the description/name
                product_desc = meta.get('final_text', 'No description available')
                # Truncate for UI neatness
                st.write(f"*{product_desc[:80]}...*") 
                
                st.write(f"Price: **₹{meta['price']}**")
                st.divider()

elif search_button and not query:
    st.warning("Please enter a search query.")
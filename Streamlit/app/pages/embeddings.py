import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pages.ressources.components import Navbar, apply_custom_css, footer

st.set_page_config(page_title="Analyse des embeddings", page_icon="📊", layout="wide")

def calculate_song_embedding(audio_features):
    """
    Convert audio features into an embedding vector
    """
    # Placeholder for actual embedding generation
    return np.array(audio_features)

def calculate_similarity(embedding1, embedding2):
    """
    Calculate similarity between two song embeddings
    """
    return cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

def find_similar_songs(target_embedding, all_embeddings, n=5):
    """
    Find n most similar songs based on embeddings
    """
    similarities = [calculate_similarity(target_embedding, emb) for emb in all_embeddings]
    return sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:n]


def show_embeddings_page():
    st.title("Understanding Music Embeddings")
    
    st.header("What are Music Embeddings?")
    st.write("""
    Music embeddings are numerical representations of songs that capture their musical characteristics, 
    allowing us to measure similarity between different tracks. These vectors typically encode information about:
    - Rhythm and tempo
    - Melodic patterns
    - Harmonic content
    - Instrumental features
    - Genre characteristics
    """)

    st.header("How We Use Embeddings")
    st.write("""
    Our playlist generation system uses these embeddings to:
    1. Convert each song into a high-dimensional vector
    2. Calculate similarities between songs
    3. Group similar songs together
    4. Create coherent playlists based on musical similarity
    """)

    st.header("Technical Details")
    st.write("""
    We use advanced embedding techniques that combine:
    - Audio feature extraction
    - Deep learning models
    - Dimensionality reduction
    - Clustering algorithms
    """)

    # Add a placeholder for future visualizations
    st.header("Embedding Visualizations")
    st.info("Interactive visualizations of song embeddings will be added here")

def main():
    apply_custom_css()
    Navbar()
    st.markdown(
        """
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-family: helvetica, sans-serif; font-size: 2.5rem;'>
            <span style='background: linear-gradient(90deg, white, #ff69b4);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;
                         background-clip: text;
                         color: white;'>
                PLAYLIST
            </span>
            <span style='background: linear-gradient(90deg, #ff69b4, white);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;
                         background-clip: text;
                         color: white;'>
                CLUSTOM
            </span>
        </h1>
        <div style='display: flex; justify-content: center; gap: 10px; margin-top: -10px;'>
            <div style='height: 2px; width: 100px; background: linear-gradient(90deg, rgba(255,255,255,0), #00f2ff, rgba(255,255,255,0));'></div>
            <div style='height: 2px; width: 100px; background: linear-gradient(90deg, rgba(255,255,255,0), #ff69b4, rgba(255,255,255,0));'></div>
        </div>
        <p style='color: #00f2ff; font-family: helvetica; letter-spacing: 2px; margin-top: 5px;'>
            INTELLIGENT <span style='color: #ff69b4;'>PLAYLIST</span> GENERATION
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    show_embeddings_page()
    footer()

if __name__ == "__main__":
    main()

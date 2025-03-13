import streamlit as st
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pages.ressources.components import Navbar, apply_custom_css, footer

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

        footer()


    if __name__ == "__main__":
        main()

if __name__ == "__main__":
    show_embeddings_page()

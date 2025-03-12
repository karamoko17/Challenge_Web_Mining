import streamlit as st
from pages.ressources.components import Navbar , apply_custom_css, footer
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np
import io

st.set_page_config(page_title="Génération de playlist", page_icon="🎵", layout="wide")

def preprocess_data(df):
    numeric_features = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy', 'release_date']
    categorical_features = ['topic', 'genre']
    
    df_numeric = df[numeric_features].fillna(0)
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)
    
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(df[categorical_features])
    
    df_scaled = np.hstack((df_numeric_scaled, df_categorical))
    
    return df_scaled, scaler, encoder

def train_clustering_model(df_scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return kmeans, clusters

def recommend_similar_songs(song_name, artist_name, df, df_scaled, kmeans, scaler, n_recommendations=15):
    match = df[(df['track_name'] == song_name) & (df['artist_name'] == artist_name)]
    if match.empty:
        return "Chanson non trouvée dans la base de données."
    
    song_index = match.index[0]
    song_cluster = kmeans.labels_[song_index]
    
    similar_songs = df[kmeans.labels_ == song_cluster]
    
    song_features = df_scaled[song_index].reshape(1, -1)
    distances = np.linalg.norm(df_scaled[kmeans.labels_ == song_cluster] - song_features, axis=1)
    similar_songs['distance'] = distances
    
    similar_songs = similar_songs[(similar_songs['track_name'] != song_name) | (similar_songs['artist_name'] != artist_name)]
    
    recommendations = similar_songs.sort_values('distance').head(n_recommendations)[['artist_name', 'track_name']]
    
    return recommendations

def main():
    apply_custom_css()
    Navbar()

    st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1 style='font-family: helvetica, sans-serif; font-size: 2.5rem;'>
            <span style='background: linear-gradient(90deg, white, #00f2ff);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;
                         background-clip: text;
                         color: white;'>
                PLAYLIST
            </span>
            <span style='background: linear-gradient(90deg, #3ed60f, white);
                         -webkit-background-clip: text;
                         -webkit-text-fill-color: transparent;
                         background-clip: text;
                         color: white;'>
                CLUSTOM
            </span>
        </h1>
        <div style='display: flex; justify-content: center; gap: 10px; margin-top: -10px;'>
            <div style='height: 2px; width: 100px; background: linear-gradient(90deg, rgba(255,255,255,0), #00f2ff, rgba(255,255,255,0));'></div>
            <div style='height: 2px; width: 100px; background: linear-gradient(90deg, rgba(255,255,255,0), #3ed60f, rgba(255,255,255,0));'></div>
        </div>
        <p style='color: #00f2ff; font-family: helvetica; letter-spacing: 2px; margin-top: 5px;'>
            INTELLIGENT <span style='color: #3ed60f;'>PLAYLIST</span> GENERATION
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <style>
            .css-1yy6isu p {
                font-size: 24px !important;
                color: #00f2ff !important;
                font-weight: bold !important;
            }
        </style>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("**:green[Upload your MP3 file here]** 🎵", type=["mp3"])
    if uploaded_file is not None:
        df = pd.read_csv(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
        
        df_scaled, scaler, encoder = preprocess_data(df)
        kmeans, clusters = train_clustering_model(df_scaled, n_clusters=10)
        df['cluster'] = clusters
        
        song_name = st.text_input("Enter the song name:")
        artist_name = st.text_input("Enter the artist name:")
        
        if song_name and artist_name:
            recommendations = recommend_similar_songs(song_name, artist_name, df, df_scaled, kmeans, scaler)
            st.write("Recommended Playlist:")
            st.write(recommendations)

    footer()

if __name__ == "__main__":
    main()
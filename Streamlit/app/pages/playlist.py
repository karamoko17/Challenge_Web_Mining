import asyncio
import os
import tempfile

import numpy as np
import pandas as pd
from audio_recorder_streamlit import audio_recorder
from pages.ressources.components import Navbar, apply_custom_css, footer
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from SongRecognizer import SongRecognizer

import streamlit as st
import streamlit.components.v1 as components
import chromadb
from mistralai import Mistral

st.set_page_config(page_title="Génération de playlist", page_icon="🎵", layout="wide")

api_key = "RXjfbTO7wkOU0RwrwP7XpFfcj1K5eq40"
model = "mistral-embed"
client = Mistral(api_key=api_key)
# Créer un client ChromaDB
chroma_client = chromadb.PersistentClient(path="./Streamlit/app/data/chroma_db")

# Collection pour les embeddings musicaux
music_collection = chroma_client.get_or_create_collection(name="musiques")
collections = chroma_client.list_collections()


# # Mettre dans un df pandas le fichier csv des chansons
# csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "tcc_ceds_music.csv")
# df = pd.read_csv(csv_path)


# Fonction pour traiter la reconnaissance de manière asynchrone
async def process_recognition(file_path):
    recognizer = SongRecognizer()
    success = await recognizer.recognize_from_file(file_path)
    return success, recognizer.track_info if success else None


def run_async(coroutine):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    result = loop.run_until_complete(coroutine)
    loop.close()
    return result


def preprocess_data(df):
    numeric_features = [
        "danceability",
        "loudness",
        "acousticness",
        "instrumentalness",
        "valence",
        "energy",
        "release_date",
    ]
    categorical_features = ["topic", "genre"]

    df_numeric = df[numeric_features].fillna(0)
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    df_categorical = encoder.fit_transform(df[categorical_features])

    df_scaled = np.hstack((df_numeric_scaled, df_categorical))
    return df_scaled, scaler, encoder


def train_clustering_model(df_scaled, n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    return kmeans, clusters


def recommend_similar_songs(
    song_name, artist_name, df, df_scaled, kmeans, scaler, n_recommendations=15
):
    match = df[(df["track_name"] == song_name) & (df["artist_name"] == artist_name)]
    if match.empty:
        return "Chanson non trouvée dans la base de données."

    song_index = match.index[0]
    song_cluster = kmeans.labels_[song_index]
    similar_songs = df[kmeans.labels_ == song_cluster]

    song_features = df_scaled[song_index].reshape(1, -1)
    distances = np.linalg.norm(
        df_scaled[kmeans.labels_ == song_cluster] - song_features, axis=1
    )
    similar_songs["distance"] = distances

    similar_songs = similar_songs[
        (similar_songs["track_name"] != song_name)
        | (similar_songs["artist_name"] != artist_name)
    ]

    recommendations = similar_songs.sort_values("distance").head(n_recommendations)[
        ["artist_name", "track_name"]
    ]
    return recommendations

def get_playlist(song_lyrics, genre, playlist_size = 15):
    response = client.embeddings.create(
        model=model,
        inputs=[song_lyrics],
    ) 
    results_music = music_collection.query(
        query_embeddings=[response.data[0].embedding], 
        n_results=playlist_size,
        where={"genre" : genre}
    )
    playlist = results_music["metadatas"][0]
    
    # Ajout des URLs de preview audio pour chaque chanson
    for song in playlist:
        song['audio_preview_url'] = song.get('audio_preview_url', '')  # Utiliser l'URL existante si présente
        if not song['audio_preview_url'] and 'track_id' in song:
            song['audio_preview_url'] = f"https://p.scdn.co/mp3-preview/{song['track_id']}"
    
    return playlist

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
                SISEZAM
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

    st.markdown(
        """
        <style>
            .css-1yy6isu p {
                font-size: 24px !important;
                color: #00f2ff !important;
                font-weight: bold !important;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        uploaded_file = st.file_uploader(
            "**Upload your music file here** 🎵",
            type=["mp3", "wav", "flac", "m4a", "ogg"],
        )

    with col2:
        delai = 15

        st.write(f"**Or record a {delai}-second clip to identify it** 🎤")
        recorded_audio = audio_recorder(
            text="",
            recording_color="red",
            neutral_color="#ff69b4",
            icon_name="microphone",
            icon_size="5x",
            energy_threshold=(-1.0, 1.0),
            pause_threshold=delai,
        )
    temp_file_path = None

    new_upload = uploaded_file is not None and uploaded_file != st.session_state.get(
        "uploaded_file"
    )
    new_recording = (
        recorded_audio is not None
        and recorded_audio != st.session_state.get("recorded_audio")
    )

    if new_upload:
        st.session_state.uploaded_file = uploaded_file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name
    elif new_recording:
        st.session_state.recorded_audio = recorded_audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(recorded_audio)
            temp_file_path = tmp_file.name

    if temp_file_path:
        with st.spinner("Reconnaissance en cours..."):
            success, track_info = run_async(process_recognition(temp_file_path))
        os.unlink(temp_file_path)
    else:
        success, track_info = False, None

    if success:
        col1, col2 = st.columns([1, 2])
        with col1:
            if track_info["coverarthq"]:
                st.image(track_info["coverarthq"], caption="Pochette d'album")
            st.subheader("Informations")
            album_display = (
                f"[{track_info['album']}]({track_info['album_url']})"
                if "album_url" in track_info and track_info["album_url"]
                else track_info["album"]
            )

            info_md = f"""
            - **Titre**: {track_info["title"]}
            - **Artiste**: {track_info["artist"]}
            - **Album**: {album_display}
            - **Label**: {track_info["label"]}
            - **Date de sortie**: {track_info["releasedate"]}
            - **Genre**: {track_info["genre"]}
            """
            st.markdown(info_md)

        with col2:
            st.subheader("Extrait audio")
            st.audio(track_info["audio_preview_url"], format="audio/mpeg")
            st.subheader("Paroles")
            if track_info["lyrics"] and track_info["lyrics"] != "Paroles non trouvées":
                lyrics = track_info["lyrics"]
                if "Lyrics" in lyrics and "Embed" in lyrics:
                    lyrics = lyrics.split("Lyrics")[1].split("Embed")[0].strip()
                st.text_area("", lyrics, height=500)

                # Ajoute un bouton "Copier le contenu" qui copie les paroles dans le presse-papiers
                escaped_lyrics = lyrics.replace("`", "\\`").replace("'", "\\'")
                components.html(
                    f"""
                    <button 
                      onclick="navigator.clipboard.writeText(`{escaped_lyrics}`)" 
                      style="
                        background-color: #6aa36f;
                        border: none;
                        color: white;
                        padding: 10px 20px;
                        text-align: center;
                        text-decoration: none;
                        display: inline-block;
                        font-size: 16px;
                        margin: 4px 2px;
                        cursor: pointer;
                      ">
                      Copier les paroles
                    </button>
                    """,
                    height=50,
                )
            else:
                st.info("Paroles non disponibles pour cette chanson.")
        # 🎵 Génération et affichage de la playlist

        st.subheader("🎶 Playlist recommandée")
        playlist = get_playlist(track_info["lyrics"], track_info["genre"].lower(), playlist_size=10)

        if playlist:
            # Style pour les conteneurs de chansons
            st.markdown("""
                <style>
                    .song-item {
                        background: rgba(255, 105, 180, 0.1);
                        border-radius: 10px;
                        padding: 15px;
                        margin: 10px 0;
                        display: flex;
                        align-items: center;
                        gap: 20px;
                    }
                    .song-number {
                        color: #ff69b4;
                        font-size: 24px;
                        font-weight: bold;
                        min-width: 40px;
                    }
                    .song-info {
                        flex-grow: 1;
                    }
                    .song-title {
                        color: white;
                        font-size: 18px;
                        font-weight: bold;
                        margin: 0;
                    }
                    .song-artist {
                        color: #ff69b4;
                        font-size: 16px;
                        margin: 5px 0;
                    }
                    .song-genre {
                        color: #888888;
                        font-size: 14px;
                    }
                </style>
            """, unsafe_allow_html=True)

            # Affichage de chaque chanson
            for i, song in enumerate(playlist, 1):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"""
                        <div class="song-item">
                            <div class="song-number">#{i}</div>
                            <div class="song-info">
                                <div class="song-title">{song['track_name']}</div>
                                <div class="song-artist">{song['artist_name']}</div>
                                <div class="song-genre">{song['genre']}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    if 'audio_preview_url' in song and song['audio_preview_url'] and song['audio_preview_url'].strip():
                        st.audio(song['audio_preview_url'], format="audio/mpeg")

        else:
            st.warning("Aucune chanson trouvée pour cette playlist.")
    else:
        if new_upload or new_recording:
            st.error(
                "Impossible de reconnaître cette chanson. Veuillez essayer avec un autre fichier audio."
            )

    footer()


if __name__ == "__main__":
    main()

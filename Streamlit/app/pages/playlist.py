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

# api_key = "RXjfbTO7wkOU0RwrwP7XpFfcj1K5eq40"
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY") or st.secrets.get("MISTRAL_API_KEY", "")
model = "mistral-embed"
client = Mistral(api_key=MISTRAL_API_KEY)
# Créer un client ChromaDB
chroma_client = chromadb.PersistentClient(path="./Streamlit/app/data/chroma_db")

# Collection pour les embeddings musicaux
music_collection = chroma_client.get_or_create_collection(name="musiques")
collections = chroma_client.list_collections()


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
    print(results_music)
    # Convertir les listes en dictionnaires
    playlist = results_music["metadatas"][0]
    return playlist

def main():
    apply_custom_css()
    Navbar()

    st.markdown(
        """
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
            "**:green[Upload your music file here]** 🎵",
            type=["mp3", "wav", "flac", "m4a", "ogg"],
        )

    with col2:
        delai = 15

        st.write(f"**:green[Or record a {delai}-second clip to identify it]** 🎤")
        recorded_audio = audio_recorder(
            text="",
            recording_color="red",
            neutral_color="#6aa36f",
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

        # Récupérer la playlist à partir des paroles et du genre de la chanson actuelle
        playlist = get_playlist(track_info["lyrics"], track_info["genre"].lower(), playlist_size=10)
        print(track_info["lyrics"])
        print(track_info["genre"].lower())
        print(playlist)
        if playlist:
            # Création d'un tableau stylisé avec Streamlit
            st.markdown(
                """
                <style>
                    .playlist-table {
                        width: 100%;
                        border-collapse: collapse;
                        font-family: Arial, sans-serif;
                        background-color: #1db954;
                        color: white;
                        text-align: left;
                        border-radius: 10px;
                        overflow: hidden;
                    }
                    .playlist-table th, .playlist-table td {
                        padding: 10px;
                        border-bottom: 1px solid #ffffff55;
                    }
                    .playlist-table tr:hover {
                        background-color: #1ed760;
                    }
                    .playlist-table th {
                        background-color: #128c7e;
                    }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Affichage de la table
            st.markdown("<table class='playlist-table'>", unsafe_allow_html=True)
            st.markdown("<tr><th>#</th><th>Artiste</th><th>Track</th><th>Genre</th></tr>", unsafe_allow_html=True)

            for i, song in enumerate(playlist):
                st.markdown(
                    f"<tr><td>{i+1}</td><td>{song['artist_name']}</td><td>{song['track_name']}</td><td><audio controls><source src='{song['url_preview']}' type='audio/mpeg'></audio></td></tr>",
                    unsafe_allow_html=True,
                )

            st.markdown("</table>", unsafe_allow_html=True)
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

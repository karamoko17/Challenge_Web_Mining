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
import time
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
    # Initialisation des variables en session
    if "track_info" not in st.session_state:
        st.session_state.track_info = None  
    if "playlist_size" not in st.session_state:
        st.session_state.playlist_size = 15  # Valeur par défaut du slider
    if "playlist" not in st.session_state:
        st.session_state.playlist = []  # Stocke la playlist générée

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
        if success:
            st.session_state.track_info = track_info  
            st.session_state.playlist = []  # 🔥 Réinitialisation de la playlist
            st.session_state.playlist_size = 15  # 🔥 Réinitialisation du slider  # Forcer une mise à jour

    # else:
    #     success, st.session_state.track_info = False, None

    if st.session_state.track_info is not None:
        track_info = st.session_state.track_info
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
                        background-color: #ff69b4;
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

        # st.subheader("🎶 Playlist recommandée")
        # playlist = get_playlist(track_info["lyrics"], track_info["genre"].lower(), playlist_size=10)
        # Ajout du style pour le slider
        st.markdown("""
            <style>
                /* Style du texte du slider */
                .stSlider label {
                    font-size: 18px;
                    font-weight: bold;
                    color: #ff69b4;
                }

                /* Style de la barre du slider */
                .stSlider .css-1dp5vir {
                    background: linear-gradient(to right, #ff69b4, #ff1493);
                    border-radius: 10px;
                }

                /* Style du curseur du slider */
                .stSlider .css-1t5p9r6 {
                    background-color: #ff1493 !important;
                    box-shadow: 0px 0px 10px rgba(255, 20, 147, 0.8);
                }
            </style>
        """, unsafe_allow_html=True)
        # Titre de la section Playlist
        st.subheader("🎶 Playlist recommandée")

        # 🎚️ **Slider pour modifier dynamiquement la taille de la playlist**
        playlist_size = st.slider("📏 Nombre de chansons", min_value=5, max_value=50, value=st.session_state.playlist_size, step=1)

        # 🔥 **Récupération de la playlist UNIQUEMENT si `track_info` est disponible**
        if playlist_size != st.session_state.playlist_size or not st.session_state.playlist:
            st.session_state.playlist_size = playlist_size
            st.session_state.playlist = get_playlist(track_info["lyrics"], track_info["genre"].lower(), playlist_size=playlist_size)
            st.rerun()
        # 🎵 Génération de la playlist si nécessaire
        if not st.session_state.playlist:
            st.session_state.playlist = get_playlist(track_info["lyrics"], track_info["genre"].lower(), playlist_size=st.session_state.playlist_size)

          # 📌 **Affichage de la playlist**
        playlist = st.session_state.playlist
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
                        justify-content: space-between;
                    }
                    /* Style de l'image de couverture */
                    .song-cover {
                        width: 60px; /* Taille ajustée */
                        height: 60px;
                        border-radius: 10px; /* Bords arrondis */
                        box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2); /* Ombre légère */
                        object-fit: cover; /* Évite la déformation */
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
                    .song-title a {
                        color: white; /* Garde le texte blanc */
                        font-size: 18px;
                        font-weight: bold;
                        margin: 0;
                        text-decoration: none; /* Enlève le soulignement */
                        transition: all 0.3s ease-in-out;
                    }
                    /* Garde le texte blanc même après un clic */
                    .song-title a:link, 
                    .song-title a:visited, 
                    .song-title a:active {
                        color: white !important;
                    }
                    /* ✨ Effet de glow sur le titre au survol ✨ */
                    .song-title:hover {
                        text-shadow: 0 0 8px #ff69b4, 0 0 12px rgba(255, 105, 180, 0.8);
                        color: #ff69b4;
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
                     /* 🎵 Style personnalisé pour le lecteur audio 🎵 */
                    .audio-player {
                        width: 170px;
                        height: 32px;
                        border-radius: 15px;
                        background: #ff69b4;
                        outline: none;
                        border: 2px solid #ff69b4;
                        box-shadow: 0 0 5px #ff69b4, 0 0 8px rgba(255, 105, 180, 0.5);
                        transition: all 0.2s ease-in-out;
                    }

                    /* Effet subtil au survol */
                    .audio-player:hover {
                        box-shadow: 0 0 6px #ff69b4, 0 0 12px rgba(255, 105, 180, 0.4);
                        filter: brightness(1.1);
                    }


                    /* Supprimer le style natif du navigateur */
                    .audio-player::-webkit-media-controls-panel {
                        background-color: rgba(255, 105, 180, 0.2);
                        border-radius: 20px;
                    }
                    .audio-player::-webkit-media-controls-play-button {
                        filter: invert(1);
                    }
                    .audio-player::-webkit-media-controls-volume-slider {
                        filter: invert(1);
                    }
                </style>
            """, unsafe_allow_html=True)

            # Affichage de chaque chanson
            for i, song in enumerate(playlist, 1):
                # Vérification que l'URL audio est bien présente
                if 'coverarturl' in song and song['coverarturl']:
                    cover_html = f'<img class="song-cover" src="{song["coverarturl"]}" alt="Cover">'

                audio_html = ""
                if 'url_preview' in song and song['url_preview']:
                    audio_html = f"""
                        <audio class="audio-player" controls>
                            <source src="{song['url_preview']}" type="audio/mpeg">
                            Votre navigateur ne supporte pas l'élément audio.
                        </audio>
                    """
                # Affichage du morceau
                st.markdown(f"""
                    <div class="song-item">
                        {cover_html}
                        <div class="song-info">
                            <div class="song-title"><a href="{song['url_album']}">{song['track_name']}</a></div>
                            <div class="song-artist">{song['artist_name']}</div>
                            <div class="song-genre">{song['genre']}</div>
                        </div>
                        {audio_html}
                    </div>
                """, unsafe_allow_html=True)

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

import lyricsgenius
import spotipy
from shazamio import Shazam
from spotipy.oauth2 import SpotifyClientCredentials
import os
from dotenv import load_dotenv
import streamlit as st

# Configuration des clés API
# Load environment variables from .env file for local development
load_dotenv()

# Configuration des clés API from environment variables or GitHub Secrets
SPOTIFY_ID = os.environ.get("SPOTIFY_ID")
SPOTIFY_SECRET = os.environ.get("SPOTIFY_SECRET")
GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN")

# Verify if credentials are available
if not all([SPOTIFY_ID, SPOTIFY_SECRET, GENIUS_TOKEN]):
    # Try getting from Streamlit secrets if environment variables are not set
    SPOTIFY_ID = st.secrets.get("SPOTIFY_ID", SPOTIFY_ID)
    SPOTIFY_SECRET = st.secrets.get("SPOTIFY_SECRET", SPOTIFY_SECRET)
    GENIUS_TOKEN = st.secrets.get("GENIUS_TOKEN", GENIUS_TOKEN)

# Initialisation des clients API
spotify = spotipy.Spotify(
    client_credentials_manager=SpotifyClientCredentials(SPOTIFY_ID, SPOTIFY_SECRET)
)
genius = lyricsgenius.Genius(GENIUS_TOKEN)
shazam = Shazam()


class SongRecognizer:
    def __init__(self):
        self.track_info = {
            "title": "",
            "artist": "",
            "album": "",
            "label": "",
            "coverart": "",
            "releasedate": "",
            "genre": "",
            "lyrics": "",
            "popularity": "",
            "id_shazam": "",
            "id_spotify": "",
            "duration_sec": "",
        }

    async def recognize_from_file(self, file_path):
        """Reconnaît une chanson à partir d'un fichier audio en utilisant Shazam"""
        try:
            # Reconnaissance Shazam
            result = await shazam.recognize_song(file_path)

            if not result.get("track"):
                print("Aucune chanson reconnue")
                return False

            # Extraction des données Shazam
            self._extract_shazam_data(result)

            # Enrichissement avec Spotify
            self._enrich_with_spotify()

            # Récupération des paroles
            self._get_lyrics()

            return True

        except Exception as e:
            print(f"Erreur lors de la reconnaissance: {e}")
            return False

    def _extract_shazam_data(self, result):
        """Extrait les données pertinentes depuis le résultat Shazam"""
        track_data = result["track"]

        # Informations de base
        self.track_info["title"] = track_data.get("title", "Unknown Title")
        self.track_info["artist"] = track_data.get("subtitle", "Unknown Artist")
        self.track_info["coverart"] = track_data.get("images", {}).get("coverart", "")
        self.track_info["id_shazam"] = result.get("matches", [{}])[0].get(
            "id", "Unknown"
        )
        self.track_info["genre"] = track_data.get("genres", {}).get(
            "primary", "Unknown"
        )

        # Extraction des métadonnées
        metadata = track_data.get("sections", [{}])[0].get("metadata", [])
        for item in metadata:
            if item.get("title") == "Album":
                self.track_info["album"] = item.get("text", "Unknown Album")
            elif item.get("title") == "Label":
                self.track_info["label"] = item.get("text", "Unknown Label")
            elif item.get("title") == "Released":
                self.track_info["releasedate"] = item.get("text", "Unknown")

    def _enrich_with_spotify(self):
        """Enrichit les informations de la chanson avec les données Spotify"""
        try:
            # Recherche sur Spotify
            query = (
                f"track:{self.track_info['title']} artist:{self.track_info['artist']}"
            )
            if self.track_info["album"]:
                query += f" album:{self.track_info['album']}"

            results = spotify.search(q=query, type="track", limit=1)

            if not results["tracks"]["items"]:
                print("Chanson non trouvée sur Spotify")
                return

            track_result = results["tracks"]["items"][0]
            artist_info = spotify.artist(track_result["artists"][0]["id"])

            # Mise à jour des informations
            self.track_info["popularity"] = track_result.get("popularity", "N/A")
            self.track_info["id_spotify"] = track_result["id"]
            self.track_info["duration_sec"] = track_result["duration_ms"] / 1000

            # Compléter les genres si disponibles
            if artist_info.get("genres"):
                if self.track_info["genre"] == "Unknown":
                    self.track_info["genre"] = ", ".join(artist_info["genres"])
                else:
                    self.track_info["genre"] += f" + {', '.join(artist_info['genres'])}"

        except Exception as e:
            print(f"Erreur lors de l'enrichissement Spotify: {e}")

    def _get_lyrics(self):
        """Récupère les paroles de la chanson via Genius"""
        try:
            song = genius.search_song(
                self.track_info["title"], self.track_info["artist"]
            )

            if song:
                self.track_info["lyrics"] = song.lyrics
            else:
                self.track_info["lyrics"] = "Paroles non trouvées"

        except Exception as e:
            print(f"Erreur lors de la récupération des paroles: {e}")
            self.track_info["lyrics"] = "Erreur lors de la récupération des paroles"

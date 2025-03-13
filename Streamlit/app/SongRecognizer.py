import os

import lyricsgenius
from dotenv import load_dotenv
from shazamio import Shazam

import streamlit as st

# Configuration des clés API
# Load environment variables from .env file for local development
load_dotenv()

# Verify if credentials are available
# Use Streamlit secrets if environment variable is not set
GENIUS_TOKEN = os.environ.get("GENIUS_TOKEN") or st.secrets.get("GENIUS_TOKEN", "")

# Initialisation des clients API
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
            "id_shazam": "",
            "duration_sec": "",
        }

    async def recognize_from_file(self, file_path):
        """Reconnaît une chanson à partir d'un fichier audio en utilisant Shazam"""
        try:
            # Reconnaissance Shazam
            result = await shazam.recognize_song(file_path)
            print(result)

            if not result.get("track"):
                print("Aucune chanson reconnue")
                return False

            # Extraction des données Shazam
            self._extract_shazam_data(result)

            # Récupération des paroles
            self._get_lyrics()

            return True

        except Exception as e:
            print(f"Erreur lors de la reconnaissance: {e}")
            return False

    def _extract_shazam_data(self, result):
        """Extrait les données pertinentes depuis le résultat Shazam"""
        track_data = result["track"]
        print(track_data)

        # Informations de base
        self.track_info["title"] = track_data.get("title", "Unknown Title")
        self.track_info["artist"] = track_data.get("subtitle", "Unknown Artist")
        self.track_info["coverarthq"] = track_data.get("images", {}).get(
            "coverarthq", ""
        )
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

        # Extraction de l'url Apple permettant de jouer un extrait de la musique
        if "hub" in track_data:
            action = next(
                (
                    a
                    for a in track_data["hub"].get("actions", [])
                    if a.get("type") == "uri"
                ),
                None,
            )
            if action:
                self.track_info["audio_preview_url"] = action.get("uri", "")

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
            self.track_info["lyrics"] = (
                f"Erreur lors de la récupération des paroles: {e}"
            )

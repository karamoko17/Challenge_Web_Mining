from pages.ressources.components import Navbar, apply_custom_css, footer

import streamlit as st

st.set_page_config(page_title="SISEZAM", page_icon="💿", layout="wide")


def main():
    apply_custom_css()
    Navbar()

    # Header with enhanced styling
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
        /* Glitch overlay effect */
        body::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.15),
                rgba(0, 0, 0, 0.15) 1px,
                transparent 1px,
                transparent 2px
            );
            pointer-events: none;
            z-index: 9999;
            opacity: 0.3;
        }
        
        /* Random glitch animation */
        @keyframes glitch {
            0% { opacity: 1; }
            7% { opacity: 0.75; }
            10% { opacity: 1; }
            27% { opacity: 1; }
            30% { opacity: 0.75; }
            35% { opacity: 1; }
            52% { opacity: 1; }
            55% { opacity: 0.75; }
            60% { opacity: 1; }
            100% { opacity: 1; }
        }
        
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                90deg,
                rgba(255, 89, 0, 0.03),
                rgba(0, 242, 255, 0.03)
            );
            pointer-events: none;
            animation: glitch 30s infinite;
            z-index: 9998;
        }
    </style>
    """,
        unsafe_allow_html=True,
    )
    # Create a styled grid layout
    st.markdown(
        """
    <div style='background-color: #1f2430; padding: 20px; border-radius: 5px; border: 1px solid rgba(0, 242, 255, 0.3);'>
        <h3 style='color: #00f2ff; text-shadow: 0 0 10px rgba(0, 255, 198, 0.7), 0 0 20px rgba(0, 255, 198, 0.4);'>Welcome to Playlist Clustom. </h3>
        <p>This system provides personnalized playlists thanks to clustering and embeddings based on a song that you'll give us.</p>
        <p>Key Features:</p>
        <ul>
        <li>Use of Shazam and Spotify API</li>
        <li>MP3 file processing</li>
        <li>Clustering KMeans to select songs</li>
        <li>NLP Analysis with embeddings</li>
        </ul>
        <p>To get started, upload your song file or record a 5 seconds clip.</p>
    </div>
    """,
        unsafe_allow_html=True,
    )
    footer()


if __name__ == "__main__":
    main()

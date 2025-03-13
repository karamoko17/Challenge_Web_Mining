import streamlit as st

def footer():
    """Affiche le pied de page avec crédits et liens."""
    return st.markdown("""
    <div style='margin-top: 30px; text-align: center;'>
        <div style='height: 1px; background: linear-gradient(90deg, rgba(0,0,0,0), rgba(0, 242, 255, 0.5), rgba(0,0,0,0)); margin: 20px 0;'></div>
        <div style='font-family: monospace; color: rgba(255,255,255,0.5); font-size: 0.8rem;'>
            Projet Web Mining SISE © 2025 | Développé par 
            <a href="https://github.com/AntoineORUEZABALA" target="_blank" style="text-decoration: none; color: #ff69b4;">Antoine Oruezabala</a> |  
            <a href="https://github.com/berangerthomas" target="_blank" style="text-decoration: none; color: #ff69b4;">Béranger Thomas</a> |  
            <a href="https://github.com/karamoko17" target="_blank" style="text-decoration: none; color: #ff69b4;">Awa Karamoko</a> |  
            <a href="https://github.com/bertrandklein" target="_blank" style="text-decoration: none; color: #ff69b4;">Bertrand Klein</a><br>  
            <a href="https://github.com/karamoko17/Challenge_Web_Mining" target="_blank" style="text-decoration: none; color: #ff69b4;">Repo GitHub</a>
        </div>
    </div>
    """, unsafe_allow_html=True)


def apply_custom_css():
    """Applique le style CSS global de l'application."""
    st.markdown("""
    <style>
        /* Style général de l'application */
        .stApp {
            background-color: #000e69;
            color: #d8d9da;
        }

        /* Titres avec effet de glow */
        h1, h2 {
            color: #E9F8FD !important;
            font-family: helvetica, sans-serif;
            text-shadow: 0 0 10px rgba(255, 105, 180, 0.7), 0 0 20px rgba(255, 105, 180, 0.4);
        }

        /* Liens en rose */
        a {
            color: #ff69b4 !important;
            text-decoration: none;
        }
        a:hover {
            color: #ffb6c1 !important;
            text-shadow: 0 0 10px #ffb6c1;
            text-decoration: underline;
        }

        /* Boutons avec effet de survol */
        .stButton>button {
            background-color: #0b0f19;
            color: #ff69b4;
            border: 1px solid #ff69b4;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #ff69b4;
            color: #0b0f19;
            box-shadow: 0 0 15px rgba(255, 105, 180, 0.8);
        }

        /* Champs de saisie */
        .stTextInput>div>div>input, 
        .stNumberInput>div>div>input,
        .stSelectbox>div>div>select {
            background-color: #1f2430;
            color: #d8d9da;
            border: 1px solid #ff69b4 !important;
        }
    </style>
    """, unsafe_allow_html=True)


def Navbar():
    """Affiche la barre de navigation."""
    with st.sidebar:
        st.markdown("## Navigation")
        st.page_link('app.py', label='Accueil', icon='🏠')
        st.page_link('pages/playlist.py', label='Génération de Playlist', icon='🎵')
        st.page_link('pages/embeddings.py', label='Analyse des embeddings', icon='📊')
        st.markdown("---")



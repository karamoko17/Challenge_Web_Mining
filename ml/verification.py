import pandas as pd
import numpy as np


# Chargement des données
df = pd.read_csv(
    "D:\\M2 SISE\\Web Mining\\Challenge_Web_Mining\\Data\\tcc_ceds_music.csv",
    sep=";",
    encoding="ISO-8859-1",
    skipinitialspace=True,
    on_bad_lines="skip"
)

def song_exists(df, song_name, artist_name):
    exists = not df[(df['track_name'].str.lower() == song_name.lower()) & 
                    (df['artist_name'].str.lower() == artist_name.lower())].empty
    return exists

# Exemple d'utilisation
song_name = "mohabbat bhi jhoothi"
artist_name = "mukesh"

if song_exists(df, song_name, artist_name):
    print(f"La chanson '{song_name}' de '{artist_name}' existe dans le fichier.")
else:
    print(f"La chanson '{song_name}' de '{artist_name}' n'existe pas dans le fichier.")

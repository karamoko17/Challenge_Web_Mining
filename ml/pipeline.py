import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
import numpy as np

def preprocess_data(df):
    # Sélection des colonnes pertinentes pour le clustering
    numeric_features = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy', 'release_date']
    categorical_features = ['topic', 'genre']
    
    # Remplacer les virgules par des points dans les colonnes numériques
    for feature in numeric_features:
        df[feature] = df[feature].replace({',': '.'}, regex=True)  # Remplace les virgules par des points
        df[feature] = pd.to_numeric(df[feature], errors='coerce')  # Convertit les valeurs en numériques (NaN si problème)

    # Normalisation des données numériques
    df_numeric = df[numeric_features].fillna(0)
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)
    
    # Encodage des données catégorielles
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(df[categorical_features])
    
    # Concaténation des données transformées
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
    
    # Trouver l'index de la chanson donnée
    song_index = match.index[0]
    song_cluster = kmeans.labels_[song_index]
    
    # Sélectionner des chansons du même cluster
    similar_songs = df[kmeans.labels_ == song_cluster]
    
    # Trier par proximité en utilisant la distance euclidienne
    song_features = df_scaled[song_index].reshape(1, -1)
    distances = np.linalg.norm(df_scaled[kmeans.labels_ == song_cluster] - song_features, axis=1)
    similar_songs['distance'] = distances
    
    # Exclure la chanson sélectionnée
    similar_songs = similar_songs[(similar_songs['track_name'] != song_name) | (similar_songs['artist_name'] != artist_name)]
    
    recommendations = similar_songs.sort_values('distance').head(n_recommendations)[['artist_name', 'track_name']]
    
    # Affichage de la playlist
    print(f"Voici la playlist recommandée depuis {song_name} de {artist_name}:\n")
    for i, row in enumerate(recommendations.itertuples(), 1):
        print(f"{i}. {row.track_name} - {row.artist_name}")
    
    return recommendations

# Chargement des données
df = pd.read_csv(
    "C:\\Users\\akaramoko\\Desktop\\Challenge_Web_Mining\\Data\\tcc_ceds_music.csv",
    sep=";",
    encoding="ISO-8859-1",
    skipinitialspace=True,
    on_bad_lines="skip"
)

df_scaled, scaler, encoder = preprocess_data(df)
kmeans, clusters = train_clustering_model(df_scaled, n_clusters=10)
df['cluster'] = clusters

# Exemple d'utilisation
song_name = "the dogs of war"  
artist_name = "pink floyd"  
recommendations = recommend_similar_songs(song_name, artist_name, df, df_scaled, kmeans, scaler)

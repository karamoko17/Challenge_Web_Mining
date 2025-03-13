import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

def preprocess_data(df):
    # Sélection des colonnes pertinentes pour le clustering
    numeric_features = ['danceability', 'loudness', 'acousticness', 'instrumentalness', 'valence', 'energy', 'release_date']
    categorical_features = ['topic', 'genre']
    
    # Vérification et nettoyage des colonnes numériques
    for col in numeric_features:
        # Remplacer les virgules par des points et convertir en numérique
        df[col] = pd.to_numeric(df[col].replace({',': '.'}, regex=True), errors='coerce')
    
    # Remplir les valeurs manquantes par 0 (ou une autre stratégie de ton choix)
    df_numeric = df[numeric_features].fillna(0)
    
    # Normalisation des données numériques
    scaler = StandardScaler()
    df_numeric_scaled = scaler.fit_transform(df_numeric)
    
    # Encodage des données catégorielles
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(df[categorical_features])
    
    # Concaténation des données transformées
    df_scaled = np.hstack((df_numeric_scaled, df_categorical))
    
    return df_scaled, scaler, encoder


def train_clustering_model(df_scaled, n_clusters=10):
    # Initialisation du modèle de KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(df_scaled)
    
    # Évaluation du modèle de clustering avec silhouette score
    silhouette_avg = silhouette_score(df_scaled, clusters)
    print(f"Silhouette Score pour {n_clusters} clusters: {silhouette_avg:.4f}")
    
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

def visualize_clusters(df_scaled, kmeans, sample_size=1000):
    # Réduction de dimensionnalité pour la visualisation des clusters (PCA)
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df_scaled)

    # Échantillonnage des données pour l'affichage
    if len(reduced_data) > sample_size:
        indices = np.random.choice(len(reduced_data), sample_size, replace=False)
        reduced_data = reduced_data[indices]
        labels = kmeans.labels_[indices]
    else:
        labels = kmeans.labels_

    # Visualisation
    plt.figure(figsize=(10, 6))
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=labels, cmap='viridis', s=50, alpha=0.6)
    plt.title("Visualisation des clusters (échantillonnage)")
    plt.xlabel("Composant principal 1")
    plt.ylabel("Composant principal 2")
    plt.colorbar(label="Cluster ID")
    plt.show()
    
# Chargement des données
df = pd.read_csv(
    "D:\\M2 SISE\\Web Mining\\Challenge_Web_Mining\\Data\\tcc_ceds_music.csv",
    sep=";",
    encoding="ISO-8859-1",
    skipinitialspace=True,
    on_bad_lines="skip"
)

# Prétraitement des données
df_scaled, scaler, encoder = preprocess_data(df)

# Entraînement du modèle de clustering
n_clusters = 3
kmeans, clusters = train_clustering_model(df_scaled, n_clusters)
df['cluster'] = clusters

# Exemple d'utilisation
song_name = "the dogs of war"  
artist_name = "pink floyd"  
recommendations = recommend_similar_songs(song_name, artist_name, df, df_scaled, kmeans, scaler)


# Appel avec un échantillon de 1000 points
visualize_clusters(df_scaled, kmeans, sample_size=1000)


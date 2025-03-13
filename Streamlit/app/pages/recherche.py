import pandas as pd
import faiss
import numpy as np
import os
from sentence_transformers import SentenceTransformer



# Chargement des données
df = pd.read_csv(
    "D:\\M2 SISE\\Web Mining\\Challenge_Web_Mining\\Data\\tcc_ceds_music.csv",
    sep=";",
    encoding="ISO-8859-1",
    skipinitialspace=True,
    on_bad_lines="skip"
)


# Charger le modèle SBERT
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Fichier CSV contenant les paroles des chansons
EMBEDDINGS_FILE = "lyrics_embeddings.npy"
FAISS_INDEX_FILE = "faiss_index.bin"

# Charger les données
df = df.dropna(subset=["lyrics"])  # Supprimer les lignes avec paroles manquantes
lyrics_list = df["lyrics"].tolist()

# Vérifier si les embeddings sont déjà calculés
if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
    print("🔹 Chargement des embeddings et de l’index FAISS...")
    lyrics_embeddings = np.load(EMBEDDINGS_FILE)
    index = faiss.read_index(FAISS_INDEX_FILE)
else:
    print("⚡ Calcul des embeddings...")
    lyrics_embeddings = model.encode(lyrics_list, convert_to_numpy=True, batch_size=64)
    lyrics_embeddings = lyrics_embeddings / np.linalg.norm(lyrics_embeddings, axis=1, keepdims=True)  # Normalisation

    # Sauvegarde des embeddings
    np.save(EMBEDDINGS_FILE, lyrics_embeddings)

    # Création d’un index FAISS optimisé (approximation rapide)
    d = lyrics_embeddings.shape[1]  # Dimension des embeddings
    index = faiss.IndexHNSWFlat(d, 32)  # 32 voisins pour accélérer la recherche
    index.add(lyrics_embeddings)

    # Sauvegarde de l’index
    faiss.write_index(index, FAISS_INDEX_FILE)

print("✅ Moteur de recherche chargé avec succès !")

# Fonction de recherche optimisée
def search_song(query, top_k=5):
    query_embedding = model.encode([query], convert_to_numpy=True)
    query_embedding = query_embedding / np.linalg.norm(query_embedding)  # Normalisation

    distances, indices = index.search(query_embedding, top_k)
    
    results = df.iloc[indices[0]][["artist_name", "track_name", "lyrics"]].copy()
    results["similarity"] = distances[0]

    return results

# Exemple d'utilisation
query = "We don't need no education"
results = search_song(query, top_k=5)

print("\n🔎 Résultats de la recherche :\n")
for i, row in results.iterrows():
    print(f"{i+1}. {row['track_name']} - {row['artist_name']} (Similarité: {row['similarity']:.2f})")    
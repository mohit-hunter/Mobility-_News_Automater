import pandas as pd
import numpy as np

# 1. Load the news.csv file into a pandas DataFrame
news_df = pd.read_csv('/content/news.csv')

# 2. Create a new column named semantic_text
news_df['semantic_text'] = news_df['Executive Headline'] + ' ' + news_df['Summary']

from sentence_transformers import SentenceTransformer

# Load a pre-trained sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

print("SentenceTransformer model 'all-MiniLM-L6-v2' loaded successfully.")
# Generate embeddings for the 'semantic_text' column
print("Generating semantic embeddings...")
embeddings = model.encode(news_df['semantic_text'].tolist(), show_progress_bar=True)
print("Embeddings generated.")

# Save the embeddings to a NumPy array file
numpy_file_path = 'News_Semantic_Embedding_v1.npy'
np.save(numpy_file_path, embeddings)
print(f"Semantic embeddings saved to {numpy_file_path}")

# Verify the shape and data type of the saved embeddings
loaded_embeddings = np.load(numpy_file_path)
print(f"\nShape of loaded embeddings: {loaded_embeddings.shape}")
print(f"Data type of loaded embeddings: {loaded_embeddings.dtype}")

"""
Beta Configuration for Frost News Collector

Set BETA_MODE to False to disable all beta features and restore original behavior.
"""

# Beta mode toggle - set to False to disable all beta features
BETA_MODE = True

# Classification categories
CATEGORIES = ["automakers", "government", "suppliers"]

# Paths to classifier artifacts
CLASSIFIER_MODEL_PATH = "Classifier/baseline_model.pkl"
EMBEDDING_PATH = "Classifier/News_Semantic_Embedding_v1.npy"
EMBEDDING_INDEX_PATH = "Classifier/embedding_index.json"

# Beta data files
BETA_USERS_PATH = "beta_users.json"
BETA_CORRECTIONS_PATH = "beta_label_corrections.json"

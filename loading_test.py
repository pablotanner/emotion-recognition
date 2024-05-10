from src import DataLoader
from src.data_processing.data_split_loader import DataSplitLoader

#data_loader = DataLoader("./data", "./data", exclude=['deepface', 'facenet', 'landmarks', 'vggface'])

data_split_loader = DataSplitLoader("./data/annotations", "./data/features", "./data/embeddings", "./data/id_splits", excluded_features=['deepface', 'facenet', 'landmarks', 'vggface'])
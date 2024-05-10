from src import DataLoader
from src.data_processing.data_split_loader import DataSplitLoader
from datetime import datetime

start = datetime.now()
#data_loader = DataLoader("./data", "./data", exclude=['deepface', 'facenet', 'landmarks', 'vggface'])

data_split_loader = DataSplitLoader("./data/annotations", "./data/features", "./data/embeddings", "./data/id_splits", excluded_features=['deepface', 'facenet', 'landmarks', 'vggface'])

print("Time taken:", datetime.now() - start)
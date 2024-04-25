"""
Like fuser but for single embeddings, so I don't have to fill up feature fuser with bunch of model embeddings.

Also, since OpenFace isn't able to extract landmarks from all images, we have to separately load emotions for all images.
"""
import os

import numpy as np

class EmbeddingLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._ids = list(self.check_features().keys())
        self.emotions = np.array(self.load_annotations()).astype(int)

        self._vgg_face = []
        self._facenet = []
        self._facenet512 = []
        self._openface = []
        self._deepface = []
        self._deepid = []
        self._arcface = []
        self._sface = []

        self.embeddings = {
            "VGG-Face": self._vgg_face,
            "Facenet": self._facenet,
            "Facenet512": self._facenet512,
            "OpenFace": self._openface,
            "DeepFace": self._deepface,
            "DeepID": self._deepid,
            "ArcFace": self._arcface,
            "SFace": self._sface
        }

        self.load_embeddings()

    def load_annotations(self):
        emotions = []
        for sample in self._ids:
            emotions.append(np.load(f"{self.data_dir}/annotations/{sample}_exp.npy"))
        return emotions

    def check_features(self):
        # Make sure that for each image, we have all feature types
        all_ids = {}
        for file in os.listdir(self.data_dir + "/embeddings"):
            if file.endswith(".npy"):
                file_id = file.split("_")[0]
                if file_id not in all_ids:
                    all_ids[file_id] = 0
                all_ids[file_id] += 1

        for file_id in all_ids:
            if all_ids[file_id] != list(all_ids.values())[0]:
                raise ValueError(f"Missing Embeddings for {file_id}")
        return all_ids

    def load_embeddings(self):
        # First create list of all file ids, for which we have all feature types
        for file_id in self._ids:
            for model in self.embeddings:
                embedding = np.load(f"{self.data_dir}/embeddings/{file_id}_{model}.npy")
                self.embeddings[model].append(embedding)
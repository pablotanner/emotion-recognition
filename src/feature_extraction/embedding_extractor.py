from deepface import DeepFace
import glob
import numpy as np
import multiprocessing
from functools import partial

face_images = glob.glob("../../data/images/*.jpg")

models = [
  "VGG-Face", # 4096
  "Facenet", # 128
  "Facenet512", # 512
  "OpenFace", # 129
  "DeepFace", # 4096
  "DeepID", # 160
  "ArcFace", # 512
  #"Dlib",
  "SFace", # 128
  #"GhostFaceNet",
]


def extract_embeddings(image_path, model="DeepFace"):
  # Extract the embedding, also force indexing 0 because there should only be 1 face in the image
  # detector_backend skips face detection because we already know there is a face in the image
    return DeepFace.represent(detector_backend="skip", model_name=model, img_path=image_path)[0]["embedding"]

def compare_models(image_path, models):
    result = {}
    for model in models:
      try:
        embedding = DeepFace.represent(detector_backend="skip", img_path=image_path, model_name=model)[0]["embedding"]
        result[model] = embedding
      except:
        result[model] = "Model not supported."
    return result

#embedding_by_model = compare_models(images[0], models)

#embedding1 = extract_embeddings(images[23])
#embedding2 = extract_embeddings(images[23])

def extract_and_save_embeddings(images, model="DeepFace"):
    for image in images:
        number = image.split("\\")[-1].split(".")[0]
        extracted = extract_embeddings(image, model)
        np.save(f"../../data/embeddings/{number}_{model}.npy", extracted)


def run_model(model, images):
  print(f"Processing with model {model}")
  extract_and_save_embeddings(images, model)

def parallel_process_models(models, images):
  # Use all available cores, or specify the number of processes you want to use
  with multiprocessing.Pool(processes=len(models)) as pool:
    pool.starmap(run_model, [(model, images) for model in models])

if __name__ == "__main__":
    parallel_process_models(models, face_images)
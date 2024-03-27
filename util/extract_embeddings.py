from deepface import DeepFace
import glob

images = glob.glob("../data/images/*.jpg")

models = [
  "VGG-Face",
  "Facenet",
  "Facenet512",
  "OpenFace",
  "DeepFace",
  "DeepID",
  "ArcFace",
  "Dlib",
  "SFace",
  "GhostFaceNet",
]


def extract_embeddings(image_path):
  # Extract the embedding, also force indexing 0 because there should only be 1 face in the image
  # detector_backend skips face detection because we already know there is a face in the image
    return DeepFace.represent(detector_backend="skip", img_path=image_path)[0]["embedding"]

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

embedding1 = extract_embeddings(images[23])
embedding2 = extract_embeddings(images[23])

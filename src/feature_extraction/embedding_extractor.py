from deepface import DeepFace
import glob
import numpy as np
import multiprocessing
import argparse

parser = argparse.ArgumentParser(description='Extract embeddings from images using different models.')
parser.add_argument('--image_path', type=str, default="../../data/images/*.jpg", help='Path to images to extract embeddings from.')
parser.add_argument('--output_path', type=str, default="../../data/embeddings/", help='Path to save the embeddings.')
args = parser.parse_args()



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

def extract_and_save_embeddings(images, output_path, model="DeepFace"):
    for image in images:
        number = image.split("/")[-1].split(".")[0]
        extracted = extract_embeddings(image, model)
        np.save(f"{output_path}/{number}_{model}.npy", extracted)


def run_model(model, images):
  print(f"Processing with model {model}")
  extract_and_save_embeddings(images, args.output_path, model)

def parallel_process_models(models, images):
  # Use all available cores, or specify the number of processes you want to use
  with multiprocessing.Pool(processes=len(models)) as pool:
    pool.starmap(run_model, [(model, images) for model in models])


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



if __name__ == "__main__":
    #useful_models = ['SFace', 'Facenet', 'ArcFace'] # These are the models that don't have terrible accuracy for emo rec
    useful_models = ['ArcFace', 'VGG-Face']
    # Get all images from the given path
    face_images = glob.glob(args.image_path + '/*.jpg')

    print(f"Extracting embeddings from {len(face_images)} images.")
    # Extract embeddings using the given model
    parallel_process_models(useful_models, face_images)
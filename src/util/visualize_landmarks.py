import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image Index (0 - 4xxx)
INDEX = 3

def show_landmarks(index, use_affect_net=True):
    base_path = "../../data"
    lnd_path = f"{base_path}/features/{index}_landmarks.npy"
    image_path = f"{base_path}/images/{index}.jpg"

    if use_affect_net:
        lnd_path = f"{base_path}/annotations/{index}_lnd.npy"

    landmarks = np.load(lnd_path).reshape(-1, 2)
    image_np = np.array(Image.open(image_path))

    plt.imshow(image_np)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='red')
    plt.show()


# Show landmarks (OpenFace vs AffectNet)
#show_landmarks(INDEX, use_affect_net=False)
show_landmarks(INDEX, use_affect_net=True)

#affect_net = np.load(f"../../data/annotations/{INDEX}_lnd.npy")
#open_face = np.load(f"../../data/features/{INDEX}_landmarks.npy")

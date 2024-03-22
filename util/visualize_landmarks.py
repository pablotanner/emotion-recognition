import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def show_landmarks(index):
    base_path = "../data"
    lnd_path = f"{base_path}/annotations/{index}_lnd.npy"
    image_path = f"{base_path}/images/{index}.jpg"

    landmarks = np.load(lnd_path).reshape(-1, 2)
    image_np = np.array(Image.open(image_path))

    plt.imshow(image_np)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, c='red')
    plt.show()


show_landmarks(15)

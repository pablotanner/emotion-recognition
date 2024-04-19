from tkinter import Image

import numpy as np
from matplotlib import pyplot as plt

def rotation_matrix(rx, ry, rz):
    """Create a 3x3 rotation matrix from rotation angles."""
    # Invert the rotation angles (So the direction is correct)
    rx, ry, rz = -rx, -ry, -rz
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])

    # Order of multiplication: first rotate around x, then y, then z
    return Rz @ Ry @ Rx

def apply_transformation(landmarks, rx, ry, rz, tx, ty, tz):
    # Calculate pre-rotation center
    pre_center = np.mean(landmarks, axis=0)

    # Apply rotation
    R = rotation_matrix(rx, ry, rz)
    rotated_landmarks = np.dot(landmarks, R.T)

    # Calculate post-rotation center and adjust translation
    post_center = np.mean(rotated_landmarks, axis=0)
    translation_adjustment = pre_center - post_center

    # Invert translation
    tx, ty, tz = -tx, -ty, -tz

    final_translation = np.array([tx, ty, tz]) + translation_adjustment

    # Apply final translation
    transformed_landmarks = rotated_landmarks + final_translation

    return transformed_landmarks

def reshape_3d_landmarks(landmarks):
    """
    Transforms the extracted landmark data from [x0,x1,...y0,y1,...z0,z1,..] to [[x0,y0,z0],[x1,y1,z1],...]
    """
    X = landmarks[0:68]
    Y = landmarks[68:136]
    Z = landmarks[136:]
    return np.vstack((X, Y, Z)).T


def visualize_landmark_difference(landmarks1, landmarks2):
    x1 = landmarks1[:, 0]
    y1 = landmarks1[:, 1]
    z1 = landmarks1[:, 2]

    x2 = landmarks2[:, 0]
    y2 = landmarks2[:, 1]
    z2 = landmarks2[:, 2]

    # Create a new matplotlib figure and its axes instance (with 3D capabilities)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x1, y1, z1, color='blue')
    ax.scatter(x2, y2, z2, color='red')

    # Setting labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Title
    ax.set_title('3D Plot of Transformed Facial Landmarks')

    # Make scale equal
    ax.set_aspect('equal')


    # For both landmarks, make the points at x16, y16, z16 and x21, y21, z21 red
    ax.scatter(x1[16], y1[16], z1[16], color='black')
    ax.scatter(x2[16], y2[16], z2[16], color='black')

    # Show the plot
    plt.show()


def get_3d_landmarks(index):
    """
    Loads the 3D landmarks for a given index.
    """
    return np.load(f"../../data/features/{index}_landmarks_3d.npy")

def get_pose(index):
    """
    Returns [pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz] for a given index.
    """
    return np.load(f"../../data/features/{index}_pose.npy")



def calculate_landmark_difference(landmarks1, landmarks2):
    """
    Calculate the difference between two sets of landmarks. (Performance metric, smaller is better)
    """
    return np.linalg.norm(landmarks1 - landmarks2)


def compare_landmarks(index1, index2, use_standardization=False):
    """
    Compare two sets of landmarks. If use_standardization is True, the landmarks will be standardized (rotated, translated)
    before comparison.
    """
    landmarks1 = reshape_3d_landmarks(get_3d_landmarks(index1))
    pose1 = get_pose(index1)

    landmarks2 = reshape_3d_landmarks(get_3d_landmarks(index2))
    pose2 = get_pose(index2)

    if use_standardization:
        landmarks1 = apply_transformation(landmarks1, pose1[3], pose1[4], pose1[5], pose1[0], pose1[1], pose1[2])
        landmarks2 = apply_transformation(landmarks2, pose2[3], pose2[4], pose2[5], pose2[0], pose2[1], pose2[2])

    print(calculate_landmark_difference(landmarks1, landmarks2))
    visualize_landmark_difference(landmarks1, landmarks2)


def standardize_landmarks(landmarks, pose):
    """
    Standardize the landmarks by applying the pose transformation. First reshapes, then standardizes, then reshapes back.
    """
    reshaped_landmarks = reshape_3d_landmarks(landmarks)
    standardized_landmarks = apply_transformation(reshaped_landmarks, pose[3], pose[4], pose[5], pose[0], pose[1], pose[2])
    x, y, z = [], [], []
    for [x_coordinate, y_coordinate, z_coordinate] in standardized_landmarks:
        x.append(x_coordinate)
        y.append(y_coordinate)
        z.append(z_coordinate)
    return np.asarray(x + y + z)



# 36 For Surprised and Front Facing
# 45 For unaligned
# 395 Looks to right
# 467 Looks to left

FIRST = "36"
SECOND = "395"


#compare_landmarks(FIRST, SECOND, use_standardization=True)
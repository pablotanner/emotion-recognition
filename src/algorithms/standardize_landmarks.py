from tkinter import Image

import numpy as np
from matplotlib import pyplot as plt

def transformation_matrix(rx, ry, rz, tx, ty, tz):
    """Create a 4x4 transformation matrix from rotation and translation."""
    # Invert the rotation angles (So the direction is correct)
    rx, ry, rz = -rx, -ry, -rz


    Rx = np.array([[1, 0, 0, 0],
                   [0, np.cos(rx), -np.sin(rx), 0],
                   [0, np.sin(rx), np.cos(rx), 0],
                   [0, 0, 0, 1]])
    Ry = np.array([[np.cos(ry), 0, np.sin(ry), 0],
                   [0, 1, 0, 0],
                   [-np.sin(ry), 0, np.cos(ry), 0],
                   [0, 0, 0, 1]])
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0, 0],
                   [np.sin(rz), np.cos(rz), 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    T = np.array([[1, 0, 0, tx],
                  [0, 1, 0, ty],
                  [0, 0, 1, tz],
                  [0, 0, 0, 1]])

    # Order of multiplication: first apply rotation, then translation
    return T @ Rx @ Ry @ Rz


def reshape_3d_landmarks(landmarks):
    """
    Transforms the extracted landmark data from [x0,x1,...y0,y1,...z0,z1,..] to [[x0,y0,z0],[x1,y1,z1],...]
    """
    X = landmarks[0:68]
    Y = landmarks[68:136]
    Z = landmarks[136:]
    return np.vstack((X, Y, Z)).T


def visualize_3d_landmarks(landmarks):
    x = landmarks[:, 0]
    y = landmarks[:, 1]
    z = landmarks[:, 2]

    # Create a new matplotlib figure and its axes instance (with 3D capabilities)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(x, y, z)

    # Setting labels
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Title
    ax.set_title('3D Plot of Transformed Facial Landmarks')

    # Show the plot
    plt.show()

def visualize_comparison(landmarks1, landmarks2):
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


    #Above doesnt work






    # Show the plot
    plt.show()


def get_3d_landmarks(index):
    return np.load(f"../../data/features/{index}_landmarks_3d.npy")

def get_pose(index):
    """
    Returns [pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz]
    """
    return np.load(f"../../data/features/{index}_pose.npy")



def standardize_3d_landmarks(landmarks, pose_data):
    pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz = pose_data
    transform_mat = transformation_matrix(pose_Rx, pose_Ry, pose_Rz, pose_Tx, pose_Ty, pose_Tz)

    # Convert to homogeneous coordinates by adding a column of ones
    landmarks_homogeneous = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])

    return np.dot(landmarks_homogeneous, transform_mat.T)


def calculate_landmark_difference(landmarks1, landmarks2):
    """
    Calculate the difference between two sets of landmarks.
    """
    return np.linalg.norm(landmarks1 - landmarks2)

# 36 For Surprised and Front Facing
# 45 For unaligned
# 395 Looks to right
# 467 Looks to left

FIRST = "395"
SECOND = "467"

# Normal Landmarks
reshaped_landmarks1 = reshape_3d_landmarks(get_3d_landmarks(FIRST))
pose1 = get_pose(FIRST)

reshaped_landmarks2 = reshape_3d_landmarks(get_3d_landmarks(SECOND))
pose2 = get_pose(SECOND)

# Rotated and Translated Landmarks
standardized_landmarks1 = standardize_3d_landmarks(reshaped_landmarks1, pose1)
standardized_landmarks2 = standardize_3d_landmarks(reshaped_landmarks2, pose2)

"""

rot1 = rotation_matrix(pose1[3], pose1[4], pose1[5])
rot2 = rotation_matrix(pose2[3], pose2[4], pose2[5])

# Rotated landmarks
rotated_landmarks1 = reshaped_landmarks1 @ rot1.T
rotated_landmarks2 = reshaped_landmarks2 @ rot2.T

"""

visualize_comparison(standardized_landmarks1, standardized_landmarks2)

#visualize_comparison(reshaped_landmarks1, reshaped_landmarks2)
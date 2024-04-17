from tkinter import Image

import numpy as np
from matplotlib import pyplot as plt


def calculate_rotation_angle(landmarks, index1, index2, axis='z'):
    if axis == 'z':
        # Horizontal alignment for roll
        point1 = landmarks[index1]
        point2 = landmarks[index2]
        delta_y = point2[1] - point1[1]
        delta_x = point2[0] - point1[0]
        angle = np.arctan2(delta_y, delta_x)
        return -angle  # Negative to correct the tilt
    elif axis == 'y':
        # Vertical alignment for pitch
        point1 = landmarks[index1]
        point2 = landmarks[index2]
        delta_z = point2[2] - point1[2]
        delta_y = point2[1] - point1[1]
        angle = np.arctan2(delta_z, delta_y)
        return -angle
    elif axis == 'x':
        # Depth alignment for yaw
        point1 = landmarks[index1]
        point2 = landmarks[index2]
        delta_z = point2[2] - point1[2]
        delta_x = point2[0] - point1[0]
        angle = np.arctan2(delta_z, delta_x)
        return -angle


def rotate_about_center(landmarks, angle, axis='z'):
    # Assuming landmarks are centered about the origin for simplicity
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    if axis == 'z':
        rotation_matrix = np.array([
            [cos_angle, -sin_angle, 0],
            [sin_angle, cos_angle, 0],
            [0, 0, 1]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_angle, 0, sin_angle],
            [0, 1, 0],
            [-sin_angle, 0, cos_angle]
        ])
    elif axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_angle, -sin_angle],
            [0, sin_angle, cos_angle]
        ])


    return np.dot(landmarks, rotation_matrix.T)


def make_front_facing(landmarks, eye_left_idx, eye_right_idx):
    # Step 1: Align horizontally (roll)
    roll_angle = calculate_rotation_angle(landmarks, eye_left_idx, eye_right_idx, 'z')
    landmarks = rotate_about_center(landmarks, roll_angle, 'z')

    # Step 2: Align vertically (pitch)
    pitch_angle = calculate_rotation_angle(landmarks, eye_left_idx, eye_right_idx, 'y')
    landmarks = rotate_about_center(landmarks, pitch_angle, 'y')

    # Step 3: Align depth (yaw)
    yaw_angle = calculate_rotation_angle(landmarks, eye_left_idx, eye_right_idx, 'x')
    landmarks = rotate_about_center(landmarks, yaw_angle, 'x')


    return landmarks

def reshape_3d_landmarks(landmarks):
    X = landmarks[0:68]
    Y = landmarks[68:136]
    Z = landmarks[136:]
    return np.vstack((X, Y, Z)).T


def visualize(landmarks):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the actual data without negating y-values
    ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], c='r', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    # Optionally set initial viewing angle if desired
    #ax.view_init(elev=70, azim=95,roll=0)
    ax.view_init(elev=0, azim=5, roll=5)

    # Enable grid for better spatial understanding
    ax.grid(True)
    plt.title("3D View of Facial Landmarks")

    # Show plot with interactive rotation enabled
    plt.show()


def format_coordinates(landmarks):
    return

# 36 For Surprised and Front Facing
# 45 For unaligned
INDEX = 36
lnd = reshape_3d_landmarks(np.load(f"../../data/features/{INDEX}_landmarks_3d.npy"))

aligned_landmarks = make_front_facing(lnd, 0, 16)

visualize(aligned_landmarks)

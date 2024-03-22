import csv
import os
import numpy as np


"""
Extracts the landmarks and Facial Action Coding System (FACS) data from the OpenFace csv files
"""

"""
***INDICES***

Landmarks 2D (in pixels): x_0 (296) to x_67 (363) and y_0 (364) to y_67 (431)
Landmarks 3D (in mm): X_0 (432) to X_67 (499) and Y_0 (500) to Y_67 (567) and Z_0 (568) to Z_67 (635)

FACS Intensity: AU01_r (676) to AU45_r (692)
FACS Presense: AU01_c (693) to AU45_c (710)
"""


def format_landmarks(landmarks):
    # Calculate the midpoint of the array
    midpoint = len(landmarks) // 2

    # Split the array into two halves
    first_half = landmarks[:midpoint]
    second_half = landmarks[midpoint:]

    # Create a new array by alternating elements from the two halves
    new_array = [None] * (len(landmarks))
    new_array[::2] = first_half
    new_array[1::2] = second_half

    return new_array

# extract landmark and Facial Unit data from the open face csv files
def extract_open_face_data(data_path):
    csv_files = []
    for file in os.listdir(data_path):
        if file.endswith(".csv"):
            csv_files.append(file)

    # for file in csv_files, save landmark and facial unit data to dictionary (skip first row in csv)
    for file in csv_files:
        file_path = f"{data_path}/{file}"
        index = file.split(".")[0]
        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    # get the parsed float landmarks (format_landmarks() is used to "fix" coordinate order of the landmarks)
                    temp_landmarks = row[296:432]
                    landmarks = np.array(format_landmarks(temp_landmarks)).astype(float)

                    # get the facial unit intensity
                    facs_intensity = np.array(row[676:693]).astype(float)

                    # get the facial unit presence (chaining necessary to convert to int)
                    facs_presence = np.array(row[693:711]).astype(float).astype(int)

                    # save the data as .npy file
                    np.save(f"../data/features/{index}_landmarks", landmarks)
                    np.save(f"../data/features/{index}_facs_intensity", facs_intensity)
                    np.save(f"../data/features/{index}_facs_presence", facs_presence)

                    line_count += 1


path_to_data = "C:/Users/41763/Desktop/OpenFace_2.2.0_win_x64/processed"

#extract_open_face_data(path_to_data)

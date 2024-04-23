"""
Extracts features from the OpenFace .csv files into INDEX_FEATURE.npy files
"""
import argparse
import csv
import os
import numpy as np

parser = argparse.ArgumentParser(description='Extract OpenFace data from CSV files.')
parser.add_argument('-input_folder', type=str, help='Path to the input folder containing CSV files.')
parser.add_argument('-output_folder', type=str, help='Path to the output folder where extracted data will be stored as .npy')
args = parser.parse_args()


def extract_open_face_data(folder_path, output_folder_path):
    csv_files = []
    # Get all the csv files in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_files.append(file)

    print(f"Extracting features from {len(csv_files)} files")

    for file in csv_files:
        file_path = f"{folder_path}/{file}"
        index = file.split(".")[0]
        if int(index) % 1000 == 0:
            print(f"Extracting features from file {index}")
        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    line_count += 1
                else:
                    landmarks = np.array(row[296:432]).astype(float)
                    np.save(f"{output_folder_path}/{index}_landmarks", landmarks)

                    landmarks_3d = np.array(row[432:636]).astype(float)
                    np.save(f"{output_folder_path}/{index}_landmarks_3d", landmarks_3d)

                    # get the facial unit intensity
                    facs_intensity = np.array(row[676:693]).astype(float)
                    np.save(f"{output_folder_path}/{index}_facs_intensity", facs_intensity)

                    # get the facial unit presence (chaining necessary to convert to int)
                    facs_presence = np.array(row[693:711]).astype(float).astype(int)
                    np.save(f"{output_folder_path}/{index}_facs_presence", facs_presence)

                    rigid_face_shape = np.array(row[636:642]).astype(float)
                    np.save(f"{output_folder_path}/{index}_rigid_face_shape", rigid_face_shape)

                    nonrigid_face_shape = np.array(row[642:676]).astype(float)
                    np.save(f"{output_folder_path}/{index}_nonrigid_face_shape", nonrigid_face_shape)

                    pose = np.array(row[290:296]).astype(float)
                    # pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz
                    np.save(f"{output_folder_path}/{index}_pose", pose)

                    line_count += 1

    print("Feature extraction complete")


if __name__ == '__main__':
    extract_open_face_data(args.input_folder, args.output_folder)

#extract_open_face_data("C:/Users/41763/Desktop/OpenFace_2.2.0_win_x64/processed", "C:/Users/41763/Desktop/test")

import os
import cv2

path_original = "Places_Dataset"
path_modified = "Places_Dataset_Modified"

#load batch of pictures
def load_picture_names():
    names = []
    for folder in os.listdir(path_original):
        folder_path = os.path.join(path_original, folder)
        if not os.path.isdir(folder_path):
            continue
        for picture in os.listdir(folder_path):
            if ".jpg" not in picture:
                continue
            names.append(os.path.join(folder,picture))
    return names


def get_modified(pictures):
    images = []
    for picture in pictures:
        images.append(cv2.imread(os.path.join(path_modified,picture.replace(".jpg", ".png"))))
    return images

def get_originals(pictures):
    images = []
    for picture in pictures:
        images.append(cv2.imread(os.path.join(path_original,picture)))
    return images


def print_result(pictures):
    pass
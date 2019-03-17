'''
TODO:
1) random rectangles (shape and count and location) (with limit)
start with one small box, random location
2) load pictures as png
3) mark as black and as alphachannel
4) save modified pictures as png
'''
import os
import random
import cv2
import numpy as np


# random rectangles (shape and count and location)
def create_rectangles(img_size, count=1, size_limit=16):
    rectangles = []
    for i in range(count):
        x_size = random.randint(10, size_limit)
        y_size = random.randint(10, size_limit)
        x_loc = random.randint(0, img_size[0] - x_size)
        y_loc = random.randint(0, img_size[1] - y_size)

        rectangles.append((x_loc, y_loc, x_size, y_size))
    return rectangles


# mark as black and as alphachannel and save
def modify_pictures(src_path, out_path, img_size):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for picture in os.listdir(src_path):
        if ".jpg" in picture:
            picture_path = os.path.join(src_path, picture)
            rectangles = create_rectangles(img_size)
            img = cv2.imread(picture_path, 1)
            # add transparancy channel
            img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            # 0 is fully transparent, 255 not transparent
            alphachannel = np.ones(img_size) * 255
            for x_loc, y_loc, x_size, y_size in rectangles:
                # set colorchannels to black
                alphachannel[y_loc:y_loc + y_size + 1, x_loc:x_loc + x_size + 1] = 0
                img[:,:,0][y_loc:y_loc + y_size + 1, x_loc:x_loc + x_size + 1] = 0
                img[:,:,1][y_loc:y_loc + y_size + 1, x_loc:x_loc + x_size + 1] = 0
                img[:,:,2][y_loc:y_loc + y_size + 1, x_loc:x_loc + x_size + 1] = 0

            # add alphachannel
            img[:, :, 3] = alphachannel
            cv2.imwrite(os.path.join(out_path, picture[:-4] + '.png'), img)

if __name__ == "__main__":
    src_path = "../../Places_Dataset"
    out_path = "../../Places_Dataset_Modified"
    img_size = (256, 256)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for folder in os.listdir(src_path):
        folderpath = os.path.join(src_path, folder)
        if os.path.isdir(folderpath):
            modify_pictures(folderpath, os.path.join(out_path, folder), img_size)

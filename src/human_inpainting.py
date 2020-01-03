#generates masks for humans and gives back a mask for filling
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np
import subprocess

class Masker:

    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def create_mask(self, image_path):
        img = Image.open(image_path)
        x = TF.to_tensor(img)
        x.unsqueeze_(0)
        out = self.model(x)[0]
        people_masks = out['masks'][out['labels'] == 1] #only use human masks

        #threshold 0.5 for mask from pytorch doku
        mask_sums = people_masks[:,0,:,:].sum(axis=0) > 0.0
        #create actual mask
        black_img = np.zeros(x.shape[1:])
        black_img[0][mask_sums] = 255
        black_img[1][mask_sums] = 255
        black_img[2][mask_sums] = 255
        black_img = black_img.astype(np.uint8).transpose(1, 2, 0)

        h, w = black_img.shape[:2]
        # Add an alpha channel, fully opaque (255)
        RGBA = np.dstack((black_img, np.zeros((h, w), dtype=np.uint8) + 255))
        # Make mask of black pixels - mask is True where image is black
        mBlack = (RGBA[:, :, 0:3] == [0, 0, 0]).all(2)
        # Make all pixels matched by mask into transparent ones
        RGBA[mBlack] = (0, 0, 0, 0)
        Image.fromarray(RGBA).save(f'samples/mask_{image_name}.png')

        np_img = np.array(img)
        np_img[:,:,0][mask_sums] = 255
        np_img[:,:,1][mask_sums] = 255
        np_img[:,:,2][mask_sums] = 255

        Image.fromarray(np_img).save(f'samples/input_{image_name}.png')


        #visualize masks
        #plt.imshow(img)
        #plt.imshow(black_img, cmap='jet', alpha=0.3)
        #plt.show()

if __name__ == '__main__':
    image_name = "small_topview"
    #create masks
    with torch.no_grad():
        masker = Masker()
        masker.create_mask(f"samples/{image_name}.jpg")

    #inpainting
    command = f"python test.py --image ../samples/input_{image_name}.png --mask ../samples/mask_{image_name}.png --output ../samples/output_{image_name}.png --checkpoint model_logs/model"
    args = command.split(' ')
    subprocess.run(args, cwd='generative_inpainting')
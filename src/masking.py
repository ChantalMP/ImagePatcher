#generates masks for humans and gives back a mask for filling
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms.functional as TF
from PIL import Image
import torch
from matplotlib import pyplot as plt
import numpy as np

class Masker:

    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def get_mask(self, image_path):
        img = Image.open(image_path)
        x = TF.to_tensor(img)
        x.unsqueeze_(0)
        out = self.model(x)[0]
        people_masks = out['masks'][out['labels'] == 1]

        #threshold 0.5 for mask from pytorch doku
        mask_sums = people_masks[:,0,:,:].sum(axis=0) > 0.5
        #create actual mask
        black_img = np.zeros(x.shape[1:])
        black_img[0][mask_sums] = 1
        black_img[1][mask_sums] = 1
        black_img[2][mask_sums] = 1

        #visualize masks
        #plt.imshow(img)
        #plt.imshow(black_img.transpose(1,2,0), cmap='jet', alpha=0.5)
        #plt.show()

        return black_img

if __name__ == '__main__':
    with torch.no_grad():
        masker = Masker()
        masker.get_mask("samples/small.jpg")
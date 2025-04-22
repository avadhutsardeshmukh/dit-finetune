import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch, torchvision
from torchvision import transforms, datasets
from datasets import load_dataset
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def show_images(im_batch):
    #Unnormalize from [-1, 1] to [0,1]
    im_batch = im_batch*0.5 + 0.5 
    grid = torchvision.utils.make_grid(im_batch)
    grid = grid.detach().cpu().permute(1,2,0) * 255
    grid_im = Image.fromarray(np.array(grid).astype(np.uint8))
    return grid_im


def dataloader(image_size, batch_size, part_type):
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size,image_size)),
            #transforms.RandomHorizontalFlip(),
            #TODO : RandomVerticalFlip, RandomRotation
            transforms.ToTensor(),
            transforms.Normalize([0.5],[0.5]), # [-1,1] Normalization
        ]
    )
    #def transform(examples):
        #images = [preprocess(image) for image in examples['image']]
        #return {'images': images}
        
    #Direct access from hf
    #dataset = load_dataset("DefectSpectrum/Defect_Spectrum", split="train")

    #In case the dataset is locally stored
    dataset = datasets.ImageFolder(
            os.path.join('Defect_Spectrum/DS-MVTec',
            part_type, 'image'),
            transform=preprocess
            )
    #dataset.set_transform(transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataset, dataloader





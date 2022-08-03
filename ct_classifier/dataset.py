'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import json
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image


class CTDataset(Dataset):

    def __init__(self, data_root, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = data_root
        self.split = split
        self.transform = ToTensor()
        
        # index data into list
        self.data = []

        # load annotation file
        annoPath = os.path.join(
            self.data_root,
            'eccv_18_annotation_files',
            'train_annotations.json' if self.split=='train' else 'cis_val_annotations.json'
        )
        meta = json.load(open(annoPath, 'r'))
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]

        # load image
        image_path = os.path.join(self.data_root, image_path)
        img = Image.open(image_path)

        # transform: convert to torch.Tensor
        # here's where we could do data augmentation:
        # https://pytorch.org/vision/stable/transforms.html
        # see Björn's lecture on Thursday, August 11.
        # For now, we only convert the image to torch.Tensor
        img_tensor = self.transform(img)

        return img_tensor, label
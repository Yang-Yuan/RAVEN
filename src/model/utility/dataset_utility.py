import os
import glob
import numpy as np
from scipy import misc

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

        
class ToTensor(object):
    def __call__(self, sample):
        return torch.tensor(sample, dtype=torch.float32)

class dataset(Dataset):
    def __init__(self, root_dir, dataset_type, img_size, transform=None, shuffle=False):
        self.root_dir = root_dir
        self.transform = transform
        self.file_names = [f for f in glob.glob(os.path.join(root_dir, "*", "*.npz")) \
                            if dataset_type in f]
        self.img_size = img_size

        # word embeddings are stored as a numpy.ndarray of object of trivial shape, i.e., scalar
        self.embeddings_dict = np.load(os.path.join(root_dir, 'embedding.npy'),
                                  allow_pickle = True, encoding = "bytes").item()
        self.shuffle = shuffle

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        data_path = self.file_names[idx]
        data = np.load(data_path)

        # image = data["image"].reshape(16, 160, 160) # the image is already in this shape
        image = data["image"]
        target = data["target"]

        # structure, pre-order traverse the A-Sig tree, to the level of "Layout"
        structure = data["structure"]

        # META_TARGET_FORMAT = ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
        # META_STRUCTURE_FORMAT = ["Singleton", "Left_Right", "Up_Down", "Out_In", "Left", "Right", "Up", "Down", "Out", "In", "Grid", "Center_Single", "Distribute_Four", "Distribute_Nine", "Left_Center_Single", "Right_Center_Single", "Up_Center_Single", "Down_Center_Single", "Out_Center_Single", "In_Center_Single", "In_Distribute_Four"]
        meta_target = data["meta_target"]
        meta_structure = data["meta_structure"]

        if self.shuffle:
            context = image[:8, :, :]
            choices = image[8:, :, :]
            indices = range(8)
            np.random.shuffle(indices)
            new_target = indices.index(target)
            new_choices = choices[indices, :, :]
            image = np.concatenate((context, new_choices))
            target = new_target
        
        resize_image = []
        for idx in range(0, 16):
            resize_image.append(misc.imresize(image[idx,:,:], (self.img_size, self.img_size)))
        resize_image = np.stack(resize_image)

        # the structure can have at most 6 non-slask strings and
        # word vectors of length 300
        embedding = torch.zeros((6, 300), dtype=torch.float)
        indicator = torch.zeros(1, dtype=torch.float)
        element_idx = 0
        for element in structure:
            # the strings read from the file are bytes encoding, not sure why.
            if element != b'/':
                embedding[element_idx, :] = torch.tensor(self.embeddings_dict.get(element), dtype=torch.float)
                element_idx += 1

        if element_idx == 6:
            indicator[0] = 1.

        if self.transform:
            resize_image = self.transform(resize_image)
            target = torch.tensor(target, dtype=torch.long)
            meta_target = self.transform(meta_target)
            meta_structure = self.transform(meta_structure)

        del data

        return resize_image, target, meta_target, meta_structure, embedding, indicator
        

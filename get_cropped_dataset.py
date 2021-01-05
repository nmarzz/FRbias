from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training, extract_face
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler
from torchvision import datasets, transforms
import numpy as np
import os

ethnicities = ['Asian','African','Caucasian','Indian']

for  ethnic in ethnicities:
    # Define the dataset root
    data_dir = os.path.join('rfw/data/',ethnic)
    pairs_path = os.path.join('rfw/txts/',ethnic, ethnic + '_pairs.txt')

    batch_size = 16
    epochs = 15
    workers = 0 if os.name == 'nt' else 8

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
        image_size=160,
        margin=14,
        device=device,
        selection_method='center_weighted_size'
    )
    # Define the data loader for the input set of images
    orig_img_ds = datasets.ImageFolder(data_dir, transform=None)

    # overwrites class labels in dataset with path so path can be used for saving output in mtcnn batches
    orig_img_ds.samples = [
        (p, p)
        for p, _ in orig_img_ds.samples
    ]

    loader = DataLoader(
        orig_img_ds,
        num_workers=workers,
        batch_size=batch_size,
        collate_fn=training.collate_pil
    )
    crop_paths = []
    box_probs = []

    for i, (x, b_paths) in enumerate(loader):
        crops = [p.replace(data_dir, data_dir + '_cropped') for p in b_paths]
        mtcnn(x, save_path=crops)
        crop_paths.extend(crops)
        print('\r{}: Batch {} of {}'.format(ethnic,i + 1, len(loader)), end='')

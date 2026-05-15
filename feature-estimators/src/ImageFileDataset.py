import os
from PIL import Image
from torch.utils.data import Dataset


class ImageFileDataset(Dataset):
    def __init__(self, folder, files, transform):
        self.folder = folder
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.folder, self.files[i])).convert("RGB")
        return self.transform(img)

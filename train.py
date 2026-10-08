import os, cv2, torch
from torch.utils.data import Dataset

class XrayDataset(Dataset):
    def __init__(self, root, files, transform=None):
        self.root, self.files, self.transform = root, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.imread(os.path.join(self.root, "images", name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.root, "masks", name), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32")

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 1 -> 3 channels for the pretrained encoder

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        return img, mask.unsqueeze(0)

import os, random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# 1. List the files and split 8 / 2
files = sorted(os.listdir("data/images"))
random.seed(42)
random.shuffle(files)
train_files, val_files = files[:8], files[8:]

# 2. Augmentations (train gets random ones, val only gets resize + normalize)
train_tf = A.Compose([
    A.Resize(512, 512),
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])
val_tf = A.Compose([
    A.Resize(512, 512),
    A.Normalize(),
    ToTensorV2(),
])

# 3. Start the class (this is the part you asked about)
train_ds = XrayDataset("data", train_files, transform=train_tf)
val_ds   = XrayDataset("data", val_files,   transform=val_tf)

# 4. DataLoaders
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False)

img, mask = train_ds[0]
print(img.shape, mask.shape)   # expect: [3, 512, 512] and [1, 512, 512]
print(mask.unique())           # expect: tensor([0., 1.])

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
        if img is None:
            raise FileNotFoundError(f"Image not found: {name}")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {name}")
        mask = cv2.imread(os.path.join(self.root, "masks", name), cv2.IMREAD_GRAYSCALE)
        # remove: mask = (mask > 127).astype("float32")
        ...

        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 1 -> 3 channels for the pretrained encoder

        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]

        if mask is None:
            raise FileNotFoundError(f"Mask not found: {name}")

        return img, mask.long()          # shape [H, W], no unsqueeze

import os, random
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

# 1. List the files and split 8 / 2
files = sorted(os.listdir("data/images"))
random.seed(42)
random.shuffle(files)
n = int(0.8* len(files))
train_files, val_files = files[:n], files[n:]

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

import torch
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, classes=1).to(device)

dice = smp.losses.DiceLoss(mode="multiclass")
ce = torch.nn.CrossEntropyLoss()

def loss_fn(pred, target):
    return dice(pred, target) + ce(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def dice_score(pred, target):
    pred = pred.argmax(1)
    scores = []
    for c in range(1, NUM_CLASSES):
        p, t = (pred == c).float(), (target == c).float()
        if t.sum() + p.sum() > 0:
            scores.append(((2 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)).item())
    return sum(scores) / max(len(scores), 1)

best = 0
for epoch in range(150):
    # train
    model.train()
    train_loss = 0
    for img, mask in train_loader:
        img, mask = img.to(device), mask.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(img), mask)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # validate
    model.eval()
    val_dice = 0
    with torch.no_grad():
        for img, mask in val_loader:
            img, mask = img.to(device), mask.to(device)
            val_dice += dice_score(model(img), mask)
    val_dice /= len(val_loader)

    print(f"Epoch {epoch+1}: loss {train_loss/len(train_loader):.4f} | val dice {val_dice:.4f}")

    if val_dice > best:
        best = val_dice
        os.makedirs("outputs", exist_ok=True)
        torch.save(model.state_dict(), "outputs/best.pth")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["loss"])
ax[0].set_title("Train loss")
ax[0].set_xlabel("Epoch")
ax[1].plot(history["val_dice"])
ax[1].set_title("Val dice")
ax[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("outputs/results.png", dpi=150)


"""
Step 2: train.   python train.py
Needs: data/images, data/masks (from convert_to_mask.py), classes.json
"""
import os, json, random
import cv2, numpy as np, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------- settings ----------------
ROOT = "data"
IMG_SIZE = 512
BATCH_SIZE = 2
EPOCHS = 150
LR = 1e-4
ENCODER = "resnet34"
# ------------------------------------------

with open("classes.json") as f:
    CLASSES = json.load(f)
NUM_CLASSES = len(CLASSES) + 1          # + background
print("Classes:", {i + 1: c for i, c in enumerate(CLASSES)})


def clean_file_list(root):
    """Keep only image/mask pairs that exist, are readable, same size, valid class IDs."""
    imgs = set(os.listdir(f"{root}/images"))
    msks = set(os.listdir(f"{root}/masks"))
    good = []
    for name in sorted(imgs):
        if name not in msks:
            print(f"SKIP {name}: no mask"); continue
        img = cv2.imread(f"{root}/images/{name}", cv2.IMREAD_GRAYSCALE)
        m = cv2.imread(f"{root}/masks/{name}", cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            print(f"SKIP {name}: unreadable"); continue
        if img.shape != m.shape:
            print(f"SKIP {name}: size mismatch {img.shape} vs {m.shape}"); continue
        if m.max() == 0:
            print(f"SKIP {name}: empty mask"); continue
        if m.max() >= NUM_CLASSES:
            print(f"SKIP {name}: mask value {m.max()} >= NUM_CLASSES {NUM_CLASSES}"); continue
        good.append(name)
    print(f"{len(good)} usable image/mask pairs")
    return good


class XrayDataset(Dataset):
    def __init__(self, root, files, transform=None):
        self.root, self.files, self.transform = root, files, transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        name = self.files[i]
        img = cv2.imread(os.path.join(self.root, "images", name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.root, "masks", name), cv2.IMREAD_GRAYSCALE)
        if img is None or mask is None:
            raise FileNotFoundError(f"Could not read image or mask: {name}")
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)   # 1 -> 3 channels for the pretrained encoder
        if self.transform:
            out = self.transform(image=img, mask=mask)
            img, mask = out["image"], out["mask"]
        return img, mask.long()                       # mask: [H, W] class IDs


files = clean_file_list(ROOT)
if len(files) < 3:
    raise SystemExit("Too few usable images, check the SKIP messages above.")

random.seed(42)
random.shuffle(files)
n = min(max(1, int(0.8 * len(files))), len(files) - 1)
train_files, val_files = files[:n], files[n:]
print(f"train: {len(train_files)}  val: {len(val_files)}")
print("val files:", val_files)

train_tf = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
    A.RandomBrightnessContrast(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])
val_tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

train_ds = XrayDataset(ROOT, train_files, train_tf)
val_ds = XrayDataset(ROOT, val_files, val_tf)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

# sanity check on ALL samples, not just the first
for ds in (train_ds, val_ds):
    for i in range(len(ds)):
        _, m = ds[i]
        assert 0 <= m.min() and m.max() < NUM_CLASSES, (ds.files[i], m.unique())
x0, m0 = train_ds[0]
print("sample shapes:", tuple(x0.shape), tuple(m0.shape))

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model = smp.Unet(encoder_name=ENCODER, encoder_weights="imagenet",
                 in_channels=3, classes=NUM_CLASSES).to(device)
dice_loss = smp.losses.DiceLoss(mode="multiclass")
ce_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def loss_fn(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)


def evaluate():
    """Dice per class over the whole validation set. Returns (mean, per-class list)."""
    model.eval()
    inter = np.zeros(NUM_CLASSES); psum = np.zeros(NUM_CLASSES); tsum = np.zeros(NUM_CLASSES)
    with torch.no_grad():
        for img, mask in val_loader:
            pred = model(img.to(device)).argmax(1).cpu()
            for c in range(1, NUM_CLASSES):
                p, t = pred == c, mask == c
                inter[c] += (p & t).sum().item()
                psum[c] += p.sum().item()
                tsum[c] += t.sum().item()
    per_class = []
    for c in range(1, NUM_CLASSES):
        d = 2 * inter[c] / (psum[c] + tsum[c]) if psum[c] + tsum[c] > 0 else float("nan")
        per_class.append(d)
    valid = [d for d in per_class if not np.isnan(d)]
    return (sum(valid) / len(valid) if valid else 0.0), per_class


history = {"loss": [], "val_dice": []}
best = -1
os.makedirs("outputs", exist_ok=True)

try:
    for epoch in range(EPOCHS):
        model.train()
        total, steps = 0.0, 0
        for img, mask in train_loader:
            try:
                img, mask = img.to(device), mask.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(img), mask)
                loss.backward()
                optimizer.step()
                total += loss.item(); steps += 1
            except RuntimeError as e:
                print("skipped a bad batch:", str(e)[:120])
        train_loss = total / max(steps, 1)

        val_dice, per_class = evaluate()
        history["loss"].append(train_loss)
        history["val_dice"].append(val_dice)
        pc = " ".join(f"{CLASSES[i]}={d:.2f}" for i, d in enumerate(per_class))
        print(f"Epoch {epoch+1}: loss {train_loss:.4f} | val dice {val_dice:.4f} | {pc}")

        if val_dice > best:
            best = val_dice
            torch.save({"model": model.state_dict(), "classes": CLASSES,
                        "encoder": ENCODER, "img_size": IMG_SIZE}, "outputs/best.pth")
except KeyboardInterrupt:
    print("Stopped by user, saving the plot...")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history["loss"]); ax[0].set_title("Train loss"); ax[0].set_xlabel("Epoch")
ax[1].plot(history["val_dice"]); ax[1].set_title("Val dice (mean over classes)"); ax[1].set_xlabel("Epoch")
plt.tight_layout()
plt.savefig("outputs/results.png", dpi=150)
print(f"Done. Best val dice {best:.4f}. Saved outputs/best.pth and outputs/results.png")

Claude finished the response

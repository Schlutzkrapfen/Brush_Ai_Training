"""
Step 3: predict.   python predict.py data/images/926.png
                   python predict.py some_folder/
Writes to outputs/predictions/: class-ID mask, visible mask, colored overlay.
"""
import os, sys, json
import cv2, numpy as np, torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt = torch.load("outputs/best.pth", map_location=device)
CLASSES = ckpt["classes"]
NUM_CLASSES = len(CLASSES) + 1
IMG_SIZE = ckpt.get("img_size", 512)

model = smp.Unet(encoder_name=ckpt.get("encoder", "resnet34"), encoder_weights=None,
                 in_channels=3, classes=NUM_CLASSES).to(device)
model.load_state_dict(ckpt["model"])
model.eval()

tf = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])
COLORS = [(255, 0, 0), (0, 255, 0), (0, 128, 255), (255, 255, 0),
          (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255)]


def predict(path, out_dir="outputs/predictions"):
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print("SKIP (unreadable):", path); return
    os.makedirs(out_dir, exist_ok=True)
    h, w = gray.shape
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    x = tf(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x).argmax(1)[0].cpu().numpy().astype("uint8")
    mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = rgb.copy()
    for c in range(1, NUM_CLASSES):
        sel = mask == c
        col = np.array(COLORS[(c - 1) % len(COLORS)])
        overlay[sel] = (0.6 * overlay[sel] + 0.4 * col).astype("uint8")

    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    cv2.imwrite(f"{out_dir}/mask_{stem}.png", mask)                          # class IDs
    cv2.imwrite(f"{out_dir}/maskvis_{stem}.png", mask * (255 // NUM_CLASSES))  # viewable
    cv2.imwrite(f"{out_dir}/overlay_{stem}.png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    found = [CLASSES[c - 1] for c in np.unique(mask) if c > 0]
    print("saved", name, "->", found or "nothing found")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python predict.py image.png [more images or folders]")
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                predict(os.path.join(arg, f))
        else:
            predict(arg)

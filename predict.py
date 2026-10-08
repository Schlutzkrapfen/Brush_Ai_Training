import os, cv2, torch, numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp

device = "cuda" if torch.cuda.is_available() else "cpu"

# load the model with the best weights
model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                 in_channels=3, classes=1).to(device)
model.load_state_dict(torch.load("outputs/best.pth", map_location=device))
model.eval()

tf = A.Compose([A.Resize(512, 512), A.Normalize(), ToTensorV2()])

def predict(path, out_dir="outputs/predictions"):
    os.makedirs(out_dir, exist_ok=True)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = gray.shape
    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    x = tf(image=rgb)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

    # back to the original image size
    mask = cv2.resize((prob > 0.5).astype("uint8") * 255, (w, h),
                      interpolation=cv2.INTER_NEAREST)

    # overlay: predicted mask in red on top of the X-ray
    overlay = rgb.copy()
    overlay[mask > 0] = (0.6 * overlay[mask > 0] + 0.4 * np.array([255, 0, 0])).astype("uint8")

    name = os.path.basename(path)
    cv2.imwrite(f"{out_dir}/mask_{name}", mask)
    cv2.imwrite(f"{out_dir}/overlay_{name}", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print("saved", name)

if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        predict(p)

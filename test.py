import torch, os, cv2
import numpy as np
from torchvision import models, transforms
import torch.nn as nn

VIDEO_PATH = "../dataset/ysj.mp4"
device = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Model
# ==========================
class TemporalConvNet(nn.Module):
    def __init__(self, in_dim=512, num_classes=10):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_dim, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(256, num_classes, 3, padding=1)
        )

    def forward(self, x):
        return self.tcn(x)

# ==========================
# Load Models
# ==========================
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

label_map = np.load("label_map.npy", allow_pickle=True).item()
inv_label_map = {v: k for k, v in label_map.items()}

model = TemporalConvNet(in_dim=512, num_classes=len(label_map))
model.load_state_dict(torch.load("tcn_model.pth", map_location=device))
model = model.to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ==========================
# Extract features from video
# ==========================
print(f"Extracting features from {VIDEO_PATH}...")
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
features = []

while True:
    ret, frame = cap.read()
    print(f"Processing frame {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}", end="\r")
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = transform(frame).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = resnet(tensor).cpu().numpy()
    features.append(feat)

cap.release()
X = torch.tensor(np.concatenate(features, axis=0)).float().T.unsqueeze(0).to(device)

# ==========================
# Predict frame-level action
# ==========================
print(f"\nPredicting actions for {X.shape[2]} frames...")
with torch.no_grad():
    logits = model(X)  # (1, num_classes, T)
    preds = logits.argmax(1).squeeze(0).cpu().numpy()

# ==========================
# Merge consecutive segments
# ==========================
print(f"Merging {len(preds)} frames into action segments...")
segments = []
prev = preds[0]
start = 0
for i in range(1, len(preds)):
    if preds[i] != prev:
        segments.append((start/fps, i/fps, inv_label_map[prev]))
        start = i
        prev = preds[i]
segments.append((start/fps, len(preds)/fps, inv_label_map[prev]))

print("Detected actions:")
for s, e, label in segments:
    print(f"{s:.2f}–{e:.2f}s: {label}")

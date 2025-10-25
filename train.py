import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
import pandas as pd
import numpy as np
import subprocess

# ==========================
# 1. Config
# ==========================
VIDEO_DIR = "../dataset"
ANNOT_DIR = "../dataset"
FEATURE_DIR = "../dataset"
os.makedirs(FEATURE_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================
# 2. Model (Feature + Temporal)
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
        return self.tcn(x)  # (B, num_classes, T)

# ==========================
# 3. Helper: Extract frames
# ==========================
def extract_clip_ffmpeg(video_path, start_time, end_time, out_dir, fps=2):
    os.makedirs(out_dir, exist_ok=True)
    duration = end_time - start_time
    cmd = [
        "ffmpeg", "-ss", str(start_time), "-t", str(duration),
        "-i", video_path, "-vf", f"fps={fps},scale=224:224",
        os.path.join(out_dir, "frame_%05d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ]
    subprocess.run(cmd)

# ==========================
# 4. Helper: Feature extraction
# ==========================
def extract_features_from_frames(frame_dir, resnet, transform):
    features = []
    frames = sorted(os.listdir(frame_dir))
    for f in frames:
        path = os.path.join(frame_dir, f)
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = resnet(tensor).cpu().numpy()
        features.append(feat)
    return np.concatenate(features, axis=0)  # (T, 512)

# ==========================
# 5. Load ResNet backbone
# ==========================
resnet = models.resnet18(weights="IMAGENET1K_V1")
resnet.fc = nn.Identity()
resnet = resnet.to(device).eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# ==========================
# 6. Prepare dataset
# ==========================
def time_to_sec(t):
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

clips = []
labels = []

label_map = {}  # action_name → class_idx
for csv_name in os.listdir(ANNOT_DIR):
    if not csv_name.endswith(".csv"):
        continue
    base = csv_name[:-4]
    video_path = os.path.join(VIDEO_DIR, base + ".mp4")
    df = pd.read_csv(os.path.join(ANNOT_DIR, csv_name))
    for _, row in df.iterrows():
        start = time_to_sec(row["start"])
        end = time_to_sec(row["end"])
        action = row["action"]
        if action not in label_map:
            label_map[action] = len(label_map)

        clip_dir = os.path.join(FEATURE_DIR, f"{base}_{action}_{int(start)}_{int(end)}")
        extract_clip_ffmpeg(video_path, start, end, clip_dir)
        feats = extract_features_from_frames(clip_dir, resnet, transform)
        np.save(os.path.join(FEATURE_DIR, f"{base}_{action}_{int(start)}_{int(end)}.npy"), feats)
        clips.append(feats)
        labels.append(label_map[action])

print("Label map:", label_map)

# ==========================
# 7. Train TCN
# ==========================
X = [torch.tensor(f).float().T for f in clips]  # (512, T)
Y = [torch.tensor([l]) for l in labels]
num_classes = len(label_map)

model = TemporalConvNet(in_dim=512, num_classes=num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    total = 0
    correct = 0
    for x, y in zip(X, Y):
        x = x.unsqueeze(0).to(device)
        y = y.to(device)
        out = model(x).mean(dim=2)  # temporal average
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += 1
    print(f"Epoch {epoch}: acc={correct/total:.3f}")

torch.save(model.state_dict(), "tcn_model.pth")
np.save("label_map.npy", label_map)
print("✅ Training done.")

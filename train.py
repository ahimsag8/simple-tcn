import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision import transforms
from dataset import TemporalActionDataset
from temporal import TemporalConvNet

device = "cuda" if torch.cuda.is_available() else "cpu"

# Feature extractor (ResNet18)
resnet = resnet18(pretrained=True)
resnet.fc = nn.Identity()
resnet = resnet.to(device)
resnet.eval()

# Temporal model
model = TemporalConvNet(in_dim=512).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

dataset = TemporalActionDataset("..\\dataset", "..\\dataset")
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

for epoch in range(5):
    for frames, labels in loader:
        B, T, C, H, W = frames.shape
        frames = frames.view(T, C, H, W).to(device)
        with torch.no_grad():
            feats = resnet(frames).detach()  # (T, 512)
        feats = feats.unsqueeze(0).permute(0, 2, 1)  # (B, 512, T)
        labels = labels.float().to(device)

        out = model(feats)  # (B, T)
        loss = criterion(out.squeeze(), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}: loss={loss.item():.4f}")

import os
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as T
import cv2
import csv
from torchvision.models import resnet18

def time_to_sec(t):
    h, m, s = t.split(":")
    return int(h)*3600 + int(m)*60 + float(s)

class TemporalActionDataset(data.Dataset):
    def __init__(self, video_dir, anno_dir, clip_len=16, fps=2):
        self.video_dir = video_dir
        self.anno_dir = anno_dir
        self.clip_len = clip_len
        self.fps = fps
        self.samples = []  # (video_path, frames, labels)
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224,224)),
            T.ToTensor()
        ])

        for csv_file in os.listdir(anno_dir):
            if not csv_file.endswith(".csv"):
                continue
            name = os.path.splitext(csv_file)[0]
            video_path = os.path.join(video_dir, name + ".mp4")
            anno_path = os.path.join(anno_dir, csv_file)
            if not os.path.exists(video_path):
                continue
            self.samples.append((video_path, anno_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, anno_path = self.samples[idx]
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # 라벨 타임 구간 읽기
        actions = []
        with open(anno_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for start, end, label in reader:
                actions.append((time_to_sec(start), time_to_sec(end), label))

        # 일정 간격으로 프레임 추출
        frame_list, label_list = [], []
        frame_interval = int(fps / self.fps)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                t = frame_idx / fps
                label = 0  # default: background
                for s, e, l in actions:
                    if s <= t <= e:
                        label = 1  # action 구간이면 1
                        break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = self.transform(frame)
                frame_list.append(frame)
                label_list.append(label)
            frame_idx += 1
        cap.release()

        x = torch.stack(frame_list)  # (T, 3, 224, 224)
        y = torch.tensor(label_list) # (T,)
        return x, y

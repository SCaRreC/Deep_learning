import torch
import cv2
from PIL import Image
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_transform_features(features):
    features_numeric = features.select_dtypes(include=['int64', 'float64']).copy()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(features_numeric)
    return scaler, X_train_scaled

def transform_features(df, scaler):
    numeric = df.select_dtypes(include=['int64', 'float64']).copy()
    X_scaled = scaler.transform(numeric)
    return X_scaled

class ds_poi(Dataset):
    def __init__(self, target, image_path, features, transform=None):
        assert len(target) == len(image_path) == len(features)
        self.target = torch.tensor(target)
        self.features = torch.tensor(features)
        self.image_path = image_path
        self.transform = transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        target = self.target[idx]
        features = self.features[idx]
        im = cv2.imread(self.image_path[idx])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(im)
        if self.transform is not None:
            im = self.transform(im)
        return target, features, im

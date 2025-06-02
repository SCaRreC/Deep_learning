utils.py
import torch
from torch.utils.data import Dataset
from torchvision import transformsimport cv2
from sklearn.preprocessing import StandardScalerimport numpy as np
###############################
# Data Processing #
##############################

def fit_transform_features(features):
  """
  Receives a DataFrame with numeric columns and returns a NumPy array
  with standardized values (zero mean, unit variance).
  Assumes that non-numeric or unwanted columns (e.g. image paths) are already excluded.
  """
  features_numeric = features.select_dtypes(include=['int64', 'float64']).copy()
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(features_numeric)
  return scaler, X_train_scaled

def transform_features(df, scaler):
    """Transforms dataframes with a scaler already trained in the train subset."""
    numeric = df.select_dtypes(include=['int64', 'float64']).copy()
    X_scaled = scaler.transform(numeric)
    return X_scaled

# Custom Dataset #

class ds_poi(Dataset):
  """
  Class that facilitates dataset processing to load it into de DataLoader.
  it needs:
  - Initialize the dataset
  - get the length of the data set
  - obtain every item in the dataset
  """

  def __init__(self, target, image_path, features, transform=None):
    assert len(target) == len(image_path) == len(features)
    # assert que target y features sean np.array o tensor
    self.target = torch.tensor(target)
    #processed_features = process_features(features)
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
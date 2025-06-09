# Imports 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

    
# Fijamos todas las semillas aleatorias para reproducibilidad
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#################
# Data processing#
#################

def data_processing(df_poi:object):
    """
    Transforms raw data frame from .csv Artgonuts according to all the decisions taken in the EDA notebook 
    to implement in other notebooks
    Returns: 
        transformed_df: ready to be split into train, val and test.
    """
    #Categoric columns tranformations

    # transform content to lists
    df_poi['categories'] = df_poi['categories'].apply(eval)
    df_poi['tags'] = df_poi['tags'].apply(eval)
    # Categories
    empty_rows = df_poi[df_poi['categories'].str.len() == 0]
    for idx in empty_rows.index:
        df_poi.at[idx, 'categories'] = ['Cultura', 'Ocio']
        print(df_poi.loc[idx, 'categories'])
    # One-hot encoding for. categories
    all_categories = set(cat for sublist in df_poi['categories'] for cat in sublist)
    for category in all_categories:
        df_poi[category] = df_poi['categories'].apply(lambda x: 1 if category in x else 0)
    df_poi = df_poi.drop('categories', axis=1)
    
    # tags into num_tags
    df_poi['number_tags'] = df_poi['tags'].apply(len)
    df_poi.drop('tags', axis=1, inplace=True)
    df_poi['number_tags'].hist()

    # images
    base_path = os.path.join('Deep_learning', 'data', 'raw')
    df_poi['main_image_path'] = base_path + '/'+ df_poi['main_image_path']

    # Create 'engagement' measurement and 'engagement_score'
    df_poi['engagement'] = 0.8 * df_poi['Likes'] +  df_poi['Bookmarks'] + 0.2* df_poi['Visits'] - 0.5 * df_poi['Dislikes']
    engagement_scores = pd.qcut(df_poi['engagement'], q=3, labels=[0,1,2])

    # Drop columns with text, (we won't be using them in this model), and columns: Visits, likes, dislikes and bookmarks, as we already calculated our target score "engagement_score" with them.
    df_poi.drop(columns=['id', 'name', 'shortDescription', 'Visits', 'Likes', 'Dislikes', 'Bookmarks'], inplace=True)
    
    return df_poi

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
#############
# Trainning #
#############

def train_epoch(model: nn.Module, device: torch.device, train_loader: DataLoader,
                criterion, optimizer, l1_lambda=None, scheduler=None):
    """
    Train one epoch of the neural network and return the training metrics.
    Adapted for HybridModel which processes both features and images.
    Args:
        model: Neural network model to train (HybridModel)
        device: Device where training will be performed (CPU/GPU)
        train_loader: DataLoader with training data (returns target, features, images)
        criterion: Loss function to use
        optimizer: Optimizer for updating weights
        scheduler: Scheduler to adjust learning rate
        l1_lambda: L1 regularization factor (optional)

    Returns:
        train_loss: Average loss on the training set
        train_acc: Accuracy on the training set (%)
        current_lr: Current learning rate after scheduler (if scheduler exists)
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (target, features, images) in enumerate(train_loader):
        features, target, images = features.to(device), target.to(device), images.to(device)
        
        optimizer.zero_grad()

        # Forward pass 
        output = model(features, images)

        loss = criterion(output, target)

        # Regularization if specified
        if l1_lambda is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_norm

        # Backward pass y optimization
        loss.backward()
        optimizer.step()

        # metrics
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    # averages
    train_loss /= len(train_loader)
    train_acc = 100. * correct / total

    # Apply scheduler after each epoch (if any)
    if scheduler is not None:
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        return train_loss, train_acc, current_lr
    else:
        return train_loss, train_acc

def eval_epoch(model: nn.Module, device: torch.device, val_loader: DataLoader,
               criterion):
    """  
    Evaluates the model on the validation set for HybridModel.  

    Args:  
        model: HybridModel to evaluate  
        device: Device where evaluation will be performed (CPU/GPU)  
        val_loader: DataLoader with validation data (returns target, features, images)  
        criterion: Loss function to use  

    Returns:  
        val_loss: Average loss on the validation set  
        val_acc: Accuracy on the validation set (%)  
    """  
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for target, features, images in val_loader:
            
            features, target, images = features.to(device), target.to(device), images.to(device)

            # Forward pass
            output = model(features, images)

            loss = criterion(output, target)

            # metrics
            val_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total

    return val_loss, val_acc

def plot_training_curves(train_losses, val_losses, train_accs, val_accs, num_epochs, test_acc=None):
    """ 
    Draws training , validation and testing plots for the loss function and accuracy over the different epochs.
    """
    plt.style.use("ggplot")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(num_epochs), train_losses, label="Train Loss")
    plt.plot(range(num_epochs), val_losses, label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(num_epochs), train_accs, label="Train Accuracy")
    plt.plot(range(num_epochs), val_accs, label="Validation Accuracy")
    if test_acc is not None:
        plt.axhline(y=test_acc, color='red', linestyle='--', label='Test Accuracy')
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def early_stopping(val_losses, patience=3):
    """
    Implements early stopping of the model training when the loss function for validation doesn't improve over the course of n (patience) epochs.
    """
    if len(val_losses) < patience + 1:
        return False
    # Verifica si el loss de validación no mejora en 'patience' épocas
    return all(val_losses[-i] >= val_losses[-patience-1] for i in range(1, patience+1))


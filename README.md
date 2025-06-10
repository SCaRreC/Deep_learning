# Predicting Tourist Engagement with Deep Learning

This project is the final assignment for the Deep Learning module. The goal is to predict the **engagement level** of different tourist attractions using both **images** and **tabular data**, based on the Artgonuts dataset.

---

## 📌 Objective

Design a Deep Learning model that classifies the engagement level of tourist attractions (low, medium, or high), combining **visual** and **structured** data. The final model integrates a pre-trained CNN with a fully connected network for tabular inputs.

---

## 🧠 Approach & Strategy

From the beginning, my goal was to build a **functional and reproducible** solution, even if simple, and later expand its complexity if time allowed.

1. **Data exploration & preprocessing**:
   - Cleaning and handling missing values.
   - Visualizing distributions and encoding categorical variables.
   - Reviewing image quality; applying *data augmentation* to improve robustness.

2. **Engagement metric engineering**:
   - Based on correlations between 'likes', 'bookmarks', 'visits', and 'dislikes':
     ```
     engagement = 0.8 * likes + bookmarks + 0.2 * visits - 0.5 * dislikes
     ```
   - This metric was then **categorized into three balanced classes**.

3. **Data splitting**:
   - Train, validation, and test sets with a fixed seed for reproducibility.

4. **Model development**:
   - Initially targeting a simple model and increase its complexity if results were not gooe       enought and time allowed it.

5. **Hyperparameter optimization**:
   - Integrated [Optuna](https://optuna.org/) for efficient automated tuning.
   - This version was developed in the `optimization` branch of the repo.

---

## 🏗️ Model Architecture

The final hybrid model consists of three main components:

### 1. **CNN for image features**
- Based on a pretrained PyTorch model (`resnet18`).
- All layers frozen except the last one.
- Output adapted to intermediate feature space.

### 2. **MLP for tabular features**
- Fully connected neural network with:
  - 2 hidden layers (ReLU activations, Dropout).
  - Input normalization.

### 3. **Final classifier**
- Combines outputs from CNN and MLP.
- Includes 2 more layers.
- Outputs 3 classes using Softmax.

---

## 📈 Results

- **Validation accuracy**: ~80 %
- The model shows consistent performance and significantly outperforms naive baselines.

---

## 🔁 Reproducibility

This repository includes everything needed to reproduce the experiment:

- `optimization_simplified_model.ipynb`: Main notebook with training and evaluation.
- `utils.py`: Helper functions for loading and preprocessing data.
- `modelo_final.pth`: Final trained model weights.
- `requirements.txt`: Development environment dependencies.
- Random seeds set to ensure reproducible results.

---

## 🧪 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/your_username/Deep_learning.git
   cd Deep_learning

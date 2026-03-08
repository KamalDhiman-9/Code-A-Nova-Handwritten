# Code-A-Nova
These are my first month projects as a intern in Code-a-Nova
# 🔢 Handwritten Digit Recognition
### Beginner Python Project using MNIST + TensorFlow/Keras

---

## 📌 What This Project Does
This project trains a Neural Network to recognize handwritten digits (0–9)
using the famous **MNIST dataset** — achieving ~98% accuracy!

---

## 📁 Project Files
| File | Description |
|------|-------------|
| `digit_recognition.py` | Main Python script (train + predict) |
| `sample_images.png` | Generated: sample MNIST images |
| `training_history.png` | Generated: accuracy & loss curves |
| `predictions.png` | Generated: model predictions |
| `digit_recognition_model.keras` | Generated: saved trained model |

---

## ⚙️ Setup & Installation

### 1. Install Python (if not already installed)
Download from: https://www.python.org/downloads/

### 2. Install Required Libraries
Open your terminal/command prompt and run:
```bash
pip install tensorflow numpy matplotlib
```

### 3. Run the Project
```bash
python digit_recognition.py
```

---

## 🧠 How It Works

### Dataset: MNIST
- 60,000 training images + 10,000 test images
- Each image = 28×28 pixels (grayscale)
- Labels = digits 0 through 9

### Model Architecture
```
Input (784 pixels)
    ↓
Dense Layer (128 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Dense Layer (64 neurons, ReLU)
    ↓
Dropout (20%)
    ↓
Output Layer (10 neurons, Softmax → probabilities for each digit)
```

### Key Concepts Explained
| Term | Meaning |
|------|---------|
| **Neuron** | A unit that processes numbers |
| **ReLU** | Activation that ignores negative values |
| **Softmax** | Converts outputs to probabilities (sum = 1) |
| **Dropout** | Randomly disables neurons to prevent overfitting |
| **Epoch** | One full pass through the training data |
| **Batch size** | Number of images processed at once |

---

## 📊 Expected Results
- Training Accuracy: ~99%
- Test Accuracy: ~97–98%

---

## 🔮 What's Next? (Improve the Project)
1. **Try CNN** — Convolutional Neural Networks are even better for images
2. **Draw your own digit** — Use OpenCV to capture a drawn digit and predict it
3. **Web App** — Build a browser UI where users can draw digits

---

## 🐛 Common Issues
| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: tensorflow` | Run `pip install tensorflow` |
| Slow training | Reduce epochs to 5 or batch_size to 64 |
| Low accuracy | Increase epochs to 15 |

# ============================================================
#   Handwritten Digit Recognition using MNIST Dataset
#   Beginner-Friendly Project | Built with TensorFlow/Keras
# ============================================================

# ── STEP 1: Import Libraries ──────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow version:", tf.__version__)


# ── STEP 2: Load and Explore the MNIST Dataset ───────────────
# MNIST has 70,000 images of handwritten digits (0–9)
# Each image is 28x28 pixels in grayscale

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

print("\n── Dataset Info ──")
print(f"Training samples : {X_train.shape[0]}")
print(f"Testing  samples : {X_test.shape[0]}")
print(f"Image shape      : {X_train.shape[1:]}  (28x28 pixels)")
print(f"Pixel value range: {X_train.min()} – {X_train.max()}")


# ── STEP 3: Visualize Sample Images ──────────────────────────
def show_sample_images(images, labels, n=10):
    """Display the first n images from the dataset."""
    plt.figure(figsize=(15, 2))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f"Label: {labels[i]}", fontsize=9)
        plt.axis('off')
    plt.suptitle("Sample MNIST Images", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig("sample_images.png", dpi=100)
    plt.show()
    print("✅ Saved: sample_images.png")

show_sample_images(X_train, y_train)


# ── STEP 4: Preprocess the Data ───────────────────────────────
# Normalize pixel values from [0, 255] → [0.0, 1.0]
# This helps the model learn faster and more accurately

X_train = X_train / 255.0
X_test  = X_test  / 255.0

# Flatten 28x28 images into 784-element 1D vectors
# (required for Dense/Fully-Connected layers)
X_train_flat = X_train.reshape(-1, 28 * 28)
X_test_flat  = X_test.reshape(-1, 28 * 28)

print("\n── After Preprocessing ──")
print(f"X_train shape: {X_train_flat.shape}")
print(f"X_test  shape: {X_test_flat.shape}")


# ── STEP 5: Build the Neural Network Model ────────────────────
# Architecture:
#   Input  → 784 neurons  (one per pixel)
#   Hidden → 128 neurons  (ReLU activation)
#   Hidden → 64  neurons  (ReLU activation)
#   Output → 10  neurons  (one per digit class 0–9, Softmax)

model = keras.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(128, activation='relu'),   # Hidden layer 1
    layers.Dropout(0.2),                    # Randomly drops 20% neurons (prevents overfitting)
    layers.Dense(64, activation='relu'),    # Hidden layer 2
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')  # Output layer: 10 classes
], name="Digit_Recognizer")

model.summary()


# ── STEP 6: Compile the Model ─────────────────────────────────
# optimizer  : Adam – adjusts learning rate automatically
# loss       : sparse_categorical_crossentropy – good for multi-class classification
# metrics    : accuracy – percentage of correct predictions

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# ── STEP 7: Train the Model ───────────────────────────────────
print("\n── Training the Model ──")

history = model.fit(
    X_train_flat, y_train,
    epochs=10,              # Number of full passes through the training data
    batch_size=32,          # Process 32 images at a time
    validation_split=0.1,   # Use 10% of training data for validation
    verbose=1
)


# ── STEP 8: Evaluate on Test Data ────────────────────────────
print("\n── Evaluating on Test Data ──")
test_loss, test_accuracy = model.evaluate(X_test_flat, y_test, verbose=0)
print(f"Test Loss    : {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


# ── STEP 9: Plot Training History ────────────────────────────
def plot_training_history(history):
    """Plot accuracy and loss curves over epochs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axes[0].plot(history.history['accuracy'],     label='Train Accuracy', color='royalblue')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy',   color='orange')
    axes[0].set_title('Model Accuracy', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss
    axes[1].plot(history.history['loss'],     label='Train Loss', color='royalblue')
    axes[1].plot(history.history['val_loss'], label='Val Loss',   color='orange')
    axes[1].set_title('Model Loss', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=100)
    plt.show()
    print("✅ Saved: training_history.png")

plot_training_history(history)


# ── STEP 10: Make Predictions ────────────────────────────────
def predict_digits(model, images, labels, n=10):
    """
    Run predictions on n test images and display results.
    Green title = correct prediction | Red title = wrong prediction
    """
    predictions = model.predict(images[:n], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    plt.figure(figsize=(15, 3))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(images[i].reshape(28, 28), cmap='gray')
        color  = 'green' if predicted_labels[i] == labels[i] else 'red'
        title  = f"P:{predicted_labels[i]}\nA:{labels[i]}"
        plt.title(title, fontsize=9, color=color)
        plt.axis('off')

    plt.suptitle("Predictions (P=Predicted, A=Actual) | Green=Correct, Red=Wrong",
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("predictions.png", dpi=100)
    plt.show()
    print("✅ Saved: predictions.png")

predict_digits(model, X_test_flat, y_test)


# ── STEP 11: Predict a Single Custom Image ───────────────────
def predict_single(model, image_flat, true_label=None):
    """
    Predict the digit in a single 784-element image array.

    Parameters:
        model      : trained Keras model
        image_flat : numpy array of shape (784,), pixel values in [0, 1]
        true_label : (optional) actual label to compare against
    """
    image_flat = image_flat.reshape(1, 784)
    probs = model.predict(image_flat, verbose=0)[0]
    predicted = np.argmax(probs)
    confidence = probs[predicted] * 100

    print(f"\nPredicted Digit : {predicted}")
    print(f"Confidence      : {confidence:.2f}%")
    if true_label is not None:
        result = "✅ Correct" if predicted == true_label else "❌ Wrong"
        print(f"Actual Label    : {true_label}  →  {result}")

    return predicted

# Example: predict the first test image
print("\n── Single Image Prediction ──")
predict_single(model, X_test_flat[0], true_label=y_test[0])


# ── STEP 12: Save the Trained Model ──────────────────────────
model.save("digit_recognition_model.keras")
print("\n✅ Model saved as: digit_recognition_model.keras")
print("   (Load it later with: model = keras.models.load_model('digit_recognition_model.keras'))")


# ── DONE ─────────────────────────────────────────────────────
print("\n🎉 Project complete! Files generated:")
print("   • sample_images.png      — visualizes training samples")
print("   • training_history.png   — accuracy & loss curves")
print("   • predictions.png        — model predictions on test images")
print("   • digit_recognition_model.keras — saved trained model")

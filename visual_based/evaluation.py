import os
import tensorflow as tf
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_image_tensor(x):
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)

    if x.ndim != 4:
        raise ValueError(f"Unsupported X shape: {x.shape}")

    model_channels = model.input_shape[-1]

    if model_channels == 3 and x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif model_channels == 1 and x.shape[-1] == 3:
        x = x[..., :1]
    elif x.shape[-1] != model_channels:
        raise ValueError(
            f"Input channels mismatch. Model expects {model_channels}, got {x.shape[-1]}"
        )

    x = x.astype("float32")
    if x.max() > 1.0:
        x /= 255.0

    return x


# Model path
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully")

# Load data from visual_based folder
X = np.load(os.path.join(BASE_DIR, "X.npy"))
y = np.load(os.path.join(BASE_DIR, "y.npy"))
X = prepare_image_tensor(X)

# Evaluate
loss, accuracy = model.evaluate(X, y, verbose=1)
print("Final Model Accuracy:", accuracy)
print("Final Model Loss:", loss)

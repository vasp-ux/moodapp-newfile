import os
import numpy as np
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# ---------------- BASE DIRECTORY ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def prepare_image_tensor(x):
    """Ensure dataset matches MobileNetV2 input format."""
    if x.ndim == 3:
        x = np.expand_dims(x, axis=-1)

    if x.ndim != 4:
        raise ValueError(f"Unsupported X shape: {x.shape}")

    if x.shape[-1] == 1:
        x = np.repeat(x, 3, axis=-1)
    elif x.shape[-1] != 3:
        raise ValueError(f"Expected 1 or 3 channels, got {x.shape[-1]}")

    x = x.astype("float32")
    if x.max() > 1.0:
        x /= 255.0

    return x


# ---------------- LOAD DATA ----------------
print("Loading dataset...")

X = np.load(os.path.join(BASE_DIR, "X.npy"))
y = np.load(os.path.join(BASE_DIR, "y.npy"))
X = prepare_image_tensor(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# ---------------- BUILD MODEL ----------------
print("Building MobileNet model...")

base_model = MobileNetV2(
    input_shape=(48, 48, 3),
    include_top=False,
    weights="imagenet",
)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dense(8, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------- CALLBACKS ----------------
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.keras")

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

checkpoint = ModelCheckpoint(
    MODEL_PATH,
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1,
)

# ---------------- TRAIN ----------------
print("Training started...")

model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop, checkpoint],
)

print("\nModel training completed")
print("Model saved at:", MODEL_PATH)

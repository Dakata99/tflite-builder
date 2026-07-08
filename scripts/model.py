import os
from pathlib import Path

try:
    import tensorflow as tf
    import numpy as np
except ImportError:
    pass
import logging

REPO_ROOT: Path = Path(os.getenv("TFLITE_BUILDER_ROOT") or Path(__file__).parent)

DEFAULT_MODEL = "model.tflite"
MODELS: Path = REPO_ROOT / "models"


def generate_model() -> None:
    # 1. Build a trivial model: y = 2x + 1
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.set_weights(
        [np.array([[2.0]], dtype=np.float32), np.array([1.0], dtype=np.float32)]
    )

    # 2. Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # 3. Save it
    with open(MODELS / DEFAULT_MODEL, "wb") as f:
        f.write(tflite_model)

    logging.info("Saved %s", DEFAULT_MODEL)


def main() -> None:
    generate_model()

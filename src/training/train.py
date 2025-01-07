from typing import Any, Dict, Tuple

import tensorflow as tf
from settings.tensorflow import TensorFlowSettings
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input


def load_mnist_data() -> Tuple[Tuple[tf.Tensor, tf.Tensor], Tuple[tf.Tensor, tf.Tensor]]:
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    return (x_train, y_train), (x_test, y_test)


def create_model() -> Sequential:
    model = Sequential(
        [
            Input(shape=(28, 28)),
            Flatten(),
            Dense(128, activation="relu"),
            Dense(10, activation="softmax"),
        ]
    )

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train_model(settings: TensorFlowSettings = TensorFlowSettings()) -> Dict[str, Any]:
    try:
        (x_train, y_train), _ = load_mnist_data()
        model = create_model()

        history = model.fit(
            x_train,
            y_train,
            epochs=settings.epochs,
            batch_size=settings.batch_size,
            validation_split=settings.validation_split,
            verbose=settings.verbose,
        )

        return {"history": history.history, "model": model}
    except Exception as e:
        raise RuntimeError(f"Training failed: {str(e)}")


if __name__ == "__main__":
    result = train_model()
    print(f"Final accuracy: {result['history']['accuracy'][-1]:.4f}")
    print(f"Final loss: {result['history']['loss'][-1]:.4f}")

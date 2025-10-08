import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def create_model(input_shape=(66, 200, 3), l2_reg=0.001, dropout_rate=0.5):
    """
    NVIDIA end-to-end self-driving car model (modern TF2.0 implementation)
    """
    l2 = regularizers.l2(l2_reg)

    model = models.Sequential([
        # 1st conv layer
        layers.Conv2D(24, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2, input_shape=input_shape),
        # 2nd conv layer
        layers.Conv2D(36, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2),
        # 3rd conv layer
        layers.Conv2D(48, (5, 5), strides=(2, 2), activation='relu', kernel_regularizer=l2),
        # 4th conv layer
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2),
        # 5th conv layer
        layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2),

        # Flatten
        layers.Flatten(),

        # Fully connected layers
        layers.Dense(1164, activation='relu', kernel_regularizer=l2),
        layers.Dropout(dropout_rate),

        layers.Dense(100, activation='relu', kernel_regularizer=l2),
        layers.Dropout(dropout_rate),

        layers.Dense(50, activation='relu', kernel_regularizer=l2),
        layers.Dropout(dropout_rate),

        layers.Dense(10, activation='relu', kernel_regularizer=l2),
        layers.Dropout(dropout_rate),

        # Output: steering angle
        layers.Dense(1, activation='tanh')  # tanh gives output in range [-1, 1]
    ])

    # Multiply by constant 2.0 to match tf.atan * 2 behavior from your old model
    model.add(layers.Lambda(lambda x: x * 2.0))

    return model


if __name__ == "__main__":
    model = create_model()
    model.summary()

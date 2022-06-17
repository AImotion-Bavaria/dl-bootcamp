from tensorflow import keras
from tensorflow.keras import layers


model = keras.Sequential([
                          
            keras.Input(shape=(32, 32, 3)),
            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dense(256, activation="relu"),
            layers.Dense(128, activation="relu"),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="sigmoid"),

        ]
    )

model.summary()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

model.fit(x_train, y_train, epochs=10, batch_size=32)
print("--------------------------------")
model.evaluate(x_test, y_test)
import pickle

import numpy as np
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential


def train_cnn():
    # Load face data
    with open('data/faces_data.pkl', 'rb') as f:
        faces_data = pickle.load(f)

    # Jika data wajah disimpan sebagai list, gabungkan semua data menjadi array
    X_train = np.vstack(faces_data)  # Menggabungkan semua data wajah menjadi satu array besar
    y_train = np.array([i for i, face_set in enumerate(faces_data) for _ in range(len(face_set))])  # Membuat label berdasarkan indeks urutan wajah

    # Reshape X_train sesuai dengan input yang diharapkan oleh model CNN (misalnya (64, 64, 1))
    X_train = X_train.reshape(X_train.shape[0], 64, 64, 1)

    # Normalisasi nilai pixel antara 0-1
    X_train = X_train / 255.0

    # Build the CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(len(faces_data), activation='softmax')  # Output sesuai dengan jumlah orang
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Save the trained model
    model.save('data/face_recognition_cnn.h5')
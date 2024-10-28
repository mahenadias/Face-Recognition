import os
import pickle

import cv2
import numpy as np
import streamlit as st

from train_cnn_model import train_cnn  # Import fungsi pelatihan model


def load_existing_data():
    if os.path.exists('data/faces_data.pkl') and os.path.exists('data/users.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('data/users.pkl', 'rb') as f:
            users = pickle.load(f)
    else:
        faces_data = []  # Initialize as empty if no data found
        users = []
    return faces_data, users

def main():
    st.title("Tambah Wajah Baru ke Dataset")

    # Input user info
    name = st.text_input("Masukkan Nama:")
    user_id = st.text_input("Masukkan NIM:")
    major = st.text_input("Masukkan Prodi:")
    add_face_button = st.button("Tambah Wajah")

    if add_face_button and name and user_id and major:
        cap = cv2.VideoCapture(1)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        faces_data, users = load_existing_data()

        face_samples = []
        count = 0
        FRAME_WINDOW = st.image([])

        while count < 1000:
            ret, frame = cap.read()
            if not ret:
                st.error("Gagal membuka kamera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face = gray[y:y+h, x:x+w]
                resized_face = cv2.resize(face, (64, 64))
                face_samples.append(resized_face)
                count += 1

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{count}/1000", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            FRAME_WINDOW.image(frame, channels="BGR")

            if count >= 1000:
                break

        cap.release()
        st.success(f"1000 gambar wajah berhasil disimpan untuk {name}.")

        if not os.path.exists('data'):
            os.makedirs('data')

        # Tambahkan wajah baru ke data yang sudah ada
        if len(faces_data) > 0:
            faces_data.append(np.array(face_samples))
        else:
            faces_data = [np.array(face_samples)]  # Handle first time case

        # Tambahkan user baru ke data pengguna yang sudah ada
        users.append({'name': name, 'user_id': user_id, 'major': major})

        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)

        with open('data/users.pkl', 'wb') as f:
            pickle.dump(users, f)

        st.write("Melatih model CNN...")
        train_cnn()  # Melatih model setelah menambahkan wajah
        st.success("Model CNN berhasil dilatih dan disimpan.")

if __name__ == '__main__':
    main()
import os
import pickle
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from PIL import Image  # Untuk menampilkan gambar di Streamlit


# Fungsi untuk memuat model dan data pengguna
def load_model_and_data():
    model = load_model('data/face_recognition_cnn.h5')
    with open('data/users.pkl', 'rb') as f:
        users = pickle.load(f)
    return model, users

# Fungsi untuk mencatat kehadiran ke dalam file CSV
def record_attendance(name, user_id, major):
    file_path = 'data/attendance.csv'
    
    # Jika file tidak ada atau file kosong, buat file baru dengan header yang benar
    if not os.path.exists(file_path) or os.stat(file_path).st_size == 0:
        df = pd.DataFrame(columns=["Nama", "NIM", "Prodi", "Waktu Kehadiran"])
        df.to_csv(file_path, index=False)
    
    # Tambahkan data kehadiran baru
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_data = pd.DataFrame([{
        "Nama": name,
        "NIM": user_id,
        "Prodi": major,
        "Waktu Kehadiran": current_time
    }])
    
    df = pd.read_csv(file_path)
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_csv(file_path, index=False)
    
# Fungsi untuk mendeteksi wajah dan menilai apakah wajah valid atau tidak
def predict_face(face, model, users, threshold=0.6):  
    face = cv2.resize(face, (64, 64))  
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    
    predictions = model.predict(face)
    max_pred = np.max(predictions)

    # Jika confidence score terlalu rendah, tandai wajah sebagai tidak dikenali
    if max_pred >= threshold:
        predicted_class = np.argmax(predictions)
        predicted_name = users[predicted_class]['name']
        user_id = users[predicted_class]['user_id']
        major = users[predicted_class]['major']
        return predicted_name, user_id, major, max_pred
    else:
        return "Tidak Dikenali", None, None, max_pred

# Deteksi gerakan sebelum deteksi wajah
def detect_movement(prev_frame, current_frame, threshold=1000):
    if prev_frame is None:
        return False, current_frame
    diff = cv2.absdiff(prev_frame, current_frame)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, diff_thresh = cv2.threshold(diff_blur, 25, 255, cv2.THRESH_BINARY)
    movement_amount = np.sum(diff_thresh)
    return movement_amount > threshold, current_frame

# Fungsi utama untuk mengambil kehadiran dengan menggunakan kamera
def take_attendance():
    st.title("Sistem Absensi Pengenalan Wajah")
    st.write("Aktifkan kamera untuk melakukan absensi.")

    # Opsi threshold yang bisa diatur oleh pengguna
    threshold = st.slider("Atur Threshold Pengakuan Wajah", min_value=0.1, max_value=1.0, value=0.6, step=0.05)

    model, users = load_model_and_data()

    # Inisialisasi kamera
    cap = cv2.VideoCapture(1)
    frame_placeholder = st.empty()  # Tempatkan untuk video feed
    comment_placeholder = st.empty()  # Tempatkan untuk komentar

    # Variabel untuk mencatat kehadiran dan deteksi gerakan
    last_recorded_time = 0
    prev_frame = None
    last_detection_time = 0  # Variabel untuk mencatat kapan terakhir pengenalan dilakukan

    detected_name = None  # Variabel untuk menyimpan wajah yang terdeteksi
    comment = ""  # Variabel untuk menampilkan komentar

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membuka kamera")
            break

        current_time = time.time()

        # Proses deteksi gerakan
        movement_detected, prev_frame = detect_movement(prev_frame, frame)
        if not movement_detected:
            comment = "Tidak ada gerakan terdeteksi, mungkin ini adalah foto statis."
            comment_placeholder.write(comment)
            continue

        # Proses deteksi wajah
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            comment = "Wajah tidak dikenali."  # Keterangan jika tidak ada wajah yang terdeteksi
            detected_name = None  # Reset variabel jika wajah tidak ditemukan
        else:
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]

                # Gunakan fungsi prediksi wajah dengan threshold yang ditentukan pengguna
                name, user_id, major, confidence = predict_face(face, model, users, threshold=threshold)

                if name != "Wajah Dikenali" and movement_detected:
                    detected_name = (name, user_id, major)  # Simpan wajah yang dikenali
                    # Tampilkan kotak wajah dan identitas secara realtime
                    cv2.putText(frame, f"Nama: {name}", (x, y-60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"NIM: {user_id}", (x, y-40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.putText(frame, f"Prodi: {major}", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Kotak hijau jika wajah dikenali

                    if current_time - last_recorded_time >= 5:
                        record_attendance(name, user_id, major)
                        comment = "Wajah Dikenali dan Tercatat Oleh Sistem"
                        last_recorded_time = current_time  # Update waktu terakhir pencatatan
                else:
                    # Jika wajah tidak dikenali
                    cv2.putText(frame, "Wajah Tidak Dikenali", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Kotak merah jika wajah tidak dikenali
                    comment = "Wajah tidak dikenali."

        # Tampilkan komentar yang sinkron dengan pencatatan kehadiran
        comment_placeholder.write(comment)

        # Konversi frame ke format RGB untuk Streamlit dan tampilkan di localhost
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(img_pil, use_column_width=True)

        # Tambahkan jeda untuk mengurangi beban memori
        time.sleep(0.1)

    cap.release()

# Fungsi utama untuk Streamlit
def main():
    take_attendance()


if __name__ == "__main__":
    main()
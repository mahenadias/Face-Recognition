import streamlit as st

# Sidebar untuk navigasi
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Add New Face", "Take Attendance", "Show Attendance"])

if option == "Home":
    st.title("Sistem Absensi Pengenalan Wajah")
    st.write("Selamat datang! Silakan gunakan sidebar untuk menavigasi.")

elif option == "Add New Face":
    import add_faces_streamlit
    add_faces_streamlit.main()

elif option == "Take Attendance":
    import take_attendance_cnn
    take_attendance_cnn.main()

elif option == "Show Attendance":
    import show_attendance
    show_attendance.main()
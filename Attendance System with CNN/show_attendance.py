import os

import pandas as pd
import streamlit as st


def main():
    st.title("Daftar Kehadiran")

    # Cek apakah file kehadiran ada
    attendance_file = "data/attendance.csv"
    if os.path.exists(attendance_file):
        # Tampilkan daftar kehadiran
        df = pd.read_csv(attendance_file)
        st.write(df)
    else:
        st.warning("Belum ada data kehadiran yang tercatat.")

if __name__ == "__main__":
    main()
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Memuat model dan encoder
try:
    model = joblib.load('best_model.pkl')
    label_encoder_model = joblib.load('label_encoder_model.pkl')
    label_encoder_wilayah = joblib.load('label_encoder_wilayah.pkl')
    label_encoder_merk = joblib.load('label_encoder_merk.pkl')
    unique_values = joblib.load('unique_values.pkl')

    # Nilai unik untuk input
    unique_models = unique_values['unique_models']
    unique_wilayah = unique_values['unique_wilayah']
    unique_merk = unique_values['unique_merk']

    # Fit encoders
    label_encoder_model.fit(unique_models)
    label_encoder_wilayah.fit(unique_wilayah)
    label_encoder_merk.fit(unique_merk)
except Exception as e:
    st.error(f"Error saat memuat model atau encoder: {str(e)}")

# Fungsi prediksi
def predict(data):
    prediction = model.predict(data)
    return prediction

# Tab navigasi
menu = st.sidebar.radio("Navigasi", ["Home", "Dataset", "Visualisasi", "Prediksi"])

# Home Page
if menu == "Home":
    st.title("Website Prediksi Harga Mobil")
    st.write("""
    Selamat datang di aplikasi prediksi harga mobil! 
    Aplikasi ini bertujuan untuk memberikan prediksi harga mobil berdasarkan data spesifikasi kendaraan.
    """)
    st.header("Informasi")
    st.write("- **Nama Pembuat**: Irsyad Thariq Hafizh")
    st.write("- **Sumber Dataset**: https://www.kaggle.com/datasets/kelompok8ai/data-mobil-bekas")
    st.write("- **Algoritma yang Digunakan**: Random Forest")
    st.write("- **Fitur yang Digunakan**: Model, Wilayah, Merk, Transmisi, Bahan Bakar, Jarak Tempuh, Kapasitas Mesin, Tahun Produksi.")

# Dataset Page
elif menu == "Dataset":
    st.title("Dataset")
    st.write("Di bawah ini adalah tampilan dataset yang digunakan dalam pembuatan model prediksi:")
    dataset_path = "dataset.csv"  # Ganti dengan path dataset Anda
    try:
        df = pd.read_csv(dataset_path)
        st.dataframe(df)
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan file dataset tersedia.")

# Visualisasi Page
elif menu == "Visualisasi":
    st.title("Visualisasi Data")
    st.write("Visualisasi dataset untuk memahami pola dan distribusi data.")
    dataset_path = "dataset.csv"  # Ganti dengan path dataset Anda
    try:
        df = pd.read_csv(dataset_path)

        # Visualisasi distribusi tahun produksi
        st.subheader("Distribusi Tahun Produksi")
        if 'tahun_produksi' in df.columns:
            fig, ax = plt.subplots()
            df['tahun_produksi'].hist(ax=ax, bins=20)
            ax.set_title("Distribusi Tahun Produksi")
            ax.set_xlabel("Tahun Produksi")
            ax.set_ylabel("Frekuensi")
            st.pyplot(fig)
        else:
            st.warning("Kolom 'tahun_produksi' tidak ditemukan dalam dataset.")

        # Visualisasi harga berdasarkan merk
        st.subheader("Rata-rata Harga Berdasarkan Merk")
        if 'merk_mobil' in df.columns and 'harga' in df.columns:
            avg_price_per_merk_mobil = df.groupby('merk_mobil')['harga'].mean().sort_values()
            fig, ax = plt.subplots()
            avg_price_per_merk_mobil.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title("Rata-rata Harga Berdasarkan merk_mobil")
            ax.set_ylabel("Harga Rata-rata")
            st.pyplot(fig)
        else:
            st.warning("Kolom 'merk' atau 'harga' tidak ditemukan dalam dataset.")
    except FileNotFoundError:
        st.error("Dataset tidak ditemukan. Pastikan file dataset tersedia.")

# Prediksi Page
elif menu == "Prediksi":
    st.title("Prediksi Harga Mobil")
    st.write("Masukkan data untuk melakukan prediksi harga mobil:")
    
    # Input data
    jarak_tempuh = st.number_input('Jarak Tempuh (dalam ribuan km)', value=0.0)
    kapasitas_mesin = st.number_input('Kapasitas Mesin (dalam liter)', value=0.0)
    tahun_produksi = st.number_input('Tahun Produksi', value=0.0)
    transmisi = st.selectbox('Transmisi', options=['Automatic', 'Manual'])
    bahan_bakar = st.selectbox('Bahan Bakar', options=['Bensin', 'Diesel'])
    model_mobil = st.selectbox('Model Mobil', options=unique_models)
    wilayah = st.selectbox('Wilayah', options=unique_wilayah)
    merk_mobil = st.selectbox('Merk Mobil', options=unique_merk)

    # Konversi fitur kategorikal
    transmisi = 0 if transmisi == 'Automatic' else 1
    bahan_bakar = 0 if bahan_bakar == 'Bensin' else 1
    model_mobil_encoded = label_encoder_model.transform([model_mobil])[0]
    wilayah_encoded = label_encoder_wilayah.transform([wilayah])[0]
    merk_mobil_encoded = label_encoder_merk.transform([merk_mobil])[0]

    # Membuat DataFrame untuk prediksi
    data = pd.DataFrame({
        'model_mobil': [model_mobil_encoded],
        'wilayah': [wilayah_encoded],
        'merk_mobil': [merk_mobil_encoded],
        'transmisi': [transmisi],
        'bahan_bakar': [bahan_bakar],
        'jarak_tempuh': [jarak_tempuh],
        'kapasitas_mesin': [kapasitas_mesin],
        'tahun_produksi': [tahun_produksi]
    })

    # Tombol prediksi
    if st.button('Prediksi'):
        try:
            result = predict(data)
            st.success(f"Harga Prediksi: Rp {result[0]:,.2f}")
        except Exception as e:
            st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")

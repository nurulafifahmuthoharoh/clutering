import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

# Load datasets
file_path = 'datasets.xlsx'
prepared_data_path = 'prepared_data.csv'
clustered_data_path = 'clustered_data.csv'

# Load raw and processed data
data_raw = pd.read_excel(file_path, sheet_name='CORRECTIVE MAINTANCE')
data_prepared = pd.read_csv(prepared_data_path)
data_clustered = pd.read_csv(clustered_data_path)

# Load encoders and scaler
le_gangguan = joblib.load('le_gangguan.pkl')
le_penyebab = joblib.load('le_penyebab.pkl')
le_zona = joblib.load('le_zona.pkl')
le_modem = joblib.load('le_modem.pkl')
scaler = joblib.load('scaler.pkl')

# Set page config
st.set_page_config(page_title="PLN Gangguan Dashboard", layout="wide")

# Dashboard Title
st.markdown("# **PLN Gangguan Dashboard**")
st.markdown("### Mengelola dan Menganalisis Data Gangguan dengan Visualisasi yang Informatif")
st.markdown("---")

# Sidebar Navigation
st.sidebar.title("Navigasi")
menu = st.sidebar.radio(
    "Pilih Menu:",
    ["Data Exploration", "Data Preparation", "Hasil Clustering", "Analisis Cluster", "Prediksi"]
)

# Menu: Data Exploration
if menu == "Data Exploration":
    st.markdown("## Data Exploration")

    # 1. Missing Values
    st.markdown("### Missing Values")
    missing_values = data_raw.isnull().sum()
    st.dataframe(missing_values)

    # 2. Distribusi Kolom Utama
    st.markdown("### Distribusi Kolom Utama")
    for col in ['Jenis Gangguan', 'Penyebab', 'ZONA', 'SITE', 'SOLUSI', 'Merk Modem']:
        st.write(f"**Distribusi {col}:**")
        st.write(data_raw[col].value_counts().head(10))  # Top 10 values

    # 3. Visualisasi Distribusi
    st.markdown("### Visualisasi Distribusi Kolom Utama")
    for col in ['Jenis Gangguan', 'Penyebab', 'ZONA', 'SITE', 'SOLUSI', 'Merk Modem']:
        st.write(f"**Distribusi {col}:**")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y=col, data=data_raw, order=data_raw[col].value_counts().index[:10], ax=ax)  # Top 10
        plt.title(f'Distribusi {col}')
        plt.xlabel('Jumlah')
        plt.ylabel(col)
        st.pyplot(fig)

# Menu: Data Preparation
elif menu == "Data Preparation":
    st.markdown("## Data Preparation")

    st.write("**Preview of Prepared Data**")
    if st.checkbox("Tampilkan Semua Data"):
        st.dataframe(data_prepared)
    else:
        st.dataframe(data_prepared.head())

    st.write("**Details of Encoded and Scaled Features:**")
    st.markdown("- Fitur yang di-encode: ['Jenis Gangguan', 'Penyebab', 'ZONA', 'Merk Modem']")
    st.markdown("- Data dinormalisasi menggunakan StandardScaler.")

# Menu: Hasil Clustering
elif menu == "Hasil Clustering":
    st.markdown("## Hasil Clustering")

    st.write("**Preview of Clustered Data**")
    if st.checkbox("Tampilkan Semua Data Clustered"):
        st.dataframe(data_clustered)
    else:
        st.dataframe(data_clustered.head())

    # Visualize Clustering
    st.write("**Clustering Results Visualization**")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x=data_clustered['Jenis Gangguan'], 
        y=data_clustered['Penyebab'], 
        hue=data_clustered['Cluster'], 
        palette='viridis', 
        s=100, 
        ax=ax
    )
    plt.title('Clustering Results')
    plt.xlabel('Jenis Gangguan')
    plt.ylabel('Penyebab')
    st.pyplot(fig)

# Menu: Analisis Cluster
elif menu == "Analisis Cluster":
    st.markdown("## Analisis Cluster")

    st.write("**Statistik per Cluster**")
    cluster_summary = data_clustered.groupby('Cluster').agg(
        count=('Cluster', 'size'),
        avg_jenis_gangguan=('Jenis Gangguan', 'mean'),
        avg_penyebab=('Penyebab', 'mean'),
        avg_zona=('ZONA', 'mean')
    ).reset_index()
    st.dataframe(cluster_summary)

    
    st.write("**Solusi yang Sering Digunakan per Cluster**")
    if 'SOLUSI' in data_clustered.columns:
        solusi_summary = data_clustered.groupby('Cluster')['SOLUSI'].apply(lambda x: x.value_counts().idxmax()).reset_index()
        solusi_summary.columns = ['Cluster', 'Solusi Paling Umum']
        st.dataframe(solusi_summary)
    else:
        st.write("Kolom 'SOLUSI' tidak tersedia dalam data.")

    st.write("**Distribusi Cluster berdasarkan Zona**")
    if 'ZONA' in data_clustered.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(x='Cluster', hue='ZONA', data=data_clustered, palette='viridis', ax=ax)
        plt.title('Distribusi Cluster berdasarkan Zona')
        st.pyplot(fig)
    else:
        st.write("Kolom 'ZONA' tidak tersedia dalam data.")

# Menu: Prediksi
elif menu == "Prediksi":
    st.markdown("## Prediksi Data Baru")

    st.write("Masukkan data baru untuk prediksi cluster:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        jenis_gangguan = st.selectbox("Jenis Gangguan", le_gangguan.classes_)
    with col2:
        penyebab = st.selectbox("Penyebab", le_penyebab.classes_)
    with col3:
        zona = st.selectbox("ZONA", le_zona.classes_)
    with col4:
        merk_modem = st.selectbox("Merk Modem", le_modem.classes_)

    # Convert opsi yang dipilih ke nilai numerik menggunakan LabelEncoder
    new_data = [
        le_gangguan.transform([jenis_gangguan])[0],
        le_penyebab.transform([penyebab])[0],
        le_zona.transform([zona])[0],
        le_modem.transform([merk_modem])[0]
    ]

    # Scale the data
    new_data_scaled = scaler.transform([new_data])

    # Load the clustering model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(data_prepared[['Jenis Gangguan', 'Penyebab', 'ZONA', 'Merk Modem']])

    # Prediksi cluster untuk data baru
    if st.button("Prediksi Cluster"):
        try:
            cluster = kmeans.predict(new_data_scaled)[0]
            st.success(f"Data baru termasuk ke dalam Cluster {cluster}")
            st.markdown("---")

            # Rekomendasi Berdasarkan Cluster
            st.write("**Rekomendasi:**")
            if cluster == 0:
                st.write("Prioritaskan pemeliharaan di zona dengan frekuensi gangguan tertinggi.")
            elif cluster == 1:
                st.write("Periksa penyebab terkait perangkat tertentu untuk pencegahan.")
            else:
                st.write("Lakukan inspeksi rutin untuk mengurangi kemungkinan gangguan.")

            # Solusi Berdasarkan Cluster
            st.markdown("**Solusi yang Disarankan:**")
            if 'SOLUSI' in data_clustered.columns:  # Sesuaikan nama kolom jika menggunakan huruf besar
                solusi_cluster = data_clustered[data_clustered['Cluster'] == cluster]['SOLUSI'].value_counts().idxmax()
                st.write(f"Solusi yang sering digunakan untuk gangguan di Cluster {cluster}: **{solusi_cluster}**")
            else:
                st.write("Kolom 'SOLUSI' tidak tersedia dalam data.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}. Pastikan semua input valid.")

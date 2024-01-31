# Manual Prototype System Klasifikasi Genus Ikan Menggunakan Viola-Jones Feature Extraction dan Boosting Berbasis Decision Tree

### Nama: Nehemiah Austen Pison
### NIM	: 1313619021

# TRAINING 
1. masukan semua gambar training ke foldernya
masing-maasing di /fish_dataset. Tambahkan juga satu folder dan isi dengan gambar negatif (gambar hitam semuanya)
2a. Sesuaikan nama kelas dan folder pada LoadImages.py
2b. image = cv2.resize(image, (350, 200)), sesuaikan (350, 200) ke resolusi yang diinginkan. (350, 200) adalah 350 x 200 piksel
3. Sesuaikan offset tiap kelas pada Dataset.py
4. Pastikan folder /Data sudah dikosongkan dari file-file pickle dan CSV agar tidak terjadi overwrite
5a. Buka __init__.py sesuaikan variabel "csv_name" sesuai kebutuhan
5b. sesuaikan initial_features = generate_features(50, 50) ke ukuran window yang ingin digunakan (50, 50) adalah window 50x50 piksel
6. Run __init__.py dan tunggu hingga cascade setiap window sudah di-generate pada folder /Data

# USE 
1. Pastikan cascade yang sesuai sudah ada di dalam /Data
2. Masukan gambar yang ingin diklasifikasi ke folder /classification_target
3. Sesuaikan teks anotasi pada predict.py sesuai kelas yang akan diklasifikasi
4. Run Predict.py
5. Hasil klasifikasi yang sudah dianotasi akan keluar di dalam /classification_results
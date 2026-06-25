# Wine Quality Classification

Sistem klasifikasi kualitas wine (**Good** vs **Bad**) berbasis *classical machine learning* —
dari 11 sifat fisikokimia (kadar alkohol, keasaman, densitas, dll.) — dengan perbandingan
beberapa classifier (**Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost**),
disajikan sebagai aplikasi web interaktif berbasis **Streamlit**.

Final Project — **Machine Learning (COMP6577001)**, Kelompok 4.

## Anggota
- Michael Peterson — 2802451541
- Jos Hwen Sim — 2802451655
- Jonathan Kuniardi — 2802461832

## Struktur Repositori
- **`app_v6.py`** — aplikasi utama (versi terbaru). Single-file Streamlit yang memuat: CSS desain,
  `load_data()` (fetch UCI + cache), navigasi sidebar, serta halaman **Home, EDA, Description,
  Data Preprocessing, Train Your Model, Result / Prediction Demo, Feature Importance, About Us**.
  Pipeline interaktif: preprocessing → training → prediksi yang mengalir lewat `st.session_state`.
- **`requirements.txt`** — dependensi Python (Streamlit, scikit-learn, pandas, numpy, matplotlib, seaborn, ucimlrepo, xgboost).
- **`Perbaiki desain website/`** — referensi desain (React + Vite + Tailwind, hasil ekspor Figma) yang menjadi acuan visual aplikasi.
- **`app.py`, `app_v2.py`–`app_v5.py`** — versi lama, disimpan sebagai histori pengembangan.

## Dataset
[Wine Quality Data Set](https://archive.ics.uci.edu/dataset/186/wine+quality)
(UCI Machine Learning Repository, ID 186) — Vinho Verde, Portugal; 6.497 sampel, 11 fitur fisikokimia.
Data diambil otomatis saat runtime via `ucimlrepo` (`fetch_ucirepo(id=186)`) dan di-*cache* dengan
`@st.cache_data`, sehingga butuh koneksi internet saat pertama dijalankan. Target skor kualitas 0–10
dibinarisasi menjadi **Good (≥ 7)** vs **Bad (< 7)**.

## Menjalankan Aplikasi
```bash
pip install -r requirements.txt
streamlit run app_v6.py
```

## Hasil
Pada data yang diseimbangkan (undersampling 50:50), **Random Forest** memberikan performa terbaik —
akurasi **≈ 80%**, **F1-Score 0,81**, dan **ROC-AUC 0,88** — dengan *recall* kelas Good mencapai 0,86.
Latency inferensi < 100 ms, memenuhi syarat sistem yang dapat dipakai secara real-time.

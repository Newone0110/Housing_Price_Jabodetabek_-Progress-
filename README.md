# Prediksi Harga Rumah Jabodetabek

Model untuk menaksir harga rumah di Jabodetabek dari luas tanah, luas bangunan, jumlah kamar, lokasi, dan beberapa atribut lain. Datanya iklan rumah123.com tahun 2022 (Kaggle, nafisbarizki), 3.553 listing.

> Demo: menyusul setelah deploy

## Hasil singkat

Model akhir XGBoost. Di data test (639 rumah yang tidak dipakai sama sekali selama pemilihan model):

| Metrik | Nilai |
|---|---|
| Rata-rata meleset (MAPE) | 20,3% |
| Median meleset | 13,3% |
| Rumah dengan tebakan meleset ≤ 20% | 65,9% |

Hasil test mirip dengan cross-validation (21,9%), jadi model tidak sekadar menghafal data latih.

## Yang saya pelajari dari proyek ini

**Seperempat data test ternyata "bocoran".** Banyak rumah yang sama diiklankan ulang oleh agen berbeda dengan judul lain, tapi semua spesifikasi dan harganya identik. Duplikat biasa tidak menangkapnya. Kalau dibiarkan, hampir seperempat rumah di test punya "kembaran" di train (kecamatan, harga, luas, dan jumlah kamar sama), sehingga skor model jadi terlalu optimis. Saya buang 251 iklan ulang yang identik, lalu split dan cross-validation dilakukan per grup (`GroupShuffleSplit`, `GroupKFold`) supaya rumah yang mirip kembar tidak terpisah ke train dan test.

**Model yang rumit tidak otomatis menang.** Dengan setelan default, XGBoost, LightGBM, dan Random Forest semuanya di kisaran 24–25% MAPE, hampir sama dengan Ridge biasa (24,6%). Tuning menurunkannya ke ±22%. Artinya yang paling membatasi di sini adalah informasi di data, bukan algoritmanya.

**Bentuk fitur penting untuk model linear.** Ridge dengan luas tanah/bangunan apa adanya menghasilkan tebakan yang meledak (R² −1.663). Setelah fitur yang miring di-log, MAPE-nya turun ke 24,6%. Masuk akal: harga rumah naik secara persentase terhadap luas, bukan bertambah rupiah tetap per m².

**Nilai kosong ternyata membawa informasi.** Rumah tanpa keterangan sertifikat punya median harga hampir 2× lipat rumah yang terisi. Karena itu nilai kosong tidak diisi modus, tetapi diberi label sendiri ("tidak diketahui").

**Rata-rata error menyembunyikan cerita.** Di Bogor, MAPE 21% tapi median melesetnya 11%, paling akurat dari semua kota. Kebanyakan rumah ditebak dengan baik, tapi ada beberapa yang meleset lebih dari 100%. Kota tersulit adalah Jakarta Selatan dan Jakarta Pusat. Untuk Jakarta Pusat, datanya juga terlalu sedikit (8 rumah di test).

**Yang paling menentukan harga adalah luas bangunan dan luas tanah** (berdasarkan SHAP), jauh di atas fitur lain. Lokasi menyusul: rumah di Bogor, misalnya, ditaksir sekitar 14% lebih murah dibanding rumah sejenis di tempat lain.

## Alur kerja

| Notebook | Isi |
|---|---|
| `01_Data_Exploration` | Mencari masalah di data mentah |
| `02_Cleaning` | Aturan pembersihan; 3.553 → 3.172 baris |
| `03_EDA` | Distribusi harga, korelasi, pengaruh lokasi, pola nilai kosong |
| `04_Feature_Engineering` | Split per grup, isi nilai kosong dari train saja, one-hot encoding |
| `05_Modelling` | Baseline, model pohon, tuning (RandomizedSearchCV + GroupKFold) |
| `06_Evaluation` | Analisis error dan SHAP |

Beberapa keputusan di cleaning:
- Kos, hotel, guest house, oper kredit, dan lelang dibuang karena harganya bukan harga rumah tinggal di pasar.
- Satu listing seharga 580 miliar dibuang. Rumah dengan cluster dan ukuran yang persis sama diiklankan 5,8 miliar, jadi hampir pasti kelebihan dua nol.
- `lat`/`long` tidak dipakai. Ternyata isinya hanya titik tengah kecamatan, dan sebagian bahkan jatuh di Cirebon dan Jawa Tengah.

Target dilatih dalam bentuk log karena harganya sangat miring (median 1,56 M, rata-rata 4,08 M).

## Keterbatasan

- Harga yang dipakai adalah harga iklan 2022, bukan harga transaksi.
- Lokasi hanya sampai level kecamatan. Dua rumah di kecamatan yang sama bisa beda jauh harganya karena jalan, akses, atau lingkungan, dan informasi itu tidak ada di data.
- Model pohon tidak bisa menaksir di atas harga tertinggi yang pernah dilihatnya, jadi rumah super mewah cenderung ditaksir terlalu rendah.
- Sebagian rumah meleset lebih dari 100%, dan penyebabnya (data iklan yang salah atau model yang salah) belum diselidiki satu per satu.

## Menjalankan

```bash
conda create -n rumah-jaboretabek python=3.11
conda activate rumah-jaboretabek
python -m pip install pandas numpy scikit-learn xgboost lightgbm shap seaborn matplotlib joblib jupyter
```

Jalankan notebook berurutan dari `01` sampai `06`. Data mentah ada di `data/raw/`, hasil olahan disimpan ke `data/processed/`, dan model ke `models/`.

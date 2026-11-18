import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# ---------- lokasi file ----------
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "model_xgb.joblib"
INFO_PATH = ROOT / "models" / "preprocess_info.joblib"
DATA_PATH = ROOT / "data" / "processed" / "rumah_clean.csv"

MEDIAN_MELESET = 0.146   # median kesalahan model di data test

st.set_page_config(page_title="Taksiran Harga Rumah Jabodetabek", page_icon="🏠")


# ---------- muat model, aturan preprocessing, dan pilihan dropdown ----------
@st.cache_resource
def muat_model():
    model = joblib.load(MODEL_PATH)
    info = joblib.load(INFO_PATH)
    explainer = shap.TreeExplainer(model)
    return model, info, explainer


@st.cache_data
def muat_pilihan():
    df = pd.read_csv(DATA_PATH)
    df["certificate"] = df["certificate"].replace(
        {"hp - hak pakai": "lainnya (ppjb,girik,adat,dll)"})
    return df


model, info, explainer = muat_model()
data = muat_pilihan()


def norm(teks):
    """samakan penulisan nama kolom: huruf kecil, spasi/simbol jadi _"""
    return re.sub(r"[^0-9a-z]+", "_", str(teks).lower()).strip("_")


PETA_KOLOM = {norm(c): c for c in info["kolom"]}


def siapkan(numerik, kategori):
    """ubah input pengguna jadi 1 baris dengan kolom yang sama persis seperti saat training"""
    x = pd.DataFrame(0.0, index=[0], columns=info["kolom"])

    # kolom angka + penanda kosong
    for col, nilai in numerik.items():
        kosong = nilai is None
        if kosong:
            nilai = info["nilai_isi"].get(col, 0)
        if col in x.columns:
            x[col] = float(nilai)
        if f"{col}_kosong" in x.columns:
            x[f"{col}_kosong"] = int(kosong)

    # district yang jarang -> "lainnya"
    if kategori["district"] not in info["district_ramai"]:
        kategori["district"] = "lainnya"

    # kolom kategori -> one-hot
    for col, nilai in kategori.items():
        kunci = norm(f"{col}_{nilai}")
        if kunci in PETA_KOLOM:
            x[PETA_KOLOM[kunci]] = 1.0

    return x


def opsi(kolom):
    return sorted(data[kolom].dropna().unique().tolist()) + ["tidak diketahui"]


def rupiah(angka):
    if angka >= 1e9:
        return f"Rp {angka / 1e9:,.2f} miliar"
    return f"Rp {angka / 1e6:,.0f} juta"


# ---------- tampilan ----------
st.title("🏠 Taksiran Harga Rumah Jabodetabek")
st.caption("Model XGBoost dari ±3.000 iklan rumah123 tahun 2022. "
           "Hasilnya taksiran harga iklan, bukan harga transaksi.")

st.subheader("Lokasi")
c1, c2 = st.columns(2)
kota = c1.selectbox("Kota", sorted(data["city"].unique()))
kecamatan = c2.selectbox("Kecamatan",
                         sorted(data.loc[data["city"] == kota, "district"].unique()))

st.subheader("Ukuran")
c1, c2 = st.columns(2)
luas_tanah = c1.number_input("Luas tanah (m²)", 10, 10000, 120)
luas_bangunan = c2.number_input("Luas bangunan (m²)", 10, 10000, 100)

c1, c2, c3 = st.columns(3)
kamar_tidur = c1.number_input("Kamar tidur", 1, 20, 3)
kamar_mandi = c2.number_input("Kamar mandi", 1, 20, 2)
lantai = c3.number_input("Jumlah lantai", 1, 5, 2)

c1, c2, c3, c4 = st.columns(4)
carport = c1.number_input("Carport", 0, 20, 1)
garasi = c2.number_input("Garasi", 0, 20, 0)
kt_art = c3.number_input("Kamar ART", 0, 10, 0)
km_art = c4.number_input("Kamar mandi ART", 0, 10, 0)

st.subheader("Detail lain")
c1, c2 = st.columns(2)
tahu_umur = c1.checkbox("Umur bangunan diketahui", value=True)
umur = c1.number_input("Umur bangunan (tahun)", 0, 100, 5, disabled=not tahu_umur)
tahu_listrik = c2.checkbox("Daya listrik diketahui", value=True)
listrik = c2.selectbox("Daya listrik (VA)", [900, 1300, 2200, 3500, 4400, 5500, 7700, 11000],
                       index=2, disabled=not tahu_listrik)

c1, c2 = st.columns(2)
sertifikat = c1.selectbox("Sertifikat", opsi("certificate"))
kondisi = c2.selectbox("Kondisi", opsi("property_condition"))
c1, c2 = st.columns(2)
perabot = c1.selectbox("Perabot", opsi("furnishing"))
arah = c2.selectbox("Arah bangunan", opsi("building_orientation"))

if st.button("Taksir harga", type="primary"):
    numerik = {
        "land_size_m2": luas_tanah, "building_size_m2": luas_bangunan,
        "bedrooms": kamar_tidur, "bathrooms": kamar_mandi, "floors": lantai,
        "carports": carport, "garages": garasi,
        "maid_bedrooms": kt_art, "maid_bathrooms": km_art,
        "building_age": umur if tahu_umur else None,
        "listrik": listrik if tahu_listrik else None,
    }
    kategori = {
        "city": kota, "district": kecamatan, "certificate": sertifikat,
        "property_condition": kondisi, "furnishing": perabot,
        "building_orientation": arah,
    }

    x = siapkan(numerik, kategori)
    harga = float(np.expm1(model.predict(x))[0])

    st.metric("Taksiran harga", rupiah(harga))
    st.write(f"Rentang wajar: **{rupiah(harga * (1 - MEDIAN_MELESET))} – "
             f"{rupiah(harga * (1 + MEDIAN_MELESET))}** "
             f"(berdasarkan median kesalahan model ±{MEDIAN_MELESET:.0%}).")

    if kategori["district"] == "lainnya":
        st.info("Kecamatan ini jarang muncul di data, jadi model hanya memakai informasi kotanya. "
                "Taksiran bisa kurang tepat.")
    if luas_tanah > 1000:
        st.warning("Rumah dengan tanah sangat luas sering ditaksir terlalu mahal oleh model.")

    st.subheader("Kenapa harganya segitu?")
    st.caption("Merah = menaikkan harga, biru = menurunkan. Skala log harga.")
    sv = explainer(x)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], max_display=10, show=False)
    st.pyplot(fig)
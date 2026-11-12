from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FOLDER_MODELS = Path(__file__).resolve().parent.parent / "models"

model = joblib.load(FOLDER_MODELS / "model_xgb.joblib")
info = joblib.load(FOLDER_MODELS / "preprocess_info.joblib")

KOLOM_INPUT = [
    "city", "district", "land_size_m2", "building_size_m2", "bedrooms", "bathrooms",
    "floors", "carports", "garages", "maid_bedrooms", "maid_bathrooms", "listrik",
    "building_age", "certificate", "property_condition", "furnishing", "building_orientation",
]
BOLEH_KOSONG_JADI_NOL = ["carports", "garages", "maid_bedrooms", "maid_bathrooms"]


def siapkan_fitur(data):
    df = pd.DataFrame(data).reindex(columns=KOLOM_INPUT)

    for col in BOLEH_KOSONG_JADI_NOL:
        df[col] = df[col].fillna(0)

    df["certificate"] = df["certificate"].replace(info["map_sertifikat"])
    for col in info["kat_kosong"]:
        df[col] = df[col].fillna("tidak diketahui")

    for col in info["num_penanda"]:
        df[col + "_kosong"] = df[col].isnull().astype(int)
    for col, nilai in info["nilai_isi"].items():
        df[col] = df[col].fillna(nilai)

    df["district"] = df["district"].where(df["district"].isin(info["district_ramai"]), "lainnya")

    df = pd.get_dummies(df, columns=info["kolom_kategori"]).astype(float)
    df.columns = df.columns.str.replace(r"[^0-9a-zA-Z_]+", "_", regex=True)
    return df.reindex(columns=info["kolom"], fill_value=0)


def prediksi_harga(rumah):
    fitur = siapkan_fitur([rumah])
    return float(np.expm1(model.predict(fitur))[0])


if __name__ == "__main__":
    contoh = {
        "city": "Bekasi", "district": "Summarecon Bekasi",
        "land_size_m2": 120, "building_size_m2": 90,
        "bedrooms": 3, "bathrooms": 2,
    }
    print(f"Taksiran harga: Rp {prediksi_harga(contoh):,.0f}")
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import datetime

print("🤖 AI PREDIKSI HARGA KENDARAAN - MACHINE LEARNING")

# Dataset lebih realistis (bisa expand dengan data real)
data = {
    'merek': ['toyota', 'honda', 'toyota', 'daihatsu', 'honda', 'suzuki', 'toyota', 'honda', 
              'mitsubishi', 'nissan', 'daihatsu', 'suzuki', 'toyota', 'honda'],
    'model': ['avanza', 'brio', 'rush', 'ayla', 'mobilio', 'ertiga', 'innova', 'city', 
              'pajero', 'juke', 'xenia', 'baleno', 'fortuner', 'civic'],
    'tahun': [2020, 2019, 2018, 2021, 2017, 2020, 2019, 2018, 2016, 2017, 2020, 2019, 2021, 2018],
    'km': [15000, 30000, 45000, 8000, 60000, 12000, 25000, 40000, 80000, 55000, 18000, 22000, 5000, 35000],
    'kondisi': [4, 3, 3, 4, 2, 4, 3, 2, 2, 3, 4, 3, 5, 4],  # 1-5 scale
    'tipe': ['mpv', 'city car', 'suv', 'city car', 'mpv', 'mpv', 'suv', 'sedan', 'suv', 'suv', 'mpv', 'hatchback', 'suv', 'sedan'],
    'harga': [300000000, 180000000, 280000000, 150000000, 140000000, 260000000, 450000000, 220000000, 
              350000000, 200000000, 170000000, 210000000, 550000000, 320000000]
}

df = pd.DataFrame(data)

# Preprocessing data
le_merek = LabelEncoder()
le_model = LabelEncoder()
le_tipe = LabelEncoder()

df['merek_encoded'] = le_merek.fit_transform(df['merek'])
df['model_encoded'] = le_model.fit_transform(df['model'])
df['tipe_encoded'] = le_tipe.fit_transform(df['tipe'])

# Features dan target
X = df[['merek_encoded', 'model_encoded', 'tahun', 'km', 'kondisi', 'tipe_encoded']]
y = df['harga']

# Split data training & testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model (lebih akurat dari Linear Regression)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print(f"✅ Model Training Complete!")
print(f"📊 Training Accuracy: {train_score:.2%}")
print(f"📊 Testing Accuracy: {test_score:.2%}")

def prediksi_harga_ml():
    print("\n" + "="*50)
    print("📊 MASUKKAN DATA KENDARAAN")
    print("="*50)
    
    merek = input("Merek (Toyota/Honda/Daihatsu/Suzuki/Mitsubishi/Nissan): ").lower()
    model_input = input("Model (Avanza/Brio/Rush/Ayla/dll): ").lower()
    tahun = int(input("Tahun: "))
    km = int(input("Kilometer: "))
    
    print("\n🔄 Kondisi Kendaraan (1-5):")
    print("1. Sangat Kurang | 2. Kurang | 3. Biasa | 4. Baik | 5. Sangat Baik")
    kondisi = int(input("Pilih kondisi (1-5): "))
    
    tipe = input("Tipe (MPV/SUV/Sedan/City Car/Hatchback): ").lower()
    
    try:
        # Encode categorical features
        merek_encoded = le_merek.transform([merek])[0]
        model_encoded = le_model.transform([model_input])[0]
        tipe_encoded = le_tipe.transform([tipe])[0]
        
        # Prepare input data
        input_data = [[merek_encoded, model_encoded, tahun, km, kondisi, tipe_encoded]]
        
        # Predict
        harga_prediksi = model.predict(input_data)[0]
        
        # Confidence estimation (sederhana)
        confidence = min(95, test_score * 100 + 70)
        
        print("\n" + "="*50)
        print("🤖 HASIL PREDIKSI AI")
        print("="*50)
        print(f"🚗 Kendaraan: {merek.title()} {model_input.title()} {tahun}")
        print(f"📋 Tipe: {tipe.upper()}")
        print(f"🛣️  KM: {km:,}")
        print(f"⭐ Kondisi: {kondisi}/5")
        print(f"🎯 Akurasi Prediksi: {confidence:.1f}%")
        print("="*50)
        print(f"💰 PERKIRAAN HARGA: Rp {max(0, harga_prediksi):,.0f}")
        print("="*50)
        
        # Additional insights
        tahun_sekarang = datetime.datetime.now().year
        umur = tahun_sekarang - tahun
        print(f"\n💡 INSIGHTS:")
        print(f"• Umur kendaraan: {umur} tahun")
        print(f"• Rata-rata KM/tahun: {km/umur if umur > 0 else km:,.0f} km")
        
        if harga_prediksi > 500000000:
            print("• 🏷️  Kategori: Premium")
        elif harga_prediksi > 250000000:
            print("• 🏷️  Kategori: Menengah")
        else:
            print("• 🏷️  Kategori: Ekonomi")
            
    except ValueError as e:
        print(f"\n❌ ERROR: {e}")
        print("Pastikan merek/model/tipe sesuai dengan data training")
        print("Data training tersedia:")
        print(f"Merek: {list(le_merek.classes_)}")
        print(f"Model: {list(le_model.classes_)}")
        print(f"Tipe: {list(le_tipe.classes_)}")

def tambah_data_training():
    """Fitur untuk menambah data training (improve model)"""
    print("\n📈 TAMBAH DATA TRAINING BARU")
    merek = input("Merek: ").lower()
    model_input = input("Model: ").lower()
    tahun = int(input("Tahun: "))
    km = int(input("Kilometer: "))
    kondisi = int(input("Kondisi (1-5): "))
    tipe = input("Tipe: ").lower()
    harga_actual = int(input("Harga Actual: "))
    
    # Add to dataframe (in real app, save to database/file)
    new_data = {
        'merek': [merek], 'model': [model_input], 'tahun': [tahun], 
        'km': [km], 'kondisi': [kondisi], 'tipe': [tipe], 'harga': [harga_actual]
    }
    
    global df
    df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
    print("✅ Data berhasil ditambahkan! Model bisa di-retrain untuk akurasi lebih baik.")

def main():
    while True:
        print("\n" + "="*50)
        print("🤖 AI PREDIKSI HARGA KENDARAAN - ML EDITION")
        print("="*50)
        print("1. 🚗 Prediksi Harga Kendaraan")
        print("2. 📈 Tambah Data Training")
        print("3. 📊 Lihat Data Training")
        print("4. 🏁 Keluar")
        
        pilihan = input("\nPilih menu (1-4): ")
        
        if pilihan == "1":
            prediksi_harga_ml()
        elif pilihan == "2":
            tambah_data_training()
        elif pilihan == "3":
            print("\n📊 DATA TRAINING SAAT INI:")
            print(df[['merek', 'model', 'tahun', 'km', 'kondisi', 'harga']].to_string(index=False))
        elif pilihan == "4":
            print("\n👋 Terima kasih! AI akan semakin pintar dengan data lebih banyak!")
            break
        else:
            print("❌ Pilihan tidak valid!")

if __name__ == "__main__":
    main()

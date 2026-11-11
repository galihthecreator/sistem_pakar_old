# app.py
import streamlit as st
import pandas as pd
from cbr_engine import jalankan_sistem_pakar

# ==========================================
# KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="Sistem Pakar Peredam Suara",
    page_icon="🔊",
    layout="wide"
)

# ==========================================
# LOAD DATASET (CACHE)
# ==========================================
@st.cache_data
def load_data():
    # Pastikan nama file sesuai dengan yang ada di foldermu
    return pd.read_csv("peredam_suara_clean.csv")

try:
    df_cases = load_data()
except FileNotFoundError:
    st.error("⚠️ File 'peredam_suara_clean.csv' tidak ditemukan. Pastikan file berada di folder yang sama dengan app.py.")
    st.stop()

# ==========================================
# HEADER
# ==========================================
st.title("🔊 Sistem Pakar Rekomendasi Peredam Suara")
st.markdown("""
Sistem ini menggunakan metode **Case-Based Reasoning (CBR)** dengan pendekatan **ICBR dan CAHWM** 
untuk merekomendasikan material dan sistem pemasangan peredam suara berdasarkan studi kasus masa lalu.
""")
st.divider()

# ==========================================
# FORM INPUT USER
# ==========================================
with st.form("input_form"):
    st.subheader("📝 Input Data Ruangan & Kebutuhan Akustik")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dimensi & Material Ruangan**")
        luas_ruangan = st.number_input("Luas Lantai Ruangan (m²)", min_value=1.0, max_value=500.0, value=40.0, step=1.0)
        luas_dinding = st.number_input("Luas Dinding yang Diberi Peredam (m²)", min_value=1.0, max_value=500.0, value=60.0, step=1.0)
        jenis_dinding = st.selectbox("Jenis Dinding Eksisting", ["hebel", "batako", "bata", "kayu", "gypsum", "beton"])
        ketebalan_dinding = st.number_input("Ketebalan Dinding Eksisting (cm)", min_value=1.0, max_value=100.0, value=10.0, step=1.0)
        
    with col2:
        st.markdown("**Karakteristik Suara**")
        sumber_suara = st.selectbox("Sumber Suara", [
            "Musik / audio", "Lalu lintas", "AC outdoor", "Genset", 
            "Percakapan", "Home theater", "Studio band", "Kompresor"
        ])
        intensitas_awal = st.number_input("Intensitas Suara Awal (dB)", min_value=30, max_value=150, value=85, step=1)
        target_suara = st.number_input("Target Suara Akhir (dB)", min_value=10, max_value=100, value=40, step=1)
        
        st.info(f"Delta Reduksi yang dibutuhkan: **{intensitas_awal - target_suara} dB**")

    # Tombol Submit
    submit = st.form_submit_button("🚀 Hitung Rekomendasi", use_container_width=True)

# ==========================================
# PROSES & TAMPILKAN HASIL
# ==========================================
if submit:
    # Susun dictionary input_user sesuai format cbr_engine
    input_user = {
        'luas_ruangan_m2': luas_ruangan,
        'jenis_dinding': jenis_dinding,
        'ketebalan_dinding_cm': ketebalan_dinding,
        'material_dinding': '-', # Tidak dipakai di engine, diisi dummy
        'luas_dinding_m2': luas_dinding,
        'sumber_suara': sumber_suara,
        'intensitas_suara_awal_db': intensitas_awal,
        'target_suara_db': target_suara,
        'delta_db': intensitas_awal - target_suara,
        'frekuensi_dominan': '-', # Tidak dipakai di engine, diisi dummy
    }
    
    with st.spinner("Menganalisis basis kasus dan menghitung validasi fisika (ASTM E336)..."):
        hasil = jalankan_sistem_pakar(input_user, df_cases)
    
    st.divider()
    st.subheader("📋 Hasil Rekomendasi Sistem Pakar")
    
    # 1. Tampilkan Status Kelayakan
    if hasil['status_tl'] == "TIDAK_MEMENUHI":
        st.warning(f"""
        ⚠️ **STATUS: TIDAK MEMENUHI KEBUTUHAN**  
        Kebutuhan TL (Transmission Loss) Anda adalah **{hasil['tl_dibutuhkan']:.1f} dB**, 
        namun TL tertinggi dari kandidat terbaik hanya **{hasil['tl_kandidat_terpilih']} dB**. 
        Sistem tidak dapat memberikan jaminan target tercapai. Silakan konsultasikan dengan ahli akustik.
        """)
    else:
        st.success(f"✅ **STATUS: MEMENUHI KEBUTUHAN** (TL Kandidat: {hasil['tl_kandidat_terpilih']} dB ≥ Kebutuhan: {hasil['tl_dibutuhkan']:.1f} dB)")

    # 2. Metrik Utama (Material, Tebal, Sistem)
    st.markdown("### 🏗️ Solusi yang Direkomendasikan")
    m1, m2, m3 = st.columns(3)
    m1.metric("Material Peredam", hasil['material'])
    m2.metric("Ketebalan Peredam", f"{hasil['tebal_mm']} mm")
    m3.metric("Sistem Pemasangan", hasil['sistem'])
    
    st.caption(f"Berdasarkan Kasus ID: **{hasil['id_kasus']}** (Skor Kemiripan: {hasil['skor_kemiripan']:.2f}% - Tingkat {hasil['tingkat_rekomendasi']})")

    # 3. Detail Fisika & Biaya
    st.markdown("### 📊 Detail Kalkulasi Fisika & Estimasi Biaya")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.info("**Validasi Fisika (ASTM E336)**")
        st.write(f"• **Koreksi Geometri Ruangan:** {hasil['koreksi_geometri']:.2f} dB")
        st.write(f"• **TL Material Terpilih:** {hasil['tl_material']} dB")
        st.write(f"• **Estimasi NR Lapangan:** {hasil['nr_aktual']} dB")
        
        if hasil['suara_luar'] <= hasil['target_user']:
            st.success(f"**Suara Akhir: {hasil['suara_luar']} dB** (Target {hasil['target_user']} dB TERCAPAI)")
        else:
            st.error(f"**Suara Akhir: {hasil['suara_luar']} dB** (Target {hasil['target_user']} dB TIDAK TERCAPAI)")

    with col_b:
        st.info("**Estimasi Biaya**")
        st.write(f"• **Total Biaya:** Rp {hasil['biaya_total']:,}")
        st.write(f"• **Biaya per m²:** Rp {hasil['biaya_per_m2']:,}")
        st.write(f"• Berdasarkan luas dinding: {luas_dinding} m²")

    # 4. Expander untuk Pengujian / Debug (Matriks & Top 10)
    st.divider()
    with st.expander("🔍 Lihat Detail Proses Retrieval (Top 10 Kasus Mirip)"):
        df_top10 = hasil['top_10'][['id', 'jenis_dinding', 'ketebalan_dinding_cm', 'sumber_suara', 'delta_db', 'material_peredam', 'similarity_score']]
        df_top10['similarity_score'] = (df_top10['similarity_score'] * 100).round(2).astype(str) + '%'
        st.dataframe(df_top10, use_container_width=True)
        
    with st.expander("🧮 Lihat Matriks Relasi Grey (R) - 6x4"):
        df_R = pd.DataFrame(
            hasil['matriks_R'],
            index=['Luas Ruangan', 'Ketebalan Dinding', 'Luas Dinding', 'Intensitas Awal', 'Target Suara', 'Delta dB'],
            columns=['Tebal Peredam (mm)', 'Estimasi NR (dB)', 'Suara di Luar (dB)', 'Estimasi Biaya (Rp)']
        )
        st.dataframe(df_R.style.background_gradient(cmap='Blues').format("{:.4f}"), use_container_width=True)

# ==========================================
# FOOTER
# ==========================================
st.divider()
st.caption("Dikembangkan untuk Tugas Mata Kuliah Sistem Pakar | Metode: AHP, Entropy, SLSQP, Grey Relational Analysis, Case-Based Reasoning.")
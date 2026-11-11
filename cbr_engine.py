# cbr_engine.py
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ==========================================
# KONFIGURASI GLOBAL
# ==========================================
FEATURE_COLS = [
    'luas_ruangan_m2', 'ketebalan_dinding_cm', 'luas_dinding_m2',
    'intensitas_suara_awal_db', 'target_suara_db', 'delta_db'
]

PROBLEM_NUM_COLS = FEATURE_COLS

SOLUTION_NUM_COLS = [
    'ketebalan_peredam_mm', 'estimasi_noise_reduction_db',
    'estimasi_suara_di_luar_db', 'estimasi_biaya_rp'
]

# Matriks AHP
J_expert = np.array([
    [1.00, 2.00, 0.33, 0.33, 0.50, 0.25],
    [0.50, 1.00, 0.25, 0.25, 0.33, 0.20],
    [3.00, 4.00, 1.00, 0.50, 2.00, 0.33],
    [3.00, 4.00, 2.00, 1.00, 2.00, 0.50],
    [2.00, 3.00, 0.50, 0.50, 1.00, 0.25],
    [4.00, 5.00, 3.00, 2.00, 4.00, 1.00],
])

# Prioritas Sistem Pemasangan (untuk Flanking Rule)
PRIORITAS_SISTEM = [
    'room-in-room', 'double 2x12', 'rongga udara', 'gypsum 12', 'ditempel'
]

# ==========================================
# FUNGSI-FUNGSI MATEMATIS
# ==========================================
def calculate_ahp_weights(J_matrix):
    m = J_matrix.shape[0]
    geom_mean = np.exp(np.mean(np.log(J_matrix), axis=1))
    Ws = geom_mean / np.sum(geom_mean)
    lambda_max = np.mean((J_matrix @ Ws) / Ws)
    CI = (lambda_max - m) / (m - 1) if m > 1 else 0.0
    RI_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.89, 5: 1.12, 6: 1.26, 7: 1.36, 8: 1.41, 9: 1.46, 10: 1.49}
    RI = RI_table.get(m, 1.26)
    CR = CI / RI if RI > 0 else 0.0
    return Ws, CR

def calculate_entropy_weights(V_matrix):
    k, m = V_matrix.shape
    eps = 1e-12
    column_sums = np.sum(V_matrix, axis=0) + eps
    Q = V_matrix / column_sums
    e_o = -1.0 / np.log(k) * np.sum(Q * np.log(Q + eps), axis=0)
    g_j = 1.0 - e_o
    Wo = g_j / np.sum(g_j)
    return Wo

def calculate_hybrid_weights(Ws, Wo, V_matrix):
    k, m = V_matrix.shape
    def objective_func(Wh):
        dev_s = np.sum(((Wh - Ws) * V_matrix) ** 2)
        dev_o = np.sum(((Wh - Wo) * V_matrix) ** 2)
        return dev_s + dev_o
    constraints = {'type': 'eq', 'fun': lambda Wh: np.sum(Wh) - 1.0}
    bounds = [(0.0, 1.0) for _ in range(m)]
    Wh_init = (Ws + Wo) / 2.0
    opt_result = minimize(objective_func, Wh_init, method='SLSQP', bounds=bounds, constraints=constraints)
    return opt_result.x

# ==========================================
# FUNGSI UTAMA (Mesin Inferensi)
# ==========================================
def jalankan_sistem_pakar(input_user, df_cases):
    # Pastikan delta_db dihitung secara eksplisit untuk mencegah inkonsistensi
    input_user['delta_db'] = input_user['intensitas_suara_awal_db'] - input_user['target_suara_db']
    
    # --- PHASE 2: HITUNG BOBOT HYBRID ---
    V_raw = df_cases[FEATURE_COLS].values.astype(float)
    v_min = V_raw.min(axis=0)
    v_max = V_raw.max(axis=0)
    v_denom = np.where((v_max - v_min) == 0, 1.0, (v_max - v_min))
    V_norm = (V_raw - v_min) / v_denom

    Ws, CR = calculate_ahp_weights(J_expert)
    Wo = calculate_entropy_weights(V_norm)
    Wh = calculate_hybrid_weights(Ws, Wo, V_norm)

    # --- PHASE 2: RETRIEVAL ---
    delta_target = input_user['delta_db']
    delta_tolerance = 30
    mask_delta = (df_cases['delta_db'] >= delta_target - delta_tolerance) & (df_cases['delta_db'] <= delta_target + delta_tolerance)
    df_pool = df_cases[mask_delta].copy()
    
    if len(df_pool) < 10:
        df_pool = df_cases.copy() # Fallback

    p0_raw = np.array([input_user[col] for col in FEATURE_COLS])
    p0_norm = (p0_raw - v_min) / v_denom

    V_pool = df_pool[FEATURE_COLS].values.astype(float)
    V_pool_norm = (V_pool - v_min) / v_denom

    diff_matrix = np.abs(V_pool_norm - p0_norm)
    sim_num_matrix = 1.0 - diff_matrix

    sim_cat_dinding = (df_pool['jenis_dinding'].str.lower().values == str(input_user['jenis_dinding']).lower()).astype(float)
    sim_cat_sumber = (df_pool['sumber_suara'].str.lower().values == str(input_user['sumber_suara']).lower()).astype(float)

    weighted_sim_num = np.dot(sim_num_matrix, Wh) / np.sum(Wh)
    total_similarity = 0.80 * weighted_sim_num + 0.10 * sim_cat_dinding + 0.10 * sim_cat_sumber

    df_pool['similarity_score'] = total_similarity
    top_10_candidates = df_pool.sort_values(by='similarity_score', ascending=False).head(10).copy()

    # --- PHASE 3: CAHWM (Case Adaptation) ---
    m_prob = len(PROBLEM_NUM_COLS)
    n_sol = len(SOLUTION_NUM_COLS)
    K_candidates = len(top_10_candidates)

    # 1. Matriks S & U
    S_raw = top_10_candidates[PROBLEM_NUM_COLS].values.astype(float)
    s_min = S_raw.min(axis=0)
    s_max = S_raw.max(axis=0)
    s_denom = np.where((s_max - s_min) == 0, 1.0, (s_max - s_min))
    S = (S_raw - s_min) / s_denom

    eps = 1e-12
    u_k = np.dot(S, Wh)
    u_norm = u_k / (np.sum(u_k) + eps)

    # 2. Matriks R (Grey Relational Analysis)
    p_2 = (S_raw - S_raw.min(0)) / (S_raw.max(0) - S_raw.min(0) + eps)
    Sol_raw = top_10_candidates[SOLUTION_NUM_COLS].values.astype(float)
    s_2 = (Sol_raw - Sol_raw.min(0)) / (Sol_raw.max(0) - Sol_raw.min(0) + eps)
    xi = 0.5
    R = np.zeros((m_prob, n_sol))
    for i in range(m_prob):
        for j in range(n_sol):
            diff = np.abs(p_2[:, i] - s_2[:, j])
            min_diff = np.min(diff)
            max_diff = np.max(diff)
            grey_coef = (min_diff + xi * max_diff) / (diff + xi * max_diff + eps)
            R[i, j] = np.sum(u_norm * grey_coef)

    # 3. Matriks A (Adaptabilitas & Penalti)
    sim_scores = top_10_candidates['similarity_score'].values
    A = np.zeros((K_candidates, n_sol))
    for k_idx in range(K_candidates):
        row_cand = top_10_candidates.iloc[k_idx]
        penalty = 1.0
        if row_cand['ketebalan_peredam_mm'] > 75 and input_user['luas_ruangan_m2'] < 25.0: penalty *= 0.50
        if row_cand['estimasi_noise_reduction_db'] < delta_target: penalty *= 0.30
        if 'ditempel langsung' in str(row_cand['sistem_pemasangan']).lower() and delta_target > 25: penalty *= 0.20
        for j in range(n_sol):
            A[k_idx, j] = 0.5 * sim_scores[k_idx] + 0.5 * penalty

    # 4. Matriks HWM
    SR_matrix = np.dot(S, R)
    WM = SR_matrix * A
    HWM = WM / (np.sum(WM, axis=0) + eps)

    # 5. Ekstraksi Solusi Numerik & Kategorikal
    adapted_numerical_solution = np.sum(HWM * Sol_raw, axis=0)
    adapted_sol_dict = {col: adapted_numerical_solution[idx] for idx, col in enumerate(SOLUTION_NUM_COLS)}

    cat_weights = u_norm * A.mean(axis=1)
    def adapt_categorical(top_df, col_name, weights):
        votes = {}
        for idx, row in top_df.reset_index(drop=True).iterrows():
            val = row[col_name]
            votes[val] = votes.get(val, 0.0) + weights[idx]
        return max(votes, key=votes.get)

    adapted_material = adapt_categorical(top_10_candidates, 'material_peredam', cat_weights)
    adapted_sistem = adapt_categorical(top_10_candidates, 'sistem_pemasangan', cat_weights)

    # --- PHASE 3: FLANKING RULE & VALIDASI FISIKA ---
    S_user = input_user['luas_dinding_m2']
    A_user = input_user['luas_ruangan_m2'] * 0.20 + 5.0
    koreksi_geometri_user = 10 * np.log10(S_user / A_user)
    tl_dibutuhkan = delta_target + koreksi_geometri_user

    # Pilih sistem dari dataset
    mask_tl = top_10_candidates['estimasi_noise_reduction_db'] >= tl_dibutuhkan
    pool = top_10_candidates[mask_tl].copy() if mask_tl.any() else top_10_candidates.copy()
    status_tl = "MEMENUHI" if mask_tl.any() else "TIDAK_MEMENUHI"

    best_case = None
    for keyword in PRIORITAS_SISTEM:
        mask_kw = pool['sistem_pemasangan'].apply(lambda s: keyword in str(s).lower())
        if mask_kw.any():
            cand = pool[mask_kw].copy()
            cand['final'] = cand['similarity_score'] * cat_weights[pool.index.get_indexer(cand.index)]
            best_row = cand.loc[cand['final'].idxmax()]
            best_case = best_row
            break
            
    if best_case is None:
        best_case = pool.loc[pool['estimasi_noise_reduction_db'].idxmax()]

    # Hitung Output Akhir
    adapted_sistem = best_case['sistem_pemasangan']
    adapted_material = best_case['material_peredam']
    tl_material_hwm = round(float(best_case['estimasi_noise_reduction_db']), 1)
    tebal_mm = int(round(float(best_case['ketebalan_peredam_mm']) / 5) * 5)

    est_nr_aktual = round(tl_material_hwm - koreksi_geometri_user, 1)
    est_suara_luar = max(0.0, round(input_user['intensitas_suara_awal_db'] - est_nr_aktual, 1))

    biaya_kasus = float(best_case['estimasi_biaya_rp'])
    luas_kasus = float(best_case['luas_dinding_m2'])
    est_biaya_total = int(round((biaya_kasus / luas_kasus) * S_user / 50000) * 50000)
    biaya_per_m2 = int(round(est_biaya_total / S_user)) if S_user > 0 else 0

    max_sim = top_10_candidates['similarity_score'].max()
    if max_sim >= 0.85: tingkat_rekomendasi = 'Tinggi'
    elif max_sim >= 0.70: tingkat_rekomendasi = 'Sedang'
    else: tingkat_rekomendasi = 'Rendah'

    # Return semua hasil dalam bentuk dictionary
    return {
        'status_tl': status_tl,
        'tl_dibutuhkan': tl_dibutuhkan,
        'tl_kandidat_terpilih': tl_material_hwm,
        'id_kasus': int(best_case['id']),
        'material': adapted_material,
        'tebal_mm': tebal_mm,
        'sistem': adapted_sistem,
        'tl_material': tl_material_hwm,
        'koreksi_geometri': koreksi_geometri_user,
        'nr_aktual': est_nr_aktual,
        'suara_luar': est_suara_luar,
        'target_user': input_user['target_suara_db'],
        'biaya_total': est_biaya_total,
        'biaya_per_m2': biaya_per_m2,
        'skor_kemiripan': float(max_sim * 100),
        'tingkat_rekomendasi': tingkat_rekomendasi,
        'matriks_R': R, # Opsional: dikirim ke Streamlit jika ingin ditampilkan
        'top_10': top_10_candidates # Opsional: untuk tabel di Streamlit
    }
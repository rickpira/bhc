# app.py — Aplicativo para Cálculo do Balanço Hídrico Climatológico — v1 (corrigido)
# Professores: Claudio Ricardo da Silva (UFU) e Nildo da Silva Dias (UFERSA)

import streamlit as st
import numpy as np
import pandas as pd
import math, os, io, zipfile, tempfile, hashlib, requests
import matplotlib.pyplot as plt
import re
import rasterio  # leitura de GeoTIFFs

# ============== VISUAL ==============
st.set_page_config(page_title="Balanço Hídrico Climatológico", layout="wide")
st.markdown("<style>.stApp{background-color:#f5f9f6;}</style>", unsafe_allow_html=True)
st.markdown("""
<h1 style='text-align:center; color:green;'>Aplicativo para Cálculo do Balanço Hídrico Climatológico</h1>
<h4 style='text-align:center;'>Desenvolvido pelos professores Claudio Ricardo da Silva (UFU) e Nildo da Silva Dias (UFERSA)</h4>
<h5 style='text-align:center; color:gray;'>Versão 1</h5>
""", unsafe_allow_html=True)
st.divider()

# ============== ESTADO PERSISTENTE ==============
if "res" not in st.session_state:
    st.session_state.res = None

# ============== CONSTANTES ==============
MESES = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
DIAS_MES = np.array([31,28,31,30,31,30,31,31,30,31,30,31], dtype=float)

# ZIP único com TAVG e PREC (sua release no GitHub)
RELEASE_TAG = "v1-data"
REPO_OWNER  = "rickpira"
REPO_NAME   = "bhc"
WC_FILE     = "worldclim_10m.zip"
def gh_release_asset_url(tag: str, filename: str) -> str:
    return f"https://github.com/{REPO_OWNER}/{REPO_NAME}/releases/download/{tag}/{filename}"
GITHUB_ZIP_URL = gh_release_asset_url(RELEASE_TAG, WC_FILE)

# ============== SIDEBAR ==============
with st.sidebar:
    st.markdown("### ⚙️ Parâmetros")
    lat = st.number_input("🧭 Latitude (Sul negativo)", value=-18.9000, step=0.0001, format="%.4f")
    lon = st.number_input("🧭 Longitude (Oeste negativo)", value=-48.3000, step=0.0001, format="%.4f")
    CAD = st.number_input("🪣 CAD do solo (mm)", value=125.0, min_value=1.0, step=1.0)

    st.markdown("### ✍️ Inserção manual (opcional)")
    permitir_manual = st.checkbox("Quero inserir/alterar dados manualmente")
    if permitir_manual:
        df_manual = pd.DataFrame({
            "Mês": MESES,
            "Temperatura Média (°C)": [np.nan]*12,
            "Precipitação (mm)": [np.nan]*12,
            "ETo (Thornthwaite) (mm)": [np.nan]*12,
        })
        st.caption("Preencha o que quiser; o que ficar em branco o app completa (base/Thornthwaite).")
        df_manual = st.data_editor(df_manual, num_rows="fixed", hide_index=True, use_container_width=True)
    else:
        df_manual = None

    if st.button("🔄 Limpar resultados"):
        st.session_state.res = None

calcular = st.button("🚀 Calcular balanço")

# ============== FUNÇÕES ==============
def fotoperiodo_mensal(latitude):
    dias_jul = [15,45,75,105,135,162,198,228,258,288,318,344]
    phi = math.radians(latitude)
    N = []
    for J in dias_jul:
        delta = math.radians(23.45 * math.sin(math.radians(360 * (284 + J)/365)))
        cosw = -math.tan(phi) * math.tan(delta)
        cosw = max(-1.0, min(1.0, cosw))
        w = math.acos(cosw)
        N.append((24/math.pi) * w)
    return np.array(N, dtype=float)

def eto_thornthwaite(T, latitude):
    """ETo via Thornthwaite com COR = (N/12)*(d/30)."""
    T = np.array(T, dtype=float).copy()
    T = np.where(np.isnan(T), 0.0, T)  # segurança
    T[T < 0] = 0.0
    i = (T/5.0)**1.514
    I = np.sum(i)
    N = fotoperiodo_mensal(latitude)
    COR = (N/12.0) * (DIAS_MES/30.0)
    if I == 0:
        return np.zeros(12), COR
    a = (6.75e-7)*(I**3) - (7.71e-5)*(I**2) + (1.792e-2)*I + 0.49239
    ETo0 = 16.0 * ((10.0 * T / I) ** a)
    return ETo0 * COR, COR

@st.cache_data(show_spinner=False)
def download_from_github(url: str) -> str:
    os.makedirs(os.path.join(tempfile.gettempdir(), "wc_cache"), exist_ok=True)
    out = os.path.join(tempfile.gettempdir(), "wc_cache", "worldclim_10m.zip")
    if not os.path.exists(out):
        # ALTERAÇÃO: aumentar timeout e permitir token opcional para evitar rate limit
        headers = {}
        try:
            tok = st.secrets.get("GITHUB_TOKEN", None)
        except Exception:
            tok = None
        if tok:
            headers["Authorization"] = f"token {tok}"
        r = requests.get(url, stream=True, timeout=180, headers=headers)
        if r.status_code == 404:
            st.error("Arquivo da release não encontrado (404). Verifique o nome do asset.")
            r.raise_for_status()
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=262144):
                if chunk:
                    f.write(chunk)
    return out

@st.cache_data(show_spinner=False)
def unzip_cached(zip_path: str) -> str:
    with open(zip_path, "rb") as f:
        h = hashlib.md5(f.read(1024*1024)).hexdigest()
    outdir = os.path.join(tempfile.gettempdir(), f"wc_unzip_{h}")
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(outdir)
    return outdir

# ====== AMOSTRAGEM SEGURA WORLDCLIM ======
def _sanitize_prec(x):
    if x is None or np.isnan(x): return np.nan
    if x < 0: return np.nan
    return float(x)

def _sanitize_temp_c(x):
    if x is None or np.isnan(x): return np.nan
    if x < -90 or x > 80: return np.nan
    return float(x)

def _read_masked_value(src, lon, lat):
    r, c = src.index(lon, lat)  # (x=lon, y=lat)
    if r < 0 or c < 0 or r >= src.height or c >= src.width:
        return np.nan, "fora_bounds"
    arr = src.read(1, masked=True)  # aplica NoData automaticamente
    val = arr[r, c]
    if np.ma.is_masked(val):
        return np.nan, "nodata"
    return float(val), "ok"

def _read_nearest_valid(src, lon, lat, radii=(3,5,8,12)):
    r, c = src.index(lon, lat)
    arr = src.read(1, masked=True)
    if 0 <= r < src.height and 0 <= c < src.width and not np.ma.is_masked(arr[r, c]):
        return float(arr[r, c]), "ok"
    for rad in radii:
        r0, r1 = max(0, r-rad), min(src.height, r+rad+1)
        c0, c1 = max(0, c-rad), min(src.width, c+rad+1)
        win = arr[r0:r1, c0:c1]
        if win.count() > 0:
            return float(win.compressed()[0]), f"preenchido_vizinho({rad}px)"
    return np.nan, "nodata"

def _detectar_escala_temp(vals_validos):
    if len(vals_validos) == 0:
        return "celsius"
    vmin = float(np.nanmin(vals_validos)); vmax = float(np.nanmax(vals_validos))
    if (vmin > -900 and vmax < 900) and (abs(vmin) > 120 or abs(vmax) > 120):
        return "c_times_10"
    return "celsius"

def _localizar_tif(pasta: str, prefix: str, m: int):
    alvo_mm = f"_{m:02d}.tif".lower()
    pfx = prefix.lower()
    for root, _, files in os.walk(pasta):
        for fn in files:
            fn_low = fn.lower()
            if fn_low.endswith(alvo_mm) and pfx in fn_low and fn_low.endswith(".tif"):
                return os.path.join(root, fn)
    return None

def ler_serie(prefix: str, pasta: str, lon: float, lat: float):
    """
    Lê 12 rasters mensais (01..12) do 'prefix' em qualquer subpasta.
    - Trata NoData via máscara.
    - Preenche AUTOMATICAMENTE com vizinho mais próximo (3,5,8,12 px).
    - Para temperatura, detecta °C×10 e converte para °C; sanitiza [-90,80].
    - Para precipitação, sanitiza (≥0).
    """
    vals_raw, status = [], []
    for m in range(1, 13):
        path = _localizar_tif(pasta, prefix, m)
        if path is None:
            vals_raw.append(np.nan); status.append("arquivo_ausente"); continue
        with rasterio.open(path) as src:
            v, q = _read_nearest_valid(src, lon, lat, radii=(3,5,8,12))
            if np.isnan(v):
                v, q2 = _read_masked_value(src, lon, lat)
                q = q if not np.isnan(v) else q2
            vals_raw.append(v); status.append(q)

    vals_raw = np.array(vals_raw, dtype=float)
    pfx = prefix.lower()
    is_temp = any(k in pfx for k in ["tavg", "tmean", "tmin", "tmax", "temp"])
    is_prec = "prec" in pfx or "ppt" in pfx or "prcp" in pfx

    if is_temp:
        validos = vals_raw[np.isfinite(vals_raw)]
        escala = _detectar_escala_temp(validos) if validos.size else "celsius"
        if escala == "c_times_10":
            vals_raw = vals_raw / 10.0
        vals = np.array([_sanitize_temp_c(v) for v in vals_raw], dtype=float)
    elif is_prec:
        vals = np.array([_sanitize_prec(v) for v in vals_raw], dtype=float)
    else:
        vals = np.where(np.abs(vals_raw) > 1e6, np.nan, vals_raw)

    if np.isnan(vals).any():
        vals = np.nan_to_num(vals, nan=0.0)
    return vals

def bloco_umido_inicio_fim(D, tol=1e-6):
    n = len(D); pos = D >= -tol
    if not np.any(pos): return None, None, 'todos_neg'
    if np.all(pos):     return 0, n-1, 'todos_pos'
    i_ini = None
    for i in range(n):
        prev = (i-1) % n
        if (D[i] >= -tol) and (D[prev] < -tol):
            i_ini = i; break
    if i_ini is None: i_ini = int(np.where(pos)[0][0])
    D2 = np.r_[D, D]; j = i_ini
    while j < i_ini + n and D2[j] >= -tol: j += 1
    i_fim = (j-1) % n
    return i_ini, i_fim, 'bloco_pos'

def calcular_ARM_exponencial(P, ETo, CAD, tol=1e-6):
    n = 12; D = np.array(P) - np.array(ETo)
    i_ini, i_fim, caso = bloco_umido_inicio_fim(D, tol)
    inicio_ARM = ((int(np.argmax(D)) + 1) % n) if caso=='todos_neg' else ((i_fim + 1) % n)

    idx_rot = [(inicio_ARM + k) % n for k in range(n)]
    D_rot = D[idx_rot]

    pos = D_rot[D_rot > 0]; neg = D_rot[D_rot < 0]
    soma_pos = float(pos.sum()) if pos.size else 0.0
    soma_neg = float(neg.sum()) if neg.size else 0.0
    if soma_pos >= CAD - tol:
        ARM_ini = CAD
    else:
        ARM_ini = min(CAD, soma_pos) if abs(soma_neg) <= tol else (soma_pos / (1.0 - math.exp(soma_neg / CAD)))

    ARM = ARM_ini; ARM_seq_rot = []
    for d in D_rot:
        if d >= -tol:
            ARM = min(CAD, ARM + d)
        else:
            delta_eq = CAD * math.log(max(ARM, 1e-9) / CAD); delta_eq += d
            ARM = CAD * math.exp(delta_eq / CAD)
            ARM = max(0.0, min(ARM, CAD))
        ARM_seq_rot.append(round(ARM, 2))

    ARM_final = [None]*n
    for k, i_orig in enumerate(idx_rot): ARM_final[i_orig] = ARM_seq_rot[k]
    ARM_final = np.array(ARM_final, float)

    ALT = np.zeros(n); ETR = np.zeros(n); DEF = np.zeros(n); EXC = np.zeros(n)
    for i in range(n):
        ip = (i-1) % n
        ALT[i] = ARM_final[i] - ARM_final[ip]
        if D[i] >= -tol:
            recarga = max(0.0, ALT[i])
            EXC[i]  = max(0.0, D[i] - recarga)
            ETR[i]  = ETo[i]
            DEF[i]  = 0.0
        else:
            retirada = -ALT[i]
            ETR[i]   = P[i] + retirada
            DEF[i]   = max(0.0, ETo[i] - ETR[i])
            EXC[i]   = 0.0

    return ARM_final.round(2), ALT.round(2), ETR.round(2), DEF.round(2), EXC.round(2), inicio_ARM

def afericao_ok(df, tol=0.2):
    sP   = float(df['Precipitação (mm)'].sum())
    sETo = float(df['ETo (Thornthwaite) (mm)'].sum())
    sPE  = float((df['Precipitação (mm)'] - df['ETo (Thornthwaite) (mm)']).sum())
    sALT = float(df['ALT (mm)'].sum())
    sETR = float(df['ETR (mm)'].sum())
    sDEF = float(df['DEF (mm)'].sum())
    sEXC = float(df['EXC (mm)'].sum())
    checks = [
        abs(sALT) <= tol,
        abs(sP - (sETo + sPE)) <= tol,
        abs(sP - (sETR + sEXC)) <= tol,
        abs(sETo - (sETR + sDEF)) <= tol,
    ]
    return all(checks)

# ===== Thornthwaite (com 2ª letra pela sua Tabela 2) =====
def class_thornthwaite(df, lat):
    ETo = df["ETo (Thornthwaite) (mm)"].to_numpy(float)
    DEF = df["DEF (mm)"].to_numpy(float)
    EXC = df["EXC (mm)"].to_numpy(float)
    PET = float(ETo.sum())
    soma_DEF = float(DEF.sum()); soma_EXC = float(EXC.sum())

    if PET <= 0:
        Ih = Ia = Im = np.nan
    else:
        Ih = 100.0 * (soma_EXC / PET)
        Ia = 100.0 * (soma_DEF / PET)
        Im = Ih - 0.6 * Ia

    def umidade(Im):
        if np.isnan(Im): return ("Indef","Indefinido")
        if Im >= 100:  return ("A","Perúmido")
        if Im >= 80:   return ("B4","Muito úmido")
        if Im >= 60:   return ("B3","Úmido")
        if Im >= 40:   return ("B2","Úmido")
        if Im >= 20:   return ("B1","Úmido")
        if Im >= 0:    return ("C2","Subúmido úmido")
        if Im >= -20:  return ("C1","Subúmido seco")
        if Im >= -40:  return ("D","Semiárido")
        return ("E","Árido")
    u_code, u_desc = umidade(Im)

    idx_verao = np.array([11,0,1]) if lat < 0 else np.array([5,6,7])
    idx_inverno = np.array([5,6,7]) if lat < 0 else np.array([11,0,1])

    def segunda_letra(u_code, Ia, Ih, DEF, EXC):
        if u_code in ["A","B4","B3","B2","B1","C2"]:
            ia = Ia if not np.isnan(Ia) else 0.0
            def_ver = float(DEF[idx_verao].sum()); def_inv = float(DEF[idx_inverno].sum())
            est = "s" if def_ver >= def_inv else "w"
            if ia < 16.7:   return "r", "sem ou pequena deficiência hídrica"
            elif ia < 33.3: return est, f"deficiência hídrica moderada no {'verão' if est=='s' else 'inverno'}"
            else:           return est+"2", f"grande deficiência hídrica no {'verão' if est=='s' else 'inverno'}"
        else:
            ih = Ih if not np.isnan(Ih) else 0.0
            exc_ver = float(EXC[idx_verao].sum()); exc_inv = float(EXC[idx_inverno].sum())
            est = "s" if exc_ver >= exc_inv else "w"
            if ih < 10:     return "d", "excedente hídrico pequeno ou nulo"
            elif ih < 20:   return est, f"excedente hídrico moderado no {'verão' if est=='s' else 'inverno'}"
            else:           return est+"2", f"grande excedente hídrico no {'verão' if est=='s' else 'inverno'}"

    saz_code, saz_desc = segunda_letra(u_code, Ia, Ih, DEF, EXC)

    ETo_verao = float(ETo[idx_verao].sum()); ETo_anual = PET
    ETo_verao_sobre_ano = np.nan if ETo_anual <= 0 else ETo_verao / ETo_anual

    def term(PET):
        if np.isnan(PET): return ("Indef","Indefinido")
        if PET >= 1140:  return ("A’","Megatérmico")
        if PET >= 997:   return ("B’4","Mesotérmico (alto)")
        if PET >= 885:   return ("B’3","Mesotérmico")
        if PET >= 712:   return ("B’2","Mesotérmico (baixo)")
        if PET >= 570:   return ("B’1","Mesotérmico (muito baixo)")
        if PET >= 427:   return ("C’2","Microtérmico (alto)")
        if PET >= 285:   return ("C’1","Microtérmico")
        if PET >= 142:   return ("D’","Tundra")
        return ("E’","Gelo perpétuo")
    t_code, t_desc = term(PET)

    formula = f"{u_code} {saz_code} {t_code}".strip()
    descricao = f"Clima {u_desc.lower()}, {t_desc.lower()}, {saz_desc}."
    return {
        "formula": formula, "descricao": descricao,
        "Ih": Ih, "Ia": Ia, "Im": Im,
        "PET_anual": ETo_anual,
        "ETo_verao_mm": ETo_verao,
        "ETo_verao_sobre_ano": ETo_verao_sobre_ano
    }

# ===== Köppen–Geiger (Peel et al., 2007) =====
# ALTERAÇÃO: assinatura agora recebe lat para usar verão/inverno por hemisfério

def class_koppen_v2(df, lat, use_minus3=True):
    """
    Implementação Köppen–Geiger seguindo Peel et al. (2007), com correções:
    - Grupos A e B usam semestre alto-sol/baixo-sol por hemisfério (verão/inverno fixos).
    - Grupos C/D mantêm a divisão hot6/cold6 por temperatura.
    """
    T = df["Temperatura Média (°C)"].to_numpy(float)
    P = df["Precipitação (mm)"].to_numpy(float)

    # --- ALTERAÇÃO: índices de verão e inverno por hemisfério ---
    if lat < 0:  # Hemisfério Sul
        verao_idx   = np.array([9,10,11,0,1,2])   # Out, Nov, Dez, Jan, Fev, Mar
        inverno_idx = np.array([3,4,5,6,7,8])     # Abr, Mai, Jun, Jul, Ago, Set
    else:        # Hemisfério Norte
        verao_idx   = np.array([3,4,5,6,7,8])     # Abr–Set
        inverno_idx = np.array([9,10,11,0,1,2])   # Out–Mar

    Tbar = float(np.nanmean(T))
    Psum = float(np.nansum(P))
    Tmin = float(np.nanmin(T))
    Tmax = float(np.nanmax(T))

    # ============================
    # Grupo E (polar)
    # ============================
    if Tmax < 10.0:
        return {"formula": "EF" if Tmax < 0.0 else "ET",
                "descricao": "Clima polar de gelo perpétuo" if Tmax < 0.0 else "Clima polar de tundra"}

    # ============================
    # Grupo B (árido)
    # ============================
    # ALTERAÇÃO: ajuste sazonal com verão/inverno fixos
    P_hot  = float(np.nansum(P[verao_idx]))
    P_cold = float(np.nansum(P[inverno_idx]))
    frac_hot = P_hot / Psum if Psum > 0 else 0.0
    frac_cold = P_cold / Psum if Psum > 0 else 0.0

    if frac_hot >= 0.70:
        add = 280.0
    elif frac_cold >= 0.70:
        add = 0.0
    else:
        add = 140.0

    Pth = 20.0 * Tbar + add
    if Psum < Pth:
        subtype = "BW" if Psum < 0.5 * Pth else "BS"
        hk = "h" if Tbar >= 18.0 else "k"
        return {"formula": subtype + hk,
                "descricao": ("Clima desértico" if subtype=="BW" else "Clima de estepe") +
                             (" quente" if hk=="h" else " frio")}

    # ============================
    # Grupo A (tropical)
    # ============================
    if Tmin >= 18.0:
        driest = float(np.nanmin(P))
        if driest >= 60.0:
            return {"formula": "Af", "descricao": "Tropical úmido, sem estação seca"}
        if driest >= (100.0 - Psum/25.0):
            return {"formula": "Am", "descricao": "Tropical monçônico"}

        # ALTERAÇÃO: Aw vs As pelo semestre correto
        P_dry_summer = float(np.nanmin(P[verao_idx]))
        P_dry_winter = float(np.nanmin(P[inverno_idx]))
        dry_in_winter = P_dry_winter <= P_dry_summer
        return {"formula": "Aw" if dry_in_winter else "As",
                "descricao": "Tropical com estação seca no " + ("inverno" if dry_in_winter else "verão")}

    # ============================
    # Grupos C/D (temperado/continental)
    # ============================
    # Mantém hot6/cold6 por temperatura
    order = np.argsort(T)[::-1]
    hot6 = np.zeros(12, dtype=bool); hot6[order[:6]] = True
    cold6 = ~hot6

    months_ge10 = int(np.sum(T >= 10.0))
    cd_limit = -3.0 if use_minus3 else 0.0
    main = "D" if Tmin < cd_limit else "C"

    P_dry_summer = float(np.nanmin(P[hot6]))
    P_wet_winter = float(np.nanmax(P[cold6]))
    P_dry_winter = float(np.nanmin(P[cold6]))
    P_wet_summer = float(np.nanmax(P[hot6]))

    is_s = (P_dry_summer < 40.0) and (P_dry_summer < (P_wet_winter / 3.0))
    is_w = (P_dry_winter < (P_wet_summer / 10.0))
    mid = "s" if is_s else ("w" if is_w else "f")

    if np.nanmax(T) >= 22.0 and months_ge10 >= 4:
        last = "a"
    elif months_ge10 >= 4:
        last = "b"
    else:
        last = "c"

    formula = main + mid + last
    legendas = {
        "Cfa":"Temperado úmido, verão quente","Cfb":"Temperado úmido, verão morno","Cfc":"Temperado úmido, verão fresco",
        "Csa":"Mediterrâneo, verão quente","Csb":"Mediterrâneo, verão morno","Csc":"Mediterrâneo, verão fresco",
        "Cwa":"Temperado com inverno seco, verão quente","Cwb":"Temperado com inverno seco, verão morno","Cwc":"Temperado com inverno seco, verão fresco",
        "Dfa":"Continental úmido, verão quente","Dfb":"Continental úmido, verão morno","Dfc":"Continental úmido, verão fresco",
        "Dwa":"Continental com inverno seco, verão quente","Dwb":"Continental com inverno seco, verão morno","Dwc":"Continental com inverno seco, verão fresco",
        "Dsa":"Continental com verão seco, verão quente","Dsb":"Continental com verão seco, verão morno","Dsc":"Continental com verão seco, verão fresco",
    }
    desc = legendas.get(formula, f"Clima {('temperado' if main=='C' else 'continental')} ({mid}{last}).")
    return {"formula": formula, "descricao": desc}

# ============== CÁLCULO (apenas quando clicar) ==============
if calcular:
    try:
        # ALTERAÇÃO: spinner informativo para o primeiro uso
        with st.spinner("Baixando e preparando dados base (pode levar até 1–2 min no primeiro acesso)..."):
            zip_path = download_from_github(GITHUB_ZIP_URL)
            pasta = unzip_cached(zip_path)

        # Séries WorldClim (NoData tratado, temp em °C)
        T_wc = ler_serie("wc2.1_10m_tavg", pasta, lon, lat)
        P_wc = ler_serie("wc2.1_10m_prec", pasta, lon, lat)

        # Completa com manual, se houver
        if df_manual is not None:
            T_eff = np.where(df_manual["Temperatura Média (°C)"].notna(),
                             df_manual["Temperatura Média (°C)"].to_numpy(float), T_wc)
            P_eff = np.where(df_manual["Precipitação (mm)"].notna(),
                             df_manual["Precipitação (mm)"].to_numpy(float), P_wc)
            ETo_eff = df_manual["ETo (Thornthwaite) (mm)"].to_numpy(float)
            mask_nan = np.isnan(ETo_eff)
            if mask_nan.any():
                ETo_calc, _ = eto_thornthwaite(T_eff, lat)
                ETo_eff[mask_nan] = ETo_calc[mask_nan]
        else:
            T_eff, P_eff = T_wc, P_wc
            ETo_eff, _ = eto_thornthwaite(T_eff, lat)

        # COR e fotoperíodo
        N_hours = fotoperiodo_mensal(lat)
        COR_vec = (N_hours/12.0) * (DIAS_MES/30.0)

        # Balanço
        ARM, ALT, ETR, DEF, EXC, inicio_ARM = calcular_ARM_exponencial(P_eff, ETo_eff, float(CAD))

        df_out = pd.DataFrame({
            "Mês": MESES,
            "Fotoperíodo (h)": np.round(N_hours, 2),
            "Temperatura Média (°C)": np.round(T_eff, 2),
            "Precipitação (mm)": np.round(P_eff, 2),
            "COR": np.round(COR_vec, 4),
            "ETo (Thornthwaite) (mm)": np.round(ETo_eff, 2),
            "P - ETo (mm)": np.round(P_eff - ETo_eff, 2),
            "ARM (mm)": ARM, "ALT (mm)": ALT, "ETR (mm)": ETR,
            "DEF (mm)": DEF, "EXC (mm)": EXC,
        })

        thorn = class_thornthwaite(df_out, lat)
        # ALTERAÇÃO: passar lat para Köppen
        kopp  = class_koppen_v2(df_out, lat=lat, use_minus3=False)  # use_minus3=True para limiar -3°C

        st.session_state.res = {
            "df_out": df_out, "inicio_ARM": inicio_ARM,
            "thorn": thorn, "kopp": kopp,
            "CAD": CAD, "lat": lat, "lon": lon
        }

    except Exception as e:
        # ALTERAÇÃO: mostrar traceback completo para depuração
        st.exception(e)
        st.error("Falha ao calcular. Veja o traceback acima. Se aparecer 'rate limit', adicione um GITHUB_TOKEN nos Secrets do app.")

# ============== RENDERIZAÇÃO PERSISTENTE ==============
if st.session_state.res is not None:
    res = st.session_state.res
    df_out = res["df_out"]; inicio_ARM = res["inicio_ARM"]
    thorn  = res["thorn"];  kopp       = res["kopp"]
    CAD    = res["CAD"];    lat        = res["lat"]; lon = res["lon"]

    # Aferição
    if afericao_ok(df_out, tol=0.2):
        st.success("✅ Cálculos aferidos (tolerância ±0,2 mm).")
    else:
        st.error("⚠️ Aferição falhou — verifique os dados de entrada.")

    # Classificações
    st.subheader("🧭 Classificações Climáticas")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**🌦 Thornthwaite — Fórmula:** `{thorn['formula']}`")
        st.markdown(f"**Descrição:** {thorn['descricao']}")
        frac = thorn["ETo_verao_sobre_ano"]
        frac_txt = "indefinido" if np.isnan(frac) else f"{frac*100:.1f}%"
        st.markdown(f"**ETo_verão/ETo_ano:** {frac_txt}  "
                    f"(ETo_verão = {thorn['ETo_verao_mm']:.1f} mm; ETo_anual = {thorn['PET_anual']:.1f} mm)")
        st.caption(f"Índices: Ih={thorn['Ih']:.1f}, Ia={thorn['Ia']:.1f}, Im={thorn['Im']:.1f}")
    with c2:
        st.markdown(f"**🌍 Köppen–Geiger — Fórmula:** `{kopp['formula']}`")
        st.markdown(f"**Descrição:** {kopp['descricao']}")

    # Tabela
    st.subheader("📋 Tabela do Balanço Hídrico")
    st.dataframe(df_out, use_container_width=True)
    st.caption(f"🔁 Início do ciclo de ARM (após bloco úmido): **{MESES[inicio_ARM]}** — CAD={CAD:.0f} mm")

    # Download Excel
    from datetime import datetime
    def gerar_excel(df_bhc: pd.DataFrame, info: dict | None = None) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as wr:
            df_bhc.to_excel(wr, index=False, sheet_name="BHC")
            if info:
                meta = pd.DataFrame([
                    ["Latitude",  info.get("lat")],
                    ["Longitude", info.get("lon")],
                    ["CAD (mm)",  info.get("CAD")],
                    ["Thornthwaite - Fórmula", info.get("thorn_formula")],
                    ["Thornthwaite - Descrição", info.get("thorn_desc")],
                    ["ETo_verão/ETo_ano (%)", info.get("eto_raz_percent")],
                    ["Köppen - Fórmula", info.get("koppen_formula")],
                    ["Köppen - Descrição", info.get("koppen_desc")],
                    ["Gerado em", datetime.now().strftime("%Y-%m-%d %H:%M")],
                ], columns=["Campo","Valor"])
                meta.to_excel(wr, index=False, sheet_name="Resumo")
            wr.close()
        buf.seek(0); return buf.getvalue()

    frac = thorn["ETo_verao_sobre_ano"]
    info_excel = {
        "lat": lat, "lon": lon, "CAD": CAD,
        "thorn_formula": thorn["formula"], "thorn_desc": thorn["descricao"],
        "eto_raz_percent": ("indefinido" if np.isnan(frac) else f"{frac*100:.1f}%"),
        "koppen_formula": kopp["formula"], "koppen_desc": kopp["descricao"],
    }
    excel_bytes = gerar_excel(df_out, info_excel)
    st.download_button(
        "💾 Baixar BHC em Excel (.xlsx)",
        data=excel_bytes,
        file_name="balanco_hidrico.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

    # Gráficos
    st.subheader("📈 Excedente (↑) e Deficiência (↓)")
    DEF_neg = -df_out["DEF (mm)"].to_numpy(float)
    EXC_pos =  df_out["EXC (mm)"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10,5))
    ax.fill_between(MESES, EXC_pos, 0, alpha=0.65, label="EXC (mm)")
    ax.fill_between(MESES, DEF_neg, 0, alpha=0.65, label="DEF (mm)")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("mm"); ax.set_title("Excedente (positivo) e Deficiência (negativo)")
    ax.grid(axis='y', linestyle='--', alpha=0.5); ax.legend()
    st.pyplot(fig)

    st.subheader("📈 CAD × ARM")
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(MESES, [CAD]*12, label="CAD (mm)")
    ax2.plot(MESES, df_out["ARM (mm)"].to_numpy(float), label="ARM (mm)")
    ax2.set_ylabel("mm"); ax2.set_title("CAD e Armazenamento (ARM)")
    ax2.grid(axis='y', linestyle='--', alpha=0.5); ax2.legend()
    st.pyplot(fig2)

    st.subheader("📈 P, ETo e ETR")
    fig3, ax3 = plt.subplots(figsize=(10,5))
    ax3.plot(MESES, df_out["Precipitação (mm)"].to_numpy(float), label="P (mm)")
    ax3.plot(MESES, df_out["ETo (Thornthwaite) (mm)"].to_numpy(float), label="ETo (mm)")
    ax3.plot(MESES, df_out["ETR (mm)"].to_numpy(float), label="ETR (mm)")
    ax3.set_ylabel("mm"); ax3.set_title("Séries mensais — P, ETo e ETR")
    ax3.grid(axis='y', linestyle='--', alpha=0.5); ax3.legend()
    st.pyplot(fig3)
else:
     st.info(
        "Preencha latitude, longitude e CAD. Depois clique em **Calcular balanço**.\n"
        "Dica: no primeiro acesso pode demorar ~1-2 min para baixar os dados base."
    )
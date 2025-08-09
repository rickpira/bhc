# app.py — Aplicativo para Cálculo do Balanço Hídrico Climatológico — v1
# Professores: Claudio Ricardo da Silva (UFU) e Nildo da Silva Dias (UFERSA)

import streamlit as st
import numpy as np
import pandas as pd
import math, os, io, zipfile, tempfile, hashlib, requests
import matplotlib.pyplot as plt

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
    st.session_state.res = None  # guardará resultados do cálculo

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
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code == 404:
            st.error("Arquivo da release não encontrado (404). Verifique o nome do asset.")
            r.raise_for_status()
        r.raise_for_status()
        with open(out, "wb") as f:
            for chunk in r.iter_content(chunk_size=262144):
                if chunk: f.write(chunk)
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

def _localizar_tif(pasta: str, prefix: str, m: int):
    """Procura .tif do mês m (01..12) em QUALQUER subpasta."""
    alvo_mm = f"_{m:02d}.tif".lower()
    pfx = prefix.lower()
    for root, _, files in os.walk(pasta):
        for fn in files:
            fn_low = fn.lower()
            if fn_low.endswith(alvo_mm) and pfx in fn_low and fn_low.endswith(".tif"):
                return os.path.join(root, fn)
    return None

def ler_serie(prefix: str, pasta: str, lon: float, lat: float):
    """Lê 12 rasters mensais (01..12) para o `prefix`, em qualquer subpasta."""
    import rasterio
    vals, faltantes = [], []
    for m in range(1,13):
        path = _localizar_tif(pasta, prefix, m)
        if path is None:
            faltantes.append(m); continue
        with rasterio.open(path) as src:
            r, c = src.index(lon, lat)  # x=lon, y=lat (WGS84)
            vals.append(float(src.read(1)[r, c]))
    if faltantes:
        candidatos = []
        for root, _, files in os.walk(pasta):
            for fn in files:
                if fn.lower().endswith(".tif"):
                    candidatos.append(os.path.join(root, fn))
        candidatos = sorted(candidatos)[:10]
        raise FileNotFoundError(
            f"Faltaram os meses {faltantes} para prefixo '{prefix}'.\n"
            "Exemplos de TIFs encontrados:\n- " + "\n- ".join(candidatos)
        )
    return np.array(vals, dtype=float)

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
            delta_eq = CAD * math.log(max(ARM, 1e-9) / CAD)
            delta_eq += d
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

# ===== Classificação Thornthwaite =====
def class_thornthwaite(df, lat):
    soma_ETo = float(df["ETo (Thornthwaite) (mm)"].sum())
    soma_EXC = float(df["EXC (mm)"].sum())
    soma_DEF = float(df["DEF (mm)"].sum())

    if soma_ETo <= 0:
        Ih = Ia = Im = np.nan
    else:
        Ih = 100.0 * (soma_EXC / soma_ETo)
        Ia = 100.0 * (soma_DEF / soma_ETo)
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

    DEF = df["DEF (mm)"].to_numpy(float)
    EXC = df["EXC (mm)"].to_numpy(float)
    is_summer = np.isin(np.arange(12), [9,10,11,0,1,2]) if lat < 0 else np.isin(np.arange(12), [3,4,5,6,7,8])

    def saz(u_code, Ia, Ih, DEF, EXC, is_summer):
        if u_code in ["A","B4","B3","B2","B1","C2"]:
            Ia = 100.0 * (DEF.sum() / soma_ETo) if soma_ETo>0 else np.nan
            if np.isnan(Ia) or Ia < 16.7: return ("r","baixo déficit ao longo do ano")
            Ia_sev = "2" if Ia >= 33.3 else ("1" if Ia >= 16.7 else "")
            def_su = DEF[is_summer].sum(); def_in = DEF[~is_summer].sum()
            return (f"s{Ia_sev}","déficit maior no verão") if def_su >= def_in else (f"w{Ia_sev}","déficit maior no inverno")
        else:
            Ih = 100.0 * (EXC.sum() / soma_ETo) if soma_ETo>0 else np.nan
            if np.isnan(Ih) or Ih < 10: return ("d","baixo excedente ao longo do ano")
            Ih_sev = "2" if Ih >= 33.3 else ("1" if Ih >= 10 else "")
            exc_su = EXC[is_summer].sum(); exc_in = EXC[~is_summer].sum()
            return (f"s{Ih_sev}","excedente maior no verão") if exc_su >= exc_in else (f"w{Ih_sev}","excedente maior no inverno")
    saz_code, saz_desc = saz(u_code, None, None, DEF, EXC, is_summer)

    # SCTE — texto ajustado:
    ETo = df["ETo (Thornthwaite) (mm)"].to_numpy(float)
    PET = soma_ETo
    SCTE = np.nan if PET<=0 else 100.0 * (np.sort(ETo)[-3:].sum()/PET)
    def scte(x):
        if np.isnan(x): return ("","indefinida")
        if x < 48.0:   return ("a’","evapotranspiração relativamente baixa no verão")
        if x <= 51.9:  return ("b’4","quase uniforme (~50%)")
        if x < 56.3:   return ("b’3","ligeiramente verão")
        if x < 61.6:   return ("b’2","moderadamente verão")
        if x < 68.0:   return ("b’1","verão marcado")
        if x < 76.3:   return ("c’2","verão muito marcado")
        if x < 88.0:   return ("c’1","verão dominante")
        return ("d’","PET muito concentrada no verão")
    scte_code, scte_desc = scte(SCTE)

    # Térmico (por PET anual)
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

    formula = f"{u_code} {saz_code} {t_code} {scte_code}"
    descricao = f"Clima {u_desc.lower()}, {t_desc.lower()}, com {saz_desc}; {scte_desc} (SCTE≈{SCTE:.1f}%)."
    return {"formula": formula, "descricao": descricao, "Ih": Ih, "Ia": None, "Im": None, "PET_anual": PET}

# ===== Classificação Köppen–Geiger =====
def class_koppen(df, lat):
    T = df["Temperatura Média (°C)"].to_numpy(float)
    P = df["Precipitação (mm)"].to_numpy(float)
    Tmed_anual = float(T.mean())
    Ptotal = float(P.sum())
    Tmin = float(T.min()); Tmax = float(T.max())

    verao = ["Dez","Jan","Fev"] if lat < 0 else ["Jun","Jul","Ago"]
    inverno = ["Jun","Jul","Ago"] if lat < 0 else ["Dez","Jan","Fev"]
    chuva_verao = float(df.loc[df["Mês"].isin(verao), "Precipitação (mm)"].sum())
    chuva_inverno = float(df.loc[df["Mês"].isin(inverno), "Precipitação (mm)"].sum())

    if chuva_verao >= 2*chuva_inverno:
        Pth = 20*Tmed_anual + 280
    elif chuva_inverno >= 2*chuva_verao:
        Pth = 20*Tmed_anual
    else:
        Pth = 20*Tmed_anual + 140

    if Ptotal < Pth:
        seco_tipo = "W" if Ptotal < 0.5*Pth else "S"
        hk = "h" if Tmed_anual >= 18 else "k"
        formula = f"B{seco_tipo}{hk}"
        desc = "Desértico" if seco_tipo=="W" else "Semiárido"
        desc += " quente" if hk=="h" else " frio"
        return {"formula": formula, "descricao": f"Clima {desc}."}

    if Tmin >= 18:
        letra1 = "A"
    elif Tmax > 10 and Tmin > 0:
        letra1 = "C"
    else:
        letra1 = "D"

    if (chuva_inverno < 40) and (chuva_inverno < 0.33*chuva_verao):
        letra2 = "w"
    elif (chuva_verao < 0.33*chuva_inverno):
        letra2 = "s"
    else:
        letra2 = "f"

    if letra1 in ["C","D"]:
        if Tmax >= 22:
            letra3 = "a"
        elif np.sum(T >= 10) >= 4:
            letra3 = "b"
        else:
            letra3 = "c"
    else:
        letra3 = ""

    formula = letra1 + letra2 + letra3
    legendas = {
        "Af":"Tropical úmido, sem estação seca","Am":"Tropical monçônico",
        "Aw":"Tropical com inverno seco","As":"Tropical com verão seco",
        "Cfa":"Temperado úmido, verão quente","Cfb":"Temperado úmido, verão morno","Cfc":"Temperado úmido, verão fresco",
        "Cwa":"Temperado com inverno seco, verão quente","Cwb":"Temperado com inverno seco, verão morno","Cwc":"Temperado com inverno seco, verão fresco",
        "Dfa":"Continental úmido, verão quente","Dfb":"Continental úmido, verão morno","Dfc":"Continental úmido, verão fresco",
    }
    descricao = legendas.get(formula, "Clima conforme Köppen–Geiger.")
    return {"formula": formula, "descricao": descricao}

# ============== CÁLCULO (apenas quando clicar) ==============
if calcular:
    try:
        zip_path = download_from_github(GITHUB_ZIP_URL)
        pasta = unzip_cached(zip_path)

        # Leia T e P do único ZIP (procura em subpastas)
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
        kopp  = class_koppen(df_out, lat)

        # guarda tudo no estado (persistência após rerun do download)
        st.session_state.res = {
            "df_out": df_out, "inicio_ARM": inicio_ARM,
            "thorn": thorn, "kopp": kopp,
            "CAD": CAD, "lat": lat, "lon": lon
        }

    except Exception as e:
        st.error(f"Erro: {e}")

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
    with c2:
        st.markdown(f"**🌍 Köppen–Geiger — Fórmula:** `{kopp['formula']}`")
        st.markdown(f"**Descrição:** {kopp['descricao']}")

    # Tabela
    st.subheader("📋 Tabela do Balanço Hídrico")
    st.dataframe(df_out, use_container_width=True)
    st.caption(f"🔁 Início do ciclo de ARM (após bloco úmido): **{MESES[inicio_ARM]}** — CAD={CAD:.0f} mm")

    # Download Excel (persiste após rerun)
    import io
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
                    ["Köppen - Fórmula", info.get("koppen_formula")],
                    ["Köppen - Descrição", info.get("koppen_desc")],
                    ["Gerado em", datetime.now().strftime("%Y-%m-%d %H:%M")],
                ], columns=["Campo","Valor"])
                meta.to_excel(wr, index=False, sheet_name="Resumo")
            wr.close()
        buf.seek(0)
        return buf.getvalue()

    info_excel = {
        "lat": lat, "lon": lon, "CAD": CAD,
        "thorn_formula": thorn["formula"], "thorn_desc": thorn["descricao"],
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
    st.info("Preencha latitude, longitude e CAD. Depois clique em **Calcular balanço**.")

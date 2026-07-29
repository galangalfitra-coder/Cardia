"""Cek Risiko Jantung - Streamlit demo skripsi (model: CatBoost, best_catboost_model.pkl).
Bukan diagnosis medis - lihat main.py untuk training & evaluasi lengkap.

Struktur: Home (landing + 2 pintu navigasi) -> Edukasi | Prediksi.
Navigasi pakai st.session_state, tetap satu file (app.py)."""

import pickle

import pandas as pd
import streamlit as st

from model_transformers import CholesterolZeroImputer  # noqa: F401

st.set_page_config(page_title="Cardia", page_icon="\u2695", layout="wide")

MODEL_PATH = "best_catboost_model.pkl"

# ---- Design tokens v2 (krem base tetap, tapi ditambah aksen "electric coral" + glow) ----
BG = "#F5EBDD"
CARD = "#FCF8F3"
CARD_BORDER = "#E6D8CA"
TEXT_DARK = "#241C14"
TEXT_MUTED = "#6B6255"
INPUT_BORDER = "#DCD2BF"
INPUT_TEXT = "#2E2A24"
BUTTON = "#FF5E3A"           # electric coral, lebih pop dari coral lama
BUTTON_HOVER = "#E8481F"
DOT_DATA_DIRI = "#A78BFA"
DOT_JANTUNG = "#F0A6A6"
DOT_AKTIVITAS = "#9BD3C0"
DISCLAIMER_BG = "#FBF0D2"
DISCLAIMER_BORDER = "#F0DFA0"
DISCLAIMER_TEXT = "#7A5E14"
RESULT_TRACK = "#EFE7D8"

# Aksen signature: dipakai buat ECG line di hero + glow di elemen highlight
ACCENT = "#FF5E3A"
ACCENT_2 = "#7B5CF0"         # lavender-electric, pasangan duo-tone si coral
GLOW = "rgba(255,94,58,.25)"

# Warna tematik dua "pintu" navigasi di Home
NAV_EDUKASI = "#000000"
NAV_PREDIKSI = "#FF5E3A"     # coral - warna aksi utama (tombol prediksi)
NAV_PREDIKSI_BG = "#FF5E3A"

RISK_TIERS = {
    "Rendah": {
        "bg": "#E4F3E6", "border": "#BFE6C6", "text": "#2F8F5D",
        "narrative": "Kabar baik! Berdasarkan data yang kamu isi, risiko kamu tergolong "
                     "rendah. Tetap jaga pola makan dan rutin bergerak ya, biar kondisi "
                     "jantung kamu terus prima.",
    },
    "Sedang": {
        "bg": "#FBF0D2", "border": "#F0DFA0", "text": "#9C7A12",
        "narrative": "Risikonya ada di level menengah nih. Nggak perlu panik, tapi ada "
                     "baiknya kamu mulai lebih perhatian sama pola hidup — dan nggak ada "
                     "salahnya ngobrol sama dokter buat pengecekan lebih lanjut.",
    },
    "Tinggi": {
        "bg": "#FBE0DE", "border": "#F3C4C0", "text": "#C4453A",
        "narrative": "Angkanya menunjukkan risiko yang cukup tinggi. Ini bukan vonis, "
                     "tapi sinyal buat kamu segera konsultasi ke tenaga medis biar dapat "
                     "pemeriksaan yang lebih akurat.",
    },
}


def risk_tier(pct: float):
    key = "Tinggi" if pct >= 60 else "Sedang" if pct >= 30 else "Rendah"
    t = RISK_TIERS[key]
    return key, t["bg"], t["border"], t["text"], t["narrative"]


# ============================================================
# STYLES (global, dipakai di semua halaman)
# ============================================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@700;800&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
    .stApp {{ background: {BG}; }}
    *, *:focus, *:focus-visible, *:focus-within {{ outline: none !important; }}

    /* ---- Responsive viewport ---- */
    /* ---- Constrain main content width (desktop bisa lebar banget karena layout=wide + sidebar) ---- */
    .stMainBlockContainer {{ max-width: 900px !important; margin: 0 auto !important; }}

    @media (max-width: 768px) {{
      .stMainBlockContainer {{ max-width: 100% !important; padding: 0.5rem 0.85rem !important; }}
      .hero-title {{ font-size: 1.5rem !important; }}
      .hero-sub {{ font-size: 0.85rem !important; }}
      /* nav-card di Home: min-height + margin-top:auto itu cuma buat nyamain tinggi
         2 kartu pas berdampingan (desktop). Di mobile kartu ke-stack vertikal, jadi
         rule itu cuma bikin "lubang" kosong sebelum tombol -- dimatiin di sini. */
      div[class*="st-key-navcard_"] {{ min-height: unset !important; }}
      div[class*="st-key-navcard_"] .stButton {{ margin-top: 1rem !important; }}
    }}

    /* ---- Top brand bar ---- */
    .brand-bar {{
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.25rem 0 1rem 0; margin-bottom: 0.25rem;
        border-bottom: 1px solid {CARD_BORDER};
    }}
    .brand-mark {{
        font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.05rem;
        color: {TEXT_DARK};
    }}
    .brand-crumb {{ font-size: 0.85rem; color: {TEXT_MUTED}; }}

    /* ---- Signature element: garis ECG yang jalan terus di belakang hero ---- */
    @keyframes ecg-scroll {{ 0% {{ background-position: 0 0; }} 100% {{ background-position: -600px 0; }} }}
    @keyframes ecg-pulse {{ 0%, 100% {{ opacity: .35; }} 50% {{ opacity: .9; }} }}
    .ecg-line {{
        height: 34px; margin: 0 auto 0.35rem auto; max-width: 26rem;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='300' height='34' viewBox='0 0 300 34'%3E%3Cpolyline points='0,17 40,17 52,4 64,30 76,17 130,17 142,6 150,28 158,17 300,17' fill='none' stroke='%23FF5E3A' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E");
        background-repeat: repeat-x; background-size: 300px 34px;
        animation: ecg-scroll 3.2s linear infinite, ecg-pulse 1.6s ease-in-out infinite;
        opacity: .8;
    }}

    /* ---- Hero (dipakai di semua halaman) ---- */
    .hero-wrap {{ text-align: center; padding: 0.75rem 0 0.25rem 0; }}
    .hero-title {{
        font-family: 'Sora', sans-serif; font-weight: 800; font-size: 2.75rem;
        color: {TEXT_DARK}; letter-spacing: -0.03em; line-height: 1.05;
    }}
    .hero-title .accent {{ color: {ACCENT}; }}
    .hero-sub {{
        font-size: 1.02rem; color: {TEXT_MUTED}; line-height: 1.6;
        max-width: 30rem; margin: 0.6rem auto 0 auto; font-weight: 500;
    }}
    .disclaimer-box {{
        background: {DISCLAIMER_BG}; border: 1px solid {DISCLAIMER_BORDER};
        border-radius: 12px; padding: 0.9rem 1rem; color: {DISCLAIMER_TEXT};
        font-size: 0.85rem; line-height: 1.55; margin: 1.1rem 0 1.25rem 0;
    }}

    /* ---- Card container generik (form, edukasi, nav) ----
       Streamlit versi lo gak punya testid stVerticalBlockBorderWrapper lagi,
       jadi kita target lewat class st-key-* yang muncul dari param key=... di
       st.container() -- ini cara resmi & stabil buat styling container spesifik.
       Rule ini cuma visual dasar (background/border/radius/shadow) -- berlaku
       ke SEMUA card termasuk nav-card, makanya gak ada min-height/flex di sini,
       biar card Edukasi & Prediksi tetep auto-height ngikut konten. */
    div[class*="st-key-card_"], div[class*="st-key-navcard_"] {{
        background-color: {CARD};
        border: 1px solid {CARD_BORDER};
        border-radius: 18px;
        margin-bottom: 1rem;
        box-shadow: 0 3px 12px rgba(0,0,0,.03);
    }}
    /* Nav-card Home (2 pintu navigasi) doang yang butuh tinggi disamain +
       tombol didorong ke dasar -- makanya di-scope khusus ke prefix navcard_,
       BUKAN ke semua card_* (itu penyebab card Edukasi/Data-Diri kemarin jadi
       punya ruang kosong nganggur di bawah teks). */
    div[class*="st-key-navcard_"] {{
        min-height: 175px;
        display: flex;
        flex-direction: column;
    }}
    div[class*="st-key-navcard_"]:hover {{
        transform: translateY(-4px);
        box-shadow: 0 14px 28px -10px {GLOW};
        border-color: {ACCENT};
    }}
    .card-header {{
        display: flex; align-items: center; gap: 0.5rem;
        font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1.05rem;
        color: {TEXT_DARK}; margin: 0 0 1.1rem 0;
    }}
    .card-header .dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
    .dot-data-diri {{ background: {DOT_DATA_DIRI}; }}
    .dot-jantung {{ background: {DOT_JANTUNG}; }}
    .dot-aktivitas {{ background: {DOT_AKTIVITAS}; }}

    /* ---- Nav card (Home: 2 pintu navigasi) ----
       Icon sejajar (horizontal) dengan title lewat .nav-card-header flex row,
       biar icon jadi pendamping judul -- bukan elemen berdiri sendiri di atas. */
    .nav-card-header {{
        display: flex;
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.6rem;
    }}
    .nav-card-icon{{
        display:flex;
        align-items:center;
        justify-content:center;
        flex-shrink:0;

        background:transparent;
        border-radius:0;

        box-shadow:none;
    }}
    /* paksa semua SVG icon di dalam nav-card-icon jadi ukuran seragam & proporsional
       buat versi horizontal, nge-override attribute width/height hardcoded di tiap <svg> */
    .nav-card-icon svg {{
        width: 54px !important;
        height: 54px !important;
    }}
    .nav-card-header .nav-card-title {{
        margin-bottom: 0;
    }}
    .nav-card-title {{
        font-family: 'Sora', sans-serif; font-weight: 800; font-size: 1.2rem;
        color: {TEXT_DARK}; margin-bottom: 0.4rem; letter-spacing: -0.01em;
    }}
    .nav-card-desc {{
        font-size: 0.88rem; color: {TEXT_MUTED}; line-height: 1.55; min-height: 72px; margin-bottom: 1.1rem;
        min-height: 3.3rem;
    }}
    /* tombol nav-card selalu sejajar di dasar card, biar kedua card kelihatan
       sama tinggi walau panjang teks desc beda -- di-scope ke navcard_ doang,
       biar tombol di card Edukasi/Prediksi biasa gak ikut kena margin-top:auto. */
    div[class*="st-key-navcard_"] .stButton {{
        margin-top: auto;
    }}

    /* ---- Edukasi: kartu faktor risiko ---- */
    .risk-pill {{
        display: inline-block; padding: 0.3rem 0.7rem; border-radius: 999px;
        font-size: 0.8rem; font-weight: 600; margin: 0.2rem 0.3rem 0.2rem 0;
    }}
    .edu-list {{ margin: 0.5; padding-left: 0.1rem; color: {TEXT_DARK}; line-height: 1.8; font-size: 0.92rem; }}
    .edu-text {{ margin: 0.5; color: {TEXT_DARK}; line-height: 1.8; font-size: 0.92rem; text-align: justify; text-justify: inter-word; }}

    label {{ color: {TEXT_DARK} !important; opacity: 1 !important; font-weight: 600 !important; font-size: 0.88rem !important; }}

    /* number input: reset wrapper bawaan lalu styling ulang input & tombol +/- */
    div[data-testid="stNumberInput"],
    div[data-testid="stNumberInput"] > div,
    div[data-testid="stNumberInput"] > div > div,
    div[data-testid="stNumberInput"] > div > div > div {{
        border: none !important; box-shadow: none !important; background: transparent !important;
    }}
    .stNumberInput input {{
        background: #FFFFFF !important; color: {INPUT_TEXT} !important;
        border: 1px solid {INPUT_BORDER} !important; border-radius: 12px !important;
        box-shadow: none !important; -webkit-appearance: none !important; appearance: none !important;
    }}
    .stNumberInput input:focus {{
        border: 1px solid {BUTTON} !important; box-shadow: 0 0 0 3px rgba(226,130,95,.15) !important;
    }}
    .stNumberInput button {{ display: none !important; border: 1px solid {INPUT_BORDER} !important; color: {INPUT_TEXT} !important; }}

    /* selectbox: kotak tertutup */
    div[data-baseweb="select"] > div {{
        background: #FFFFFF !important; border: 1px solid {INPUT_BORDER} !important;
        border-radius: 12px !important; box-shadow: none !important;
    }}
    div[data-baseweb="select"]:focus-within > div {{
        border-color: {BUTTON} !important; box-shadow: 0 0 0 3px rgba(226,130,95,.15) !important;
    }}
    div[data-baseweb="select"] > div * {{ color: {INPUT_TEXT} !important; opacity: 1 !important; }}

    /* selectbox: dropdown menu saat terbuka */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    div[data-baseweb="menu"] {{
        border: none !important; box-shadow: none !important; outline: none !important;
    }}
    ul[data-testid="stSelectboxVirtualDropdown"],
    [role="listbox"] {{
        background: #FFFFFF !important; border: 1px solid {CARD_BORDER} !important;
        border-radius: 12px !important; box-shadow: 0 8px 24px rgba(0,0,0,.08) !important;
        outline: none !important; overflow: hidden !important;
    }}
    [role="option"],
    [role="option"] * {{
        background: transparent !important; background-image: none !important;
        -webkit-background-clip: initial !important; background-clip: initial !important;
        color: {INPUT_TEXT} !important; opacity: 1 !important;
        -webkit-text-fill-color: {INPUT_TEXT} !important;
        text-shadow: none !important; filter: none !important; border: none !important; margin: 0 !important;
    }}
    [role="option"] {{ border-radius: 8px !important; padding: 0.55rem 0.75rem !important; transition: background-color 0.15s ease; }}
    [role="option"]:hover, [role="option"]:hover * {{ background: #F3ECDC !important; }}
    [aria-selected="true"], [aria-selected="true"] * {{
        background: #FBEADD !important; color: {BUTTON} !important;
        -webkit-text-fill-color: {BUTTON} !important; font-weight: 600 !important;
    }}

    /* st.warning / st.expander native */
    div[data-testid="stAlert"], div[data-testid="stAlert"] * {{ color: {TEXT_DARK} !important; opacity: 1 !important; }}
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * {{ color: {TEXT_DARK} !important; opacity: 1 !important; }}
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"],
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] * {{ color: {TEXT_DARK} !important; opacity: 1 !important; }}

    .stButton > button {{
        background: {BUTTON} !important; color: #FFFFFF !important; border: none !important; border-radius: 14px !important;
        font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1rem;
        padding: 0.85rem 1rem; transition: background 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 6px 16px -6px {GLOW};
    }}
    .stButton > button:hover {{
        background: {BUTTON_HOVER} !important; color: #FFFFFF !important;
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 22px -6px {GLOW};
    }}
    .stButton > button:active {{ transform: translateY(0) scale(0.99); }}

    /* tombol sekunder (kembali / navigasi ringan) */
    .stButton > button[kind="secondary"] {{
        background: transparent; color: {TEXT_DARK}; border: 1px solid {INPUT_BORDER};
    }}
    .stButton > button[kind="secondary"]:hover {{ background: #F1E9D8; color: {TEXT_DARK}; }}

    .result-card {{
        border-radius: 16px; padding: 1.75rem; display: flex; flex-direction: column;
        align-items: center; gap: 0.85rem; text-align: center; margin-top: 0.75rem;
    }}
    .result-pct {{ font-family: 'Sora', sans-serif; font-weight: 800; font-size: 3.2rem; }}
    .result-badge {{ background: #FFFFFF; border-radius: 999px; padding: 0.4rem 1rem; font-size: 0.85rem; font-weight: 700; }}
    .result-bar-track {{
        width: 100%; max-width: 26rem; height: 12px; background: {RESULT_TRACK};
        border-radius: 999px; overflow: hidden; margin-top: 0.25rem;
    }}
    .result-bar-fill {{ height: 100%; border-radius: 999px; transition: width 0.6s ease; }}
    .result-narrative {{ font-size: 0.92rem; color: #5C5346; line-height: 1.6; max-width: 28rem; margin-top: 0.3rem; }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        payload = pickle.load(f)
    return payload["pipeline"], payload["threshold"], payload["feature_names"], payload["reported_metrics"]


# ---- Mapping teks UI -> angka model (diverifikasi dari dataset asli) ----
SEX_MAP = {"Pria": 1, "Wanita": 0}
CHEST_PAIN_MAP = {
    "Nyeri Dada Khas (Terasa Tertekan/Sesak saat Aktivitas)": 1,
    "Nyeri Dada Tidak Biasa": 2,
    "Nyeri Dada Ringan (Bukan Angina)": 3,
    "Tidak Ada Nyeri Dada": 4,
}
FBS_MAP = {"Normal (< 120 mg/dL)": 0, "Tinggi (≥ 120 mg/dL)": 1}
ECG_MAP = {"Normal": 0, "Ada Kelainan Gelombang Jantung": 1, "Pembesaran Jantung Kiri": 2}
ANGINA_MAP = {"Tidak": 0, "Ya": 1}
ST_SLOPE_MAP = {"Menanjak (Naik)": 1, "Datar": 2, "Menurun (Turun)": 3}


# ============================================================
# ROUTING
# ============================================================
if "page" not in st.session_state:
    st.session_state.page = "home"
if "nav_stack" not in st.session_state:
    st.session_state.nav_stack = []  # riwayat halaman yang dilewati, buat tombol "Kembali"


def go_to(page_name: str):
    if page_name != st.session_state.page:
        st.session_state.nav_stack.append(st.session_state.page)
    st.session_state.page = page_name


def go_back():
    if st.session_state.nav_stack:
        st.session_state.page = st.session_state.nav_stack.pop()
    else:
        st.session_state.page = "home"


def render_back_button():
    if st.session_state.nav_stack:
        st.button("\u2190 Kembali", key=f"back_{st.session_state.page}",
                  type="secondary", on_click=go_back)


def render_sidebar_nav():
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:Sora,sans-serif; font-weight:800; font-size:1.1rem; '
            f'letter-spacing:-0.02em; color:{TEXT_DARK}; margin-bottom:1.25rem;">'
            f'\u2695CARDIA</div>',
            unsafe_allow_html=True,
        )
        nav_items = [("home", "Beranda"), ("edukasi", "Edukasi Penyakit Jantung"), ("prediksi", "Mesin Prediksi")]
        for page_key, label in nav_items:
            is_active = st.session_state.page == page_key
            st.button(
                label, key=f"sidebar_{page_key}", use_container_width=True,
                type="primary" if is_active else "secondary",
                on_click=go_to, args=(page_key,),
            )


# ============================================================
# HALAMAN: HOME
# ============================================================
def render_home():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="ecg-line"></div>
            <div class="hero-title">CAR<span class="accent">DIA</span></div>
            <div class="hero-sub">Cardiovascular Disease Analysis</div>
            <div class="hero-sub">Know your risk before it knows you🧡</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True, key="navcard_edukasi"):
            st.markdown(
                f"""
                <div class="nav-card-header">
                <div class="nav-card-icon" style="background:; color:{NAV_EDUKASI};"><svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="#FF5E3A" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-heart-icon lucide-book-heart"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H19a1 1 0 0 1 1 1v18a1 1 0 0 1-1 1H6.5a1 1 0 0 1 0-5H20"/><path d="M8.62 9.8A2.25 2.25 0 1 1 12 6.836a2.25 2.25 0 1 1 3.38 2.966l-2.626 2.856a.998.998 0 0 1-1.507 0z"/></svg></div>
                <div class="nav-card-title">Edukasi Penyakit Jantung</div>
                </div>
                <div class="nav-card-desc">Mulai dengan memahami kesehatan jantungmu sendiri</div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Mulai Belajar", key="nav_edukasi", use_container_width=True, type="primary",
                      on_click=go_to, args=("edukasi",))

    with col2:
        with st.container(border=True, key="navcard_prediksi"):
            st.markdown(
                f"""
                <div class="nav-card-header">
                <div class="nav-card-icon" style="background:; color:{NAV_PREDIKSI};"><svg xmlns="http://www.w3.org/2000/svg"
                    width="72"
                    height="72"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#FF5E3A"
                    stroke-width="1.8"
                    stroke-linecap="round"
                    stroke-linejoin="round">
                    <path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/>
                    <path d="M3.22 13H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/>
                </svg></div>
                <div class="nav-card-title">Cek Risiko Sekarang</div>
                </div>
                <div class="nav-card-desc">Lanjutkan dengan mengetahui estimasi risiko penyakit jantungmu</div>
                """,
                unsafe_allow_html=True,
            )
            st.button("Mulai Prediksi", key="nav_prediksi", use_container_width=True, type="primary",
                      on_click=go_to, args=("prediksi",))

    st.markdown(
        """
        <div class="disclaimer-box">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg> 
            <strong>Ini bukan diagnosis medis.</strong> Aplikasi ini dibuat untuk keperluan
            akademik dan hasil prediksinya cuma estimasi awal, bukan pengganti
            pemeriksaan dokter.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# HALAMAN: EDUKASI
# ============================================================
def render_edukasi():
    render_back_button()
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">Edukasi Penyakit Jantung <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-book-copy-icon lucide-book-copy"><path d="M5 7a2 2 0 0 0-2 2v11"/><path d="M5.803 18H5a2 2 0 0 0 0 4h9.5a.5.5 0 0 0 .5-.5V21"/><path d="M9 15V4a2 2 0 0 1 2-2h9.5a.5.5 0 0 1 .5.5v14a.5.5 0 0 1-.5.5H11a2 2 0 0 1 0-4h10"/></svg></div>
            <div class="hero-sub">Sebelum cek risiko, yuk kenalan dulu sama penyakit jantung
            koroner  apa itu, kenapa bisa terjadi, dan gimana cara mencegahnya.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="card_edu_intro"):
        st.markdown(
            '<div class="card-header"><span class="dot dot-jantung"></span> Apa Itu Penyakit Jantung Koroner?</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <p class="edu-text">
            Penyakit jantung koroner adalah kondisi ketika pembuluh darah yang memasok oksigen dan nutrisi ke otot jantung mengalami penyempitan atau penyumbatan akibat penumpukan plak lemak. Akibatnya, aliran darah ke jantung menjadi berkurang sehingga jantung tidak dapat bekerja secara optimal. Jika kondisi ini terus berlanjut, risiko terjadinya serangan jantung akan meningkat.
            </p>
            """,
            unsafe_allow_html=True,
        )

    with st.container(border=True, key="card_edu_risk"):
        st.markdown(
            '<div class="card-header"><span class="dot dot-data-diri"></span> Faktor Risiko</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Tidak dapat diubah**")
            st.markdown(
                """<ul class="edu-list">
                <li>Usia (risiko naik seiring bertambah usia)</li>
                <li>Jenis kelamin (pria cenderung berisiko lebih awal)</li>
                <li>Riwayat keluarga dengan penyakit jantung</li>
                </ul>""",
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown("**Dapat diubah**")
            st.markdown(
                """<ul class="edu-list">
                <li>Tekanan darah tinggi (hipertensi)</li>
                <li>Kadar kolesterol tinggi</li>
                <li>Merokok dan kurang aktivitas fisik</li>
                <li>Diabetes / gula darah tidak terkontrol</li>
                <li>Obesitas dan pola makan tidak sehat</li>
                </ul>""",
                unsafe_allow_html=True,
            )

    with st.container(border=True, key="card_edu_symptom"):
        st.markdown(
            '<div class="card-header"><span class="dot dot-aktivitas"></span> Gejala yang Perlu Diwaspadai</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """<ul class="edu-list">
            <li>Nyeri atau rasa tertekan di dada, terutama saat beraktivitas</li>
            <li>Sesak napas, bahkan saat aktivitas ringan</li>
            <li>Cepat lelah yang tidak biasa</li>
            <li>Jantung berdebar tidak teratur</li>
            <li>Nyeri menjalar ke lengan, rahang, atau punggung</li>
            </ul>""",
            unsafe_allow_html=True,
        )
        st.warning(
            "Kalau kamu mengalami gejala-gejala di atas, terutama nyeri dada mendadak "
            "yang tidak biasa, segera cari pertolongan medis. Jangan menunggu."
        )

    with st.container(border=True, key="card_edu_tips"):
        st.markdown(
            '<div class="card-header"><span class="dot dot-data-diri"></span> Tips Pencegahan</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """<ul class="edu-list">
            <li>Jaga pola makan rendah lemak jenuh, garam, dan gula</li>
            <li>Rutin berolahraga minimal 150 menit per minggu</li>
            <li>Berhenti merokok dan batasi konsumsi alkohol</li>
            <li>Kelola stres dengan baik</li>
            <li>Cek tekanan darah, kolesterol, dan gula darah secara berkala</li>
            </ul>""",
            unsafe_allow_html=True,
        )

    st.write("")
    st.button("Lanjut ke Cek Risiko \u2192", use_container_width=True, type="primary",
              on_click=go_to, args=("prediksi",))


# ============================================================
# HALAMAN: PREDIKSI
# ============================================================
def render_prediksi():
    pipeline, model_threshold, feature_names, reported_metrics = load_model()
    render_back_button()

    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">Cek Risiko Jantung
            <svg xmlns="http://www.w3.org/2000/svg" width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-heart-pulse-icon lucide-heart-pulse"><path d="M2 9.5a5.5 5.5 0 0 1 9.591-3.676.56.56 0 0 0 .818 0A5.49 5.49 0 0 1 22 9.5c0 2.29-1.5 4-3 5.5l-5.492 5.313a2 2 0 0 1-3 .019L5 15c-1.5-1.5-3-3.2-3-5.5"/><path d="M3.22 13H9.5l.5-1 2 4.5 2-7 1.5 3.5h5.27"/></svg>
            </div>
            <div class="hero-sub">Isi data kesehatan kamu pada formulir di bawah ini untuk mendapatkan estimasi risiko penyakit jantung berdasarkan model machine learning yang telah dikembangkan.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="disclaimer-box">
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="lucide lucide-triangle-alert-icon lucide-triangle-alert"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>
             <strong>Ini bukan diagnosis medis.</strong> Hasil di sini cuma estimasi
            berbasis machine learning buat referensi awal. Kalau kamu punya keluhan atau
            khawatir, tetap konsultasi ke dokter langsung.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True, key="card_pred_diri"):
        st.markdown('<div class="card-header"><span class="dot dot-data-diri"></span> Data Diri</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Usia (tahun)", min_value=20, max_value=90, value=None, step=1, placeholder="Contoh: 20")
        with c2:
            sex_label = st.selectbox("Jenis Kelamin", list(SEX_MAP.keys()), index=None, placeholder="Pilih jenis kelamin")

    with st.container(border=True, key="card_pred_jantung"):
        st.markdown('<div class="card-header"><span class="dot dot-jantung"></span> Kondisi Jantung &amp; Pembuluh Darah</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            resting_bp = st.number_input("Tekanan Darah saat Istirahat (mmHg)", min_value=80, max_value=210, value=None, step=1, placeholder="Contoh: 130")
            cholesterol = st.number_input("Kadar Kolesterol dalam Darah (mg/dL)", min_value=100, max_value=450, value=None, step=1, placeholder="Contoh: 220")
            fbs_label = st.selectbox("Kadar Gula Darah Puasa", list(FBS_MAP.keys()), index=None, placeholder="Pilih kadar gula darah")
        with c2:
            chest_pain_label = st.selectbox("Jenis Nyeri Dada yang Dirasakan", list(CHEST_PAIN_MAP.keys()), index=None, placeholder="Pilih jenis nyeri dada")
            ecg_label = st.selectbox("Hasil Rekam Jantung (EKG) saat Istirahat", list(ECG_MAP.keys()), index=None, placeholder="Pilih hasil EKG")

    with st.container(border=True, key="card_pred_aktivitas"):
        st.markdown('<div class="card-header"><span class="dot dot-aktivitas"></span> Respons saat Aktivitas Fisik</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            max_hr = st.number_input("Detak Jantung Maksimum saat Aktivitas", min_value=60, max_value=210, value=None, step=1, placeholder="Contoh: 150")
            angina_label = st.selectbox("Apakah Merasakan Nyeri Dada saat Berolahraga?", list(ANGINA_MAP.keys()), index=None, placeholder="Pilih jawaban")
        with c2:
            oldpeak = st.number_input("Tingkat Stres Jantung saat Olahraga", min_value=-3.0, max_value=7.0, value=None, step=0.1, placeholder="Contoh: 1.0", help="0 = normal, makin tinggi makin berat")
            st_slope_label = st.selectbox("Pola Aktivitas Jantung saat Olahraga", list(ST_SLOPE_MAP.keys()), index=None, placeholder="Pilih pola aktivitas jantung")

    st.write("")
    predict_clicked = st.button("Prediksi Sekarang", use_container_width=True, type="primary")

    if predict_clicked:
        inputs = {
            "Usia": age, "Jenis Kelamin": sex_label, "Tekanan Darah": resting_bp,
            "Kolesterol": cholesterol, "Gula Darah": fbs_label,
            "Nyeri Dada": chest_pain_label, "EKG": ecg_label,
            "Detak Jantung Maks": max_hr, "Nyeri Saat Olahraga": angina_label,
            "Oldpeak": oldpeak, "ST Slope": st_slope_label,
        }
        kosong = [k for k, v in inputs.items() if v is None]
        if kosong:
            st.warning(f"Masih ada data yang belum diisi: {', '.join(kosong)}")
            st.stop()

        input_dict = {
            "age": age, "sex": SEX_MAP[sex_label], "chest pain type": CHEST_PAIN_MAP[chest_pain_label],
            "resting bp s": resting_bp, "cholesterol": cholesterol, "fasting blood sugar": FBS_MAP[fbs_label],
            "resting ecg": ECG_MAP[ecg_label], "max heart rate": max_hr, "exercise angina": ANGINA_MAP[angina_label],
            "oldpeak": oldpeak, "ST slope": ST_SLOPE_MAP[st_slope_label],
        }
        input_df = pd.DataFrame([input_dict])[feature_names]

        probability = pipeline.predict_proba(input_df)[0, 1]
        percentage = probability * 100
        tier_label, bg, border, text_color, narrative = risk_tier(percentage)
        fill_pct = max(2, min(100, percentage))

        st.markdown(
            f"""
            <div class="result-card" style="background:{bg}; border:1px solid {border};">
                <div class="result-pct" style="color:{text_color};">{percentage:.1f}%</div>
                <div class="result-badge" style="color:{text_color}; border:1px solid {border};">Risiko {tier_label}</div>
                <div class="result-bar-track"><div class="result-bar-fill" style="width:{fill_pct}%; background:{text_color};"></div></div>
                <div class="result-narrative">{narrative}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        with st.expander("Lihat detail input yang digunakan"):
            st.dataframe(input_df.T.rename(columns={0: "Nilai"}), use_container_width=True)

        st.warning(
            "**Ingat ya:** hasil ini bukan diagnosis dokter. Kalau ada keluhan atau khawatir, "
            "langsung ke dokter buat cek yang lebih proper."
        )

    st.divider()
    with st.expander("Informasi Model (untuk keperluan akademik)"):
        st.markdown(f"""
        - **Algoritma:** CatBoost Classifier
        - **Dataset:** Statlog Cleveland-Hungary Heart Disease (1190 baris, 11 fitur klinis)
        - **Skema pembagian data terbaik:** {reported_metrics['scheme']}
        - **Akurasi (data uji):** {reported_metrics['test_accuracy']*100:.2f}%
        - **Precision / Recall (data uji):** {reported_metrics['precision']*100:.2f}% / {reported_metrics['recall']*100:.2f}%
        - **ROC-AUC (data uji):** {reported_metrics['roc_auc']:.4f}
        - **Threshold klasifikasi:** {model_threshold:.3f}
        - **Catatan:** model dilatih ulang pada seluruh dataset untuk deployment; metrik di atas
          berasal dari evaluasi data uji terpisah, bukan dari model hasil retrain ini.
        """)


# ============================================================
# MAIN
# ============================================================
render_sidebar_nav()

if st.session_state.page == "home":
    render_home()
elif st.session_state.page == "edukasi":
    render_edukasi()
elif st.session_state.page == "prediksi":
    render_prediksi()
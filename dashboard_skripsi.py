import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os
import math
import plotly.graph_objects as go
import streamlit.components.v1 as components
from textwrap import dedent
import io
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from scipy.stats import shapiro
from scipy.stats import wilcoxon, norm
from scipy.stats import rankdata
import pingouin as pg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = "app_list.json"

if "app_list" not in st.session_state:
    st.session_state.app_list = []

    if os.path.exists(APP_FILE):
        try:
            st.session_state.app_list = pd.read_json(APP_FILE, typ="series").tolist()
        except:
            st.session_state.app_list = []

# Pastikan confirm_reset juga diinisialisasi di sini agar rapi
if "confirm_reset" not in st.session_state:
    st.session_state.confirm_reset = False

def interpret_ueq(score):
    if score > 1.5:
        return "Excellent"
    elif score > 0.8:
        return "Good"
    elif score > 0:
        return "Above Average"
    elif score > -0.8:
        return "Below Average"
    else:
        return "Bad"

def wilcoxon_full_spss(light, dark):

    light = pd.to_numeric(light, errors="coerce")
    dark = pd.to_numeric(dark, errors="coerce")

    mask = ~(light.isna() | dark.isna())
    light = light[mask]
    dark = dark[mask]

    diff = dark - light
    df = pd.DataFrame({"diff": diff})

    # buang ties
    df = df[df["diff"] != 0]

    df["abs"] = df["diff"].abs()
    df["rank"] = rankdata(df["abs"], method="average")

    negative = df[df["diff"] < 0]
    positive = df[df["diff"] > 0]

    n = len(df)

    # ======================
    # RANKS TABLE
    # ======================
    ranks_table = pd.DataFrame({
        "": ["Negative Ranks", "Positive Ranks", "Ties", "Total"],
        "N": [
            len(negative),
            len(positive),
            len(diff) - n,
            len(diff)
        ],
        "Mean Rank": [
            round(negative["rank"].mean(), 2) if len(negative) > 0 else 0,
            round(positive["rank"].mean(), 2) if len(positive) > 0 else 0,
            "",
            ""
        ],
        "Sum of Ranks": [
            round(negative["rank"].sum(), 2),
            round(positive["rank"].sum(), 2),
            "",
            ""
        ]
    })

    # ======================
    # HITUNG W
    # ======================
    W_pos = positive["rank"].sum()
    W_neg = negative["rank"].sum()
    T = W_neg

    # ======================
    # MEAN & SD + TIE CORRECTION
    # ======================
    mean_T = n * (n + 1) / 4

    ties_count = df["abs"].value_counts()
    tie_sum = np.sum(ties_count**3 - ties_count)

    var_T = (n * (n + 1) * (2*n + 1) - 0.5 * tie_sum) / 24
    sd_T = np.sqrt(var_T)

   
    

    

    # ======================
    # 🔥 P-VALUE DARI SCIPY (SUDAH BENAR)
    # ======================
    light_clean = light[df.index]
    dark_clean = dark[df.index]
    
    res = wilcoxon(
        light,
        dark,
        zero_method='wilcox',
        correction=False,   # 🔥 WAJIB
        alternative='two-sided',
        method='approx'
    )

    z = stats.norm.ppf(res.pvalue / 2) * (-1 if W_neg < W_pos else 1)
    p = res.pvalue

    W_pos = positive["rank"].sum()
    W_neg = negative["rank"].sum()
    W = min(W_pos, W_neg)

    mean_T = n * (n + 1) / 4

    ties_count = df["abs"].value_counts()
    tie_sum = np.sum(ties_count**3 - ties_count)

    var_T = (n * (n + 1) * (2*n + 1) - 0.5 * tie_sum) / 24
    sd_T = np.sqrt(var_T)

    # 🔥 SPSS correction logic (FIX)
    # SPSS secara default tidak menggunakan continuity correction untuk nilai Z
    correction = 0

    z = (W - mean_T + correction) / sd_T

    # p-value dari Z
    p = 2 * (1 - norm.cdf(abs(z)))

    stats_table = pd.DataFrame({
        "": ["Z", "Asymp. Sig (2-tailed)"],
        "Value": [round(z, 3), round(p, 3)]
    })

    return ranks_table, stats_table


def compute_wilcoxon_pair(light, dark, light_lbl, dark_lbl):
    """Compute Wilcoxon stats for one pair and return a display-ready dict."""
    ranks_table, stats_table = wilcoxon_full_spss(light, dark)

    z_val = float(stats_table.iloc[0, 1])
    p_val = float(stats_table.iloc[1, 1])

    neg_n   = int(ranks_table.iloc[0]["N"])
    pos_n   = int(ranks_table.iloc[1]["N"])
    ties_n  = int(ranks_table.iloc[2]["N"])
    total_n = int(ranks_table.iloc[3]["N"])

    neg_mean = ranks_table.iloc[0]["Mean Rank"]
    pos_mean = ranks_table.iloc[1]["Mean Rank"]
    neg_sum  = ranks_table.iloc[0]["Sum of Ranks"]
    pos_sum  = ranks_table.iloc[1]["Sum of Ranks"]

    def fmt(v, decimals=2):
        try:
            return f"{float(v):.{decimals}f}"
        except (TypeError, ValueError):
            return ""

    return {
        "var_name":  f"{dark_lbl} - {light_lbl}",
        "light_lbl": light_lbl,
        "dark_lbl":  dark_lbl,
        "neg_n": neg_n, "pos_n": pos_n, "ties_n": ties_n, "total_n": total_n,
        "neg_mean": fmt(neg_mean), "pos_mean": fmt(pos_mean),
        "neg_sum":  fmt(neg_sum),  "pos_sum":  fmt(pos_sum),
        "z_val": z_val, "p_val": p_val,
    }


def render_spss_wilcoxon(pairs_data):
    """
    Render output Wilcoxon identik dengan SPSS Style.
    Huruf (a-i) hanya muncul jika baris tersebut memiliki nilai N > 0.
    """
    ranks_rows = ""
    footnotes = []
    abc = "abcdefghijklmnopqrstuvwxyz"
    fn_idx = 0
    
    for pd_item in pairs_data:
        vn = pd_item["var_name"]
        l_lbl = pd_item["light_lbl"]
        d_lbl = pd_item["dark_lbl"]
        
        # Penampung huruf superscript untuk 3 kategori per Task
        labels = ["", "", ""] 
        
        # Kita definisikan arah hubungan secara statis sesuai urutan SPSS
        hubungan = [
            f"{d_lbl} < {l_lbl}", # Negative
            f"{d_lbl} > {l_lbl}", # Positive
            f"{l_lbl} = {d_lbl}"  # Ties
        ]
        
        # Cek data N untuk Negative, Positive, dan Ties
        n_vals = [pd_item['neg_n'], pd_item['pos_n'], pd_item['ties_n']]
        
        for i in range(3):
            # Huruf footnote harus selalu berlanjut (a, b, c...) mengikuti kategori,
            # tapi tampilannya dikontrol oleh nilai N.
            current_letter = abc[fn_idx]
            
            if n_vals[i] > 0:
                labels[i] = f"<sup>{current_letter}</sup>"
                footnotes.append(f"{current_letter}. {hubungan[i]}")
            
            fn_idx += 1 # Index huruf tetap naik agar urutan a-i konsisten
            
        ranks_rows += f"""
        <tr>
            <td rowspan="4" style="border:1px solid #bbb;padding:7px 12px;font-weight:600;
                background:#f5f5f5;vertical-align:middle;white-space:nowrap;">{vn}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;">Negative Ranks</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['neg_n']}{labels[0]}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['neg_mean']}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['neg_sum']}</td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb;padding:7px 12px;">Positive Ranks</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['pos_n']}{labels[1]}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['pos_mean']}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['pos_sum']}</td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb;padding:7px 12px;">Ties</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{pd_item['ties_n']}{labels[2]}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;"></td>
            <td style="border:1px solid #bbb;padding:7px 12px;"></td>
        </tr>
        <tr>
            <td style="border:1px solid #bbb;padding:7px 12px;font-weight:600;">Total</td>
            <td style="border:1px solid #bbb;padding:7px 12px;text-align:right;font-weight:600;">{pd_item['total_n']}</td>
            <td style="border:1px solid #bbb;padding:7px 12px;"></td>
            <td style="border:1px solid #bbb;padding:7px 12px;"></td>
        </tr>"""

    footnote_html = "<br>".join(footnotes)
    
    # Render gabungan tabel Ranks dan Test Statistics
    ranks_html = f"""
    <div style="margin:20px 0 8px 0;">
        <div style="font-weight:700;font-size:14px;border-bottom:2px solid #333;padding-bottom:4px;margin-bottom:0;">Ranks</div>
        <table style="border-collapse:collapse;font-size:13px;font-family:Arial,sans-serif;width:100%;">
            <thead>
                <tr style="background:#d9d9d9;">
                    <th colspan="2" style="border:1px solid #aaa;padding:7px 12px;"></th>
                    <th style="border:1px solid #aaa;padding:7px 12px;text-align:center;">N</th>
                    <th style="border:1px solid #aaa;padding:7px 12px;text-align:center;">Mean Rank</th>
                    <th style="border:1px solid #aaa;padding:7px 12px;text-align:center;">Sum of Ranks</th>
                </tr>
            </thead>
            <tbody>{ranks_rows}</tbody>
        </table>
        <div style="font-size:11px;color:#444;margin-top:5px;font-style:italic;line-height:1.6;">
            {footnote_html}
        </div>
    </div>"""
    
    # (Bagian Test Statistics tetap menggunakan format yang sudah benar sebelumnya)
    st.markdown(ranks_html, unsafe_allow_html=True)


def dataset_manager(df, expected_columns, save_path, title, filename_base):

    st.markdown("""
    <div style="
    font-size:16px;
    font-weight:600;
    color:#1e293b;
    margin-bottom:8px;
    ">
    Kelola Dataset
    </div>
    """, unsafe_allow_html=True)

    action = st.radio(
        "Pilih aksi",
        ["Export Dataset", "Import Dataset"],
        horizontal=True,
        key=f"dataset_action_{filename_base}"
    )

    # ======================
    # EXPORT
    # ======================
    if action == "Export Dataset":

        file_type = st.selectbox(
            "Pilih format file",
            ["Excel (.xlsx)","CSV (.csv)","PDF (.pdf)"],
            key=f"file_type_{filename_base}"
        )

        buffer = io.BytesIO()

        if file_type == "Excel (.xlsx)":

            with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
                df.to_excel(writer, index=False, sheet_name="Dataset")

            buffer.seek(0)

            st.download_button(
                "Download File",
                data=buffer,
                file_name=f"{filename_base}.xlsx"
            )

        elif file_type == "CSV (.csv)":

            csv = df.to_csv(index=False)

            st.download_button(
                "Download File",
                data=csv,
                file_name=f"{filename_base}.csv"
            )

        elif file_type == "PDF (.pdf)":

            doc = SimpleDocTemplate(buffer, pagesize=A4)
            styles = getSampleStyleSheet()

            elements = []
            elements.append(Paragraph(title, styles["Title"]))
            elements.append(Spacer(1,20))

            data = [df.columns.tolist()] + df.values.tolist()

            table = Table(data)

            table.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),colors.darkblue),
                ("TEXTCOLOR",(0,0),(-1,0),colors.white),
                ("ALIGN",(0,0),(-1,-1),"CENTER"),
                ("GRID",(0,0),(-1,-1),0.5,colors.grey),
                ("FONTSIZE",(0,0),(-1,-1),8)
            ]))

            elements.append(table)

            doc.build(elements)

            buffer.seek(0)

            st.download_button(
                "Download File",
                data=buffer,
                file_name=f"{filename_base}.pdf"
            )
    


    # ======================
    # IMPORT
    # ======================
    if action == "Import Dataset":

        uploaded_file = st.file_uploader(
            "Upload file dataset",
            type=["xlsx","csv"],
            key=f"upload_{filename_base}"
        )

        if uploaded_file is not None:

            if uploaded_file.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded_file)
            else:
                df_new = pd.read_csv(uploaded_file)

            if list(df_new.columns) != expected_columns:
                st.error("Struktur dataset tidak sesuai.")
                return

            df_new.to_csv(save_path, index=False)

            st.success("Dataset berhasil diimport.")
            st.rerun()
            



theme = st.get_option("theme.base")
if theme == "dark":
    plt.style.use("dark_background")
else:
    plt.style.use("default")
    
if theme == "dark":

    bg_main = "#020617"
    bg_card = "#0f172a"
    bg_sidebar = "#020617"
    bg_insight = "#1e293b"

    text_main = "#f1f5f9"
    text_soft = "#94a3b8"

    border = "#1e293b"

else:

    bg_main = "linear-gradient(135deg,#f8fafc,#eef2f7)"
    bg_card = "#ffffff"
    bg_sidebar = "#f8fafc"
    bg_insight = "#f1f5f9"

    text_main = "#111827"
    text_soft = "#6b7280"

    border = "#e5e7eb"

st.set_page_config(page_title="UX Research Dashboard", layout="wide")

# ======================
# CSS MODERN
# ======================

st.markdown("""
<style>

/* Sidebar Styling yang lebih clean */
[data-testid="stSidebar"]  {
    background-color: #ffffff;
    border-right: 1px solid #e5e7eb;
}

/* Header di Sidebar */
.sidebar-branding {
    padding: 4px 0;
    margin-bottom: 12px;
    border-bottom: 2px solid #111827;
}

.sidebar-title {
    font-size: 16px;
    font-weight: 800;
    color: #111827;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* Kategori Menu */
.menu-label {
    font-weight: 700;
    font-size: 8px;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 4px !important;
    margin-top: 6px !important;
}

/* Tombol Reset Minimalis */
.stButton > button {
    width: 100%;
    border-radius: 4px;
    font-size: 11px !important;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    transition: all 0.3s;
    padding: 6px !important;
}

[data-testid="stMetricV2"] {
    background-color: white;
    color: #1e293b;
}
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background-color: white !important;
}

.block-container {
    max-width: 1500px;
    padding-top: 70px;
}

/* Tambahkan ini di dalam <style> */
.stDataFrame, .stTable {
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e5e7eb;
}

.chart-container {
    padding: 10px;
    background: transparent;
}

.main-title {
    font-size: 28px;
    font-weight: 600;
    color: #111827;
}

.subtitle {
    color: #6b7280;
    margin-bottom: 30px;
}

.section-title {
    font-size: 18px;
    font-weight: 600;
    margin-top: 40px;
    margin-bottom: 15px;
}

.card {
    background: white;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid #f1f5f9;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.3s ease;
    height: 100%;
}

.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.08), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    border-color: #6366f1;
}

.metric-container {
    display: flex;
    flex-direction: column;
}

.metric-title {
    font-size: 14px;
    font-weight: 500;
    color: #64748b;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.metric-value {
    font-size: 26px;
    font-weight: 800;
    color: #1e293b;
    line-height: 1.2;
}

.metric-footer {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid #f8fafc;
}

.status-badge {
    display: inline-block;
    padding: 4px 8px;
    border-radius: 6px;
    font-size: 10px;
    font-weight: 700;
    margin-top: 5px;
}

/* Warna identitas untuk label */
.val-light {
    color: #6366f1;
    font-weight: 800;
}

.val-dark {
    color: #a78bfa;
    font-weight: 800;
}

.vs-divider {
    color: #94a3b8;
    font-size: 14px;
    font-weight: 400;
    margin: 0 4px;
}

.pref-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 15px;
    padding: 20px;
    text-align: center;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.pref-label {
    font-size: 12px;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    margin-bottom: 10px;
}
.pref-value {
    font-size: 20px;
    font-weight: 700;
    color: #111827;
}

h3 {
    font-size: 16px !important;
}

.p-card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

/* Kasih jarak normal sidebar */
[data-testid="stSidebar"] .block-container {
    padding-top: 5px !important;
    padding-bottom: 5px !important;
}

/* Biar tiap komponen ga nempel */
section[data-testid="stSidebar"] .stSelectbox,
section[data-testid="stSidebar"] .stNumberInput {
    margin-top: 6px;
    margin-bottom: 6px !important;
}
            
.sidebar-header h1 {
    font-size: 18px !important;
}

.sidebar-header p {
    font-size: 9px !important;
}

.sidebar-card {
    padding: 10px !important;
    font-size: 10px !important;
}
            
section[data-testid="stSidebar"] > div:first-child {
    height: 100vh;
    display: flex;
    flex-direction: column;
    
}
            
div[data-baseweb="select"] {
    margin-top: 4px;
}

/* Fix jarak label ke input */
label[data-testid="stWidgetLabel"] {
    margin-bottom: 4px !important;
}

/* Khusus sidebar selectbox */
section[data-testid="stSidebar"] .stSelectbox {
    margin-bottom: 10px;
}           

</style>
""", unsafe_allow_html=True)

# CSS override tema — blok f-string terpisah, hanya berisi aturan tema
st.markdown(f"""
<style>
[data-testid="stSidebar"] {{
    background: {bg_sidebar} !important;
    border-right: 1px solid {border} !important;
}}
[data-testid="stMetricV2"] {{
    background-color: {bg_card} !important;
    color: {text_main} !important;
}}
div[data-testid="stVerticalBlockBorderWrapper"] > div {{
    background-color: {bg_card} !important;
}}
.main-title {{ color: {text_main} !important; }}
.subtitle {{ color: {text_soft} !important; }}
.card {{
    background: {bg_card} !important;
    border-color: {border} !important;
    color: {text_main} !important;
}}
.card b, .card strong {{ color: {text_main} !important; }}
.card span {{ color: {text_soft} !important; }}
.card li {{ color: {text_main} !important; }}
.metric-title {{ color: {text_soft} !important; }}
.metric-value {{ color: {text_main} !important; }}
.pref-card {{
    background: {bg_card} !important;
    border-color: {border} !important;
    color: {text_main} !important;
}}
.pref-label {{ color: {text_soft} !important; }}
.pref-value {{ color: {text_main} !important; }}
.p-card {{
    background-color: {bg_card} !important;
    border-color: {border} !important;
    color: {text_main} !important;
}}
</style>
""", unsafe_allow_html=True)


# ========================================================
# 1. TEMPATKAN FUNGSI GLOBAL DI SINI (SETELAH IMPORT)
# ========================================================
def create_donut_chart(data_dict, colors):
    if not data_dict: 
        return None
    fig = go.Figure(data=[go.Pie(
        labels=list(data_dict.keys()),
        values=list(data_dict.values()),
        hole=.6,
        marker=dict(colors=colors, line=dict(color='#FFFFFF', width=2)),
        textinfo='none', 
        showlegend=False, 
        hoverinfo='label+percent'
    )])
    fig.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        height=160,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    return fig

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; padding-bottom: 20px;">
            <h1 style='font-size: 22px; color: #6366f1; margin-bottom: 0;'>🚀 UX Analytics</h1>
            <p style='font-size: 11px; color: #94a3b8; text-transform: uppercase; letter-spacing: 1.5px;'>Research Platform</p>
        </div>
    """, unsafe_allow_html=True)

    # --- TAB MANAJEMEN APLIKASI ---
    st.markdown('<p class="menu-label">Manage Objects</p>', unsafe_allow_html=True)
    
    with st.expander("➕ Tambah/Hapus Aplikasi", expanded=len(st.session_state.app_list) == 0):

    # TAMBAH
        new_app = st.text_input("Nama Aplikasi Baru", placeholder="Contoh: TikTok")

        if st.button("➕ Tambah ke List", use_container_width=True):
            if new_app and new_app.strip() not in st.session_state.app_list:
                st.session_state.app_list.append(new_app.strip())
                pd.Series(st.session_state.app_list).to_json(APP_FILE)
                st.rerun()

        # HAPUS
        if st.session_state.app_list:
            st.markdown("### Hapus Aplikasi")

            app_delete = st.selectbox(
                "Pilih aplikasi yang mau dihapus",
                st.session_state.app_list,
                key="delete_app"
            )

            if st.button("🗑️ Hapus Aplikasi", use_container_width=True):

                st.session_state.app_list.remove(app_delete)
                pd.Series(st.session_state.app_list).to_json(APP_FILE)

                # 🔥 OPTIONAL: hapus file CSV terkait juga
                files = [
                    f"data_tot_{app_delete}.csv",
                    f"data_error_{app_delete}.csv",
                    f"data_ueq_light_{app_delete}.csv",
                    f"data_ueq_dark_{app_delete}.csv",
                    f"data_pref_{app_delete}.csv"
                ]

                for f in files:
                    path = os.path.join(BASE_DIR, f)
                    if os.path.exists(path):
                        os.remove(path)

                st.success(f"{app_delete} berhasil dihapus")
                st.rerun()

    # --- PROTEKSI UTAMA ---
    # Jika list kosong, tampilkan info dan hentikan eksekusi kode di bawahnya
    if not st.session_state.app_list:
        st.info("Silakan tambah aplikasi objek terlebih dahulu di atas.")
        st.stop() # Menghindari AttributeError: 'NoneType'

    # Pilih aplikasi aktif dari list yang ada
    app = st.selectbox("Pilih Aplikasi Analisis", st.session_state.app_list)
    n = st.number_input("Sample Size (N)", min_value=1, max_value=100, value=25)

    # --- DEFINISI FILE (Setelah variabel 'app' pasti ada isinya) ---
    file_tot = os.path.join(BASE_DIR, f"data_tot_{app}.csv")
    file_error = os.path.join(BASE_DIR, f"data_error_{app}.csv")
    file_ueq_light = os.path.join(BASE_DIR, f"data_ueq_light_{app}.csv")
    file_ueq_dark = os.path.join(BASE_DIR, f"data_ueq_dark_{app}.csv")
    file_pref = os.path.join(BASE_DIR, f"data_pref_{app}.csv")

    # Project Info Card
    st.markdown(f"""
        <div style="background-color: #f8fafc; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; margin: 15px 0;">
            <p style="font-size: 9px; color: #6366f1; font-weight: 800; margin-bottom: 5px; text-transform: uppercase;">Active Project</p>
            <p style="font-size: 14px; color: #1e293b; font-weight: 700; margin-bottom: 2px;">{app.upper() if app else "No App"}</p>
            <p style="font-size: 10px; color: #64748b; line-height: 1.2;">Method: Within-Subjects Design</p>
        </div>
    """, unsafe_allow_html=True)

    # --- NAVIGASI ---
    st.markdown('<p class="menu-label">Main Navigation</p>', unsafe_allow_html=True)
    menu = st.selectbox("Menu", ["Home", "Overview", "Time on Task", "Error Rate", "UEQ Analysis", "Preferensi Responden"], label_visibility="collapsed")

    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    # --- SECTION 4: DATA CONTROLS ---
    st.sidebar.markdown("---")
    st.markdown('<p class="menu-label">System Control</p>', unsafe_allow_html=True)
    
    if "confirm_reset" not in st.session_state:
        st.session_state.confirm_reset = False

    

    if not st.session_state.confirm_reset:
        if st.button("🗑️ RESET SEMUA DATA", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.markdown("""
            <div style="background-color: #fef2f2; padding: 10px; border-radius: 8px; border: 1px solid #fee2e2; margin-bottom: 10px;">
                <p style="font-size: 11px; color: #991b1b; text-align: center; margin: 0;"><b>Hapus semua file .csv?</b></p>
            </div>
        """, unsafe_allow_html=True)
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            if st.button("Batal", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
        with col_res2:
            if st.button("Ya, Hapus", type="primary", use_container_width=True):
                # Logika hapus file tetap dipertahankan
                files_to_delete = [
                    os.path.join(BASE_DIR, f"data_tot_{app}.csv"),
                    os.path.join(BASE_DIR, f"data_error_{app}.csv"),
                    os.path.join(BASE_DIR, f"data_ueq_light_{app}.csv"),
                    os.path.join(BASE_DIR, f"data_ueq_dark_{app}.csv"),
                    os.path.join(BASE_DIR, f"preferensi_positif_{app}.csv"),
                    os.path.join(BASE_DIR, f"preferensi_negatif_{app}.csv")
                ]
                for f in files_to_delete:
                    if os.path.exists(f): os.remove(f)
                st.session_state.confirm_reset = False
                st.success("Data direset!")
                st.rerun()

# ======================
# FILE
# ======================

file_tot = os.path.join(BASE_DIR, f"data_tot_{app}.csv")
file_error = os.path.join(BASE_DIR, f"data_error_{app}.csv")
file_ueq_light = os.path.join(BASE_DIR, f"data_ueq_light_{app}.csv")
file_ueq_dark = os.path.join(BASE_DIR, f"data_ueq_dark_{app}.csv")
file_pref = os.path.join(BASE_DIR, f"data_pref_{app}.csv")





# ======================
# ADJUST RESPONDEN
# ======================

def adjust_dataframe(df,n):

    if len(df) < n:

        new_rows = pd.DataFrame({
        "Responden":[f"R{i+1}" for i in range(len(df),n)]
        })

        df = pd.concat([df,new_rows],ignore_index=True)

    if len(df) > n:
        df = df.iloc[:n]

    df["Responden"] = [f"R{i+1}" for i in range(len(df))]

    return df
# ======================
# LOAD DATA TOT
# ======================

columns = ["Responden","Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]

if os.path.exists(file_tot):
    df_tot = pd.read_csv(file_tot)

    if "reload_data" in st.session_state:
        del st.session_state["reload_data"]

else:
    df_tot = pd.DataFrame(columns=columns)

df_tot = adjust_dataframe(df_tot,n)

for c in columns[1:]:
    if c not in df_tot:
        df_tot[c] = 0

# ======================
# LOAD DATA ERROR
# ======================

if os.path.exists(file_error):
    df_error = pd.read_csv(file_error)
else:
    df_error = pd.DataFrame(columns=columns)

df_error = adjust_dataframe(df_error,n)

for c in columns[1:]:
    if c not in df_error:
        df_error[c] = 0

# ======================
# LOAD DATA UEQ
# ======================

scales = {
"Daya tarik":[1,12,14,16,24,25],
"Kejelasan":[2,4,13,21],
"Efisiensi":[9,20,22,23],
"Ketepatan":[8,11,17,19],
"Stimulasi":[5,6,7,18],
"Kebaruan":[3,10,15,26]
}

items=[f"I{i}" for i in range(1,27)]

# Light Mode
if os.path.exists(file_ueq_light):
    light_df = pd.read_csv(file_ueq_light)
else:
    light_df = pd.DataFrame(4,index=range(n),columns=items)

# Dark Mode
if os.path.exists(file_ueq_dark):
    dark_df = pd.read_csv(file_ueq_dark)
else:
    dark_df = pd.DataFrame(4,index=range(n),columns=items)

# ======================
# FIX DATA TYPE UEQ
# ======================

light_df = light_df[items]
dark_df = dark_df[items]

light_df = light_df.apply(pd.to_numeric, errors="coerce")
dark_df = dark_df.apply(pd.to_numeric, errors="coerce")

# ======================
# PREPROCESS UEQ (SAMA SEPERTI UEQ TOOL)
# ======================

def preprocess_ueq(df):
    """EXACT UEQ Tools Excel preprocessing"""
    df = df.copy()
    
    # Step 1: Transform to -3 to +3 (1→-3, 4→0, 7→+3)
    df = df - 4  
    
    # Step 2: Reverse negative items (multiply by -1)
    reverse_items = [2,4,6,8,11,13,15,17,19,21,23,25]
    for item in reverse_items:
        col = f"I{item}"
        if col in df.columns:
            df[col] = -df[col]
    
    return df



def calculate_ueq_tool_style(df):
    """100% UEQ Tools Excel: AVERAGE semua cells per scale"""
    df_proc = preprocess_ueq(df)
    results = []
    
    for scale_name, item_list in scales.items():
        # Ambil semua items untuk scale ini
        scale_cols = [f"I{i}" for i in item_list]
        
        # 🔥 UEQ TOOLS EXACT: AVERAGE SEMUA CELLS (flatten 2D → 1D)
        all_scale_values = df_proc[scale_cols].stack().values  # Semua cells jadi 1 array
        
        # Excel AVERAGE(): skip blank, include 0
        valid_values = all_scale_values[~np.isnan(all_scale_values)]
        
        if len(valid_values) == 0:
            scale_mean = np.nan
            scale_var = np.nan
        else:
            # 🔥 EXACT Excel formulas
            scale_mean = np.mean(valid_values)  # =AVERAGE()
            scale_var = np.var(valid_values, ddof=0)  # =VAR.P() population variance
        
        results.append({
            "Scale": scale_name,
            "Mean": round(scale_mean, 3),
            "Variance": round(scale_var, 3) if not np.isnan(scale_var) else np.nan
        })
    
    return pd.DataFrame(results)


def calculate_ueq_mean_only(light_df, dark_df):
    """Mean per responden untuk paired t-test per scale"""
    light_proc = preprocess_ueq(light_df)
    dark_proc = preprocess_ueq(dark_df)
    
    results = []
    for scale_name, item_indices in scales.items():
        scale_cols = [f"I{i}" for i in item_indices]
        
        # Mean PER RESPONDEN (bukan flatten)
        light_means = light_proc[scale_cols].mean(axis=1)
        dark_means = dark_proc[scale_cols].mean(axis=1)
        
        # Paired t-test
        mask = ~(light_means.isna() | dark_means.isna())
        if mask.sum() >= 2:
            t_stat, p_val = stats.ttest_rel(light_means[mask], dark_means[mask])
        else:
            t_stat, p_val = np.nan, np.nan
        
        results.append({
            "Scale": scale_name,
            "Light Mean": round(light_means.mean(), 3),
            "Dark Mean": round(dark_means.mean(), 3),
            "T-stat": round(t_stat, 3),
            "P-value": round(p_val, 3)
        })
    
    return pd.DataFrame(results)

# ======================
# FUNCTION PAIRED T TEST
# ======================

def paired_test_spss(light, dark):

    light = pd.to_numeric(light, errors="coerce")
    dark = pd.to_numeric(dark, errors="coerce")

    diff = np.array(light) - np.array(dark)

    mean = np.mean(diff)
    std = np.std(diff, ddof=1)

    n = len(diff)

    se = std / np.sqrt(n)

    t, p_two = stats.ttest_rel(light, dark)

    df = n - 1

    p_one = p_two / 2

    ci_low, ci_up = stats.t.interval(
        0.95,
        df,
        loc=mean,
        scale=se
    )

    return mean, std, se, ci_low, ci_up, t, df, p_one, p_two


if menu == "Home":

    # ======================
    # HERO (PRO LOOK)
    # ======================
    st.markdown(f"""
    <div style="
    background: linear-gradient(135deg,#4f46e5,#6366f1);
    padding:35px;
    border-radius:20px;
    color:white;
    margin-bottom:25px;
    ">

    <div style="font-size:26px;font-weight:800;">
    UX Research Analytics Platform
    </div>

    <div style="font-size:13px;margin-top:10px;max-width:600px;line-height:1.6;">
    Platform analisis kuantitatif untuk mengevaluasi pengalaman pengguna 
    menggunakan <b>Time on Task</b>, <b>Error Rate</b>, <b>UEQ</b>, dan 
    <b>Preferensi Responden</b>.  
    Dapat digunakan untuk berbagai studi UX dan penelitian lanjutan.
    </div>

    <div style="margin-top:15px;font-size:12px;">
    📊 Current Study: <b>{app}</b> &nbsp; | &nbsp; 👥 Sample Size: <b>{n}</b>
    </div>

    </div>
    """, unsafe_allow_html=True)

    # ======================
    # WORKFLOW (STEP)
    # ======================
    st.markdown("### Research Workflow")

    col1, col2, col3, col4 = st.columns(4)

    steps = [
        ("1. Input Data", "Masukkan data eksperimen pengguna"),
        ("2. Process", "Sistem menghitung mean & statistik"),
        ("3. Compare", "Bandingkan Light vs Dark Mode"),
        ("4. Insight", "Dapatkan kesimpulan penelitian")
    ]

    for col, (title, desc) in zip([col1,col2,col3,col4], steps):
        col.markdown(f"""
        <div class="card">
            <div style="font-size:13px;font-weight:700;color:{text_main}">{title}</div>
            <div style="font-size:11px;color:{text_soft};margin-top:6px;">
            {desc}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ======================
    # METRICS (INTI PENELITIAN)
    # ======================
    st.markdown("### Metrics Available")

    col1, col2, col3 = st.columns(3)

    col1.markdown(f"""
    <div class="card">
    <b style="color:{text_main}">Time on Task</b><br>
    <span style="font-size:11px;color:{text_soft};">
    Mengukur efisiensi penyelesaian tugas
    </span>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
    <b style="color:{text_main}">Error Rate</b><br>
    <span style="font-size:11px;color:{text_soft};">
    Mengukur tingkat kesalahan pengguna
    </span>
    </div>
    """, unsafe_allow_html=True)

    col3.markdown(f"""
    <div class="card">
    <b style="color:{text_main}">UEQ Analysis</b><br>
    <span style="font-size:11px;color:{text_soft};">
    Evaluasi pengalaman pengguna (UX)
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # STATISTICAL METHODS
    # ======================
    st.markdown("### Statistical Methods")

    col1, col2 = st.columns(2)

    col1.markdown(f"""
    <div class="card">
    <b style="color:{text_main}">Wilcoxon Test</b><br>
    <span style="font-size:11px;color:{text_soft};">
    Untuk Time on Task (non-parametrik)
    </span>
    </div>
    """, unsafe_allow_html=True)

    col2.markdown(f"""
    <div class="card">
    <b style="color:{text_main}">Paired T-Test</b><br>
    <span style="font-size:11px;color:{text_soft};">
    Untuk Error Rate &amp; UEQ Analysis
    </span>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # USE CASE (INI YANG BIKIN KELIATAN REUSABLE)
    # ======================
    st.markdown("### Research Use Cases")

    st.markdown(f"""
    <div class="card">
    <ul style="font-size:12px;color:{text_main};line-height:1.8;">
    <li>Perbandingan Light Mode vs Dark Mode</li>
    <li>Evaluasi redesign UI aplikasi</li>
    <li>Analisis usability fitur baru</li>
    <li>Benchmark antar aplikasi digital</li>
    <li>Eksperimen UX berbasis pengguna</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # CTA (BIAR NGARAH KE MENU)
    # ======================
    st.markdown("### Start Analysis")

    st.success("Gunakan menu di sidebar untuk mulai penelitian dan analisis data.")


# ======================
# OVERVIEW
# ======================

if menu == "Overview":

    # ======================
    # HEADER
    # ======================

    st.markdown("""
    <div class="main-title">Analytics</div>
    <div class="subtitle">UX analysis for mobile applications</div>
    """, unsafe_allow_html=True)

    
    # ======================
    # HITUNG UX METRICS
    # ======================

    avg_light_tot = df_tot[["Light_T1","Light_T2","Light_T3"]].mean().mean()
    avg_dark_tot = df_tot[["Dark_T1","Dark_T2","Dark_T3"]].mean().mean()

    avg_light_err = df_error[["Light_T1","Light_T2","Light_T3"]].mean().mean()
    avg_dark_err = df_error[["Dark_T1","Dark_T2","Dark_T3"]].mean().mean()

    # ======================
    # UEQ PREPROCESS (SAMA SEPERTI UEQ TOOL)
    # ======================


    light_ueq_mean = calculate_ueq_tool_style(light_df)["Mean"].mean()
    dark_ueq_mean = calculate_ueq_tool_style(dark_df)["Mean"].mean()

    # ======================
    # LOAD DATA PREFERENSI
    # ======================

    file_pos = os.path.join(BASE_DIR, f"preferensi_positif_{app}.csv")
    file_neg = os.path.join(BASE_DIR, f"preferensi_negatif_{app}.csv")

    aspek = {

    "Readability":["R1","R2","R3","R4"],
    "Eye Strain":["ES1","ES2","ES3","ES4"],
    "Usability":["U1","U2","U3","U4"],
    "Battery":["B1","B2","B3","B4"],
    "Efficiency":["E1","E2","E3","E4"],
    "Aesthetic":["ED1","ED2","ED3","ED4"]

    }

    aspek_result = []

    if os.path.exists(file_pos) and os.path.exists(file_neg):

        df_pos = pd.read_csv(file_pos)
        df_neg = pd.read_csv(file_neg)

        df_pos = df_pos.fillna(0)
        df_neg = df_neg.fillna(0)

        for a,cols in aspek.items():

            pos_val = df_pos[cols].mean().mean()
            neg_val = (8 - df_neg[cols]).mean().mean()

            if pd.isna(pos_val) or pd.isna(neg_val):
                continue

            mean_val = (pos_val + neg_val) / 2

            prefer = "Light Mode" if mean_val < 4 else "Dark Mode"

            aspek_result.append(prefer)

    light_pref = aspek_result.count("Light Mode")
    dark_pref = aspek_result.count("Dark Mode")

    best_pref = "Light Mode" if light_pref > dark_pref else "Dark Mode"


    # ======================
    # KPI CARDS (WITH MODE IDENTIFIERS)
    # ======================

    col0, col1, col2, col3, col4 = st.columns(5)

    with col0:
        icon = "📘" if app == "Facebook" else "🛒"
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Aplikasi</div>
            <div class="metric-value">{icon} {app}</div>
            <div class="metric-footer">Sampel: <b>{n} Responden</b></div>
        </div>
        """, unsafe_allow_html=True)

    with col1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">UEQ Mean Score</div>
            <div class="metric-value">
                <span class="val-light">{round(light_ueq_mean,2)}</span>
                <span class="vs-divider">|</span>
                <span class="val-dark">{round(dark_ueq_mean,2)}</span>
            </div>
            <div class="metric-footer">
                <span style="color:#6366f1">●</span> Light <span style="margin-left:10px; color:#1e293b">●</span> Dark
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Time on Task (Avg)</div>
            <div class="metric-value">
                <span class="val-light">{round(avg_light_tot,1)}s</span>
                <span class="vs-divider">|</span>
                <span class="val-dark">{round(avg_dark_tot,1)}s</span>
            </div>
            <div class="metric-footer">
                <span style="color:#6366f1">●</span> Light <span style="margin-left:10px; color:#1e293b">●</span> Dark
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Error Rate (Avg)</div>
            <div class="metric-value">
                <span class="val-light">{round(avg_light_err,2)}</span>
                <span class="vs-divider">|</span>
                <span class="val-dark">{round(avg_dark_err,2)}</span>
            </div>
            <div class="metric-footer">
                <span style="color:#6366f1">●</span> Light <span style="margin-left:10px; color:#1e293b">●</span> Dark
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        pref_color = "#6366f1" if best_pref == "Light Mode" else "#1e293b"
        pref_bg = "rgba(99, 102, 241, 0.1)" if best_pref == "Light Mode" else "rgba(30, 41, 59, 0.1)"
        st.markdown(f"""
        <div class="card">
            <div class="metric-title">Best Preference</div>
            <div class="metric-value" style="color:{pref_color}">{best_pref}</div>
            <div class="status-badge" style="background:{pref_bg}; color:{pref_color}">RESULT</div>
            <div class="metric-footer">Kesimpulan 6 Aspek</div>
        </div>
        """, unsafe_allow_html=True)
    
    



    # ======================
    # PREPARASI DATA (LOGIKA PROPORSI TOTAL 100%)
    # ======================
    
    # 1. Inisialisasi awal dengan 0% untuk semua aspek (Default jika data kosong)
    pos_percent = {a: 0.0 for a in aspek.keys()}
    neg_percent = {a: 0.0 for a in aspek.keys()}

    # --- HITUNG PROPORSI POSITIF ---
    if os.path.exists(file_pos):
        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        raw_values_pos = {}
        for a, cols in aspek.items():
            existing_cols = [c for c in cols if c in df_pos_raw.columns]
            if existing_cols:
                # Ambil rata-rata skor mentah (1-7)
                mean_val = df_pos_raw[existing_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
                raw_values_pos[a] = mean_val if (not pd.isna(mean_val) and mean_val > 0) else 0
        
        # Normalisasi ke 100%
        total_pos = sum(raw_values_pos.values())
        if total_pos > 0:
            pos_percent = {k: (v / total_pos) * 100 for k, v in raw_values_pos.items()}

    # --- HITUNG PROPORSI NEGATIF (LOGIKA SAMA DENGAN POSITIF) ---
    if os.path.exists(file_neg):
        df_neg_raw = pd.read_csv(file_neg).fillna(0)
        raw_values_neg = {}
        for a, cols in aspek.items():
            existing_cols = [c for c in cols if c in df_neg_raw.columns]
            if existing_cols:
                # Karena Anda bilang logikanya sama, kita ambil mean mentahnya juga
                # (Semakin tinggi skor aspek negatif, semakin besar porsinya di pie)
                mean_val = df_neg_raw[existing_cols].apply(pd.to_numeric, errors='coerce').mean().mean()
                raw_values_neg[a] = mean_val if (not pd.isna(mean_val) and mean_val > 0) else 0
        
        # Normalisasi ke 100%
        total_neg = sum(raw_values_neg.values())
        if total_neg > 0:
            neg_percent = {k: (v / total_neg) * 100 for k, v in raw_values_neg.items()}

    col_tot, col_err = st.columns(2)

    # ======================
    # TIME ON TASK
    # ======================
    with col_tot:
        st.markdown("##### Time on Task")

        fig_tot, ax_tot = plt.subplots(figsize=(4,3))

        ax_tot.bar(
            ["Light", "Dark"],
            [avg_light_tot, avg_dark_tot],
            color=["#6366f1","#1e293b"]
        )

        ax_tot.set_ylabel("Seconds")
        st.pyplot(fig_tot)


    # ======================
    # ERROR RATE
    # ======================
    with col_err:
        st.markdown("##### Error Rate")

        fig_err, ax_err = plt.subplots(figsize=(4,3))

        ax_err.bar(
            ["Light", "Dark"],
            [avg_light_err, avg_dark_err],
            color=["#6366f1","#1e293b"]
        )

        ax_err.set_ylabel("Error")
        st.pyplot(fig_err)


    # ======================
    # UEQ (FULL WIDTH)
    # ======================
    st.markdown("##### UEQ Score")

    fig_ueq, ax_ueq = plt.subplots(figsize=(6,3))

    ax_ueq.bar(
        ["Light", "Dark"],
        [light_ueq_mean, dark_ueq_mean],
        color=["#6366f1","#1e293b"]
    )

    ax_ueq.set_ylabel("Score")
    st.pyplot(fig_ueq)


    
    # ======================
    # 2. PIE POSITIF & NEGATIF (BOXED & WHITE BACKGROUND)
    # ======================
    col_left, col_right = st.columns(2)

    # --- KOLOM KIRI: PREFERENSI POSITIF ---
    with col_left:
        # Pindahkan judul ke dalam container agar ikut terbungkus
        with st.container(border=True):
            st.markdown('<div style="font-size:16px; font-weight:700; color:#1e293b; margin-bottom:15px;">Preferensi Positif</div>', unsafe_allow_html=True)
            
            if pos_percent:
                c1, c2 = st.columns([1, 1.2]) 
                
                with c1:
                    colors_p = ["#4338ca", "#4f46e5", "#6366f1", "#818cf8", "#a5b4fc", "#c7d2fe"]
                    fig_p = create_donut_chart(pos_percent, colors_p)
                    st.plotly_chart(fig_p, use_container_width=True, config={'displayModeBar': False})
                
                with c2:
                    legend_html = '<div style="margin-top: 5px;">'
                    for i, (name, val) in enumerate(pos_percent.items()):
                        color = colors_p[i % len(colors_p)]
                        legend_html += f"""
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; align-items:center;">
                            <div style="display:flex; align-items:center;">
                                <div style="width:8px; height:8px; background:{color}; border-radius:50%; margin-right:8px;"></div>
                                <span style="font-size:11px; font-weight:600; color:#1e293b;">{name}</span>
                            </div>
                            <span style="font-size:11px; font-weight:700; color:#1e293b;">{round(val,1)}%</span>
                        </div>"""
                    legend_html += '</div>'
                    st.markdown(legend_html, unsafe_allow_html=True)
            else:
                st.info("Data tidak tersedia")

    # --- KOLOM KANAN: PREFERENSI NEGATIF ---
    with col_right:
        with st.container(border=True):
            st.markdown('<div style="font-size:16px; font-weight:700; color:#1e293b; margin-bottom:15px;">Preferensi Negatif</div>', unsafe_allow_html=True)
            
            if neg_percent:
                c1, c2 = st.columns([1, 1.2])
                
                with c1:
                    colors_n = ["#1e3a8a", "#1e40af", "#1d4ed8", "#2563eb", "#3b82f6", "#60a5fa"]
                    fig_n = create_donut_chart(neg_percent, colors_n)
                    st.plotly_chart(fig_n, use_container_width=True, config={'displayModeBar': False})
                
                with c2:
                    legend_html = '<div style="margin-top: 10px;">'
                    for i, (name, val) in enumerate(neg_percent.items()):
                        color = colors_n[i % len(colors_n)]
                        legend_html += f"""
                        <div style="display:flex; justify-content:space-between; margin-bottom:8px; align-items:center;">
                            <div style="display:flex; align-items:center;">
                                <div style="width:8px; height:8px; background:{color}; border-radius:50%; margin-right:8px;"></div>
                                <span style="font-size:11px; font-weight:600; color:#1e293b;">{name}</span>
                            </div>
                            <span style="font-size:11px; font-weight:700; color:#1e293b;">{round(val,1)}%</span>
                        </div>"""
                    legend_html += '</div>'
                    st.markdown(legend_html, unsafe_allow_html=True)
            else:
                st.info("Data tidak tersedia")

    # ======================
    # PREPARASI DATA (PASTIKAN BAGIAN INI ADA DI ATAS)
    # ======================
    pref_diff = []  # Inisialisasi awal agar tidak Error

    if os.path.exists(file_pos) and os.path.exists(file_neg):
        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        df_neg_raw = pd.read_csv(file_neg).fillna(0)
        
        cols_all = [c for cols in aspek.values() for c in cols]
        # Pastikan kolom ada di dataframe
        existing_cols_pos = [c for c in cols_all if c in df_pos_raw.columns]
        existing_cols_neg = [c for c in cols_all if c in df_neg_raw.columns]
        
        if existing_cols_pos and existing_cols_neg:
            p_val_v = df_pos_raw[existing_cols_pos].apply(pd.to_numeric, errors='coerce').mean(axis=1)
            n_val_v = (8 - df_neg_raw[existing_cols_neg].apply(pd.to_numeric, errors='coerce')).mean(axis=1)
            pref_diff = (p_val_v - n_val_v).dropna().tolist()


    # ========================================================
    # STATISTICAL SIGNIFICANCE ANALYSIS (FIXED UI)
    # ========================================================
    st.markdown('<div class="section-title">Statistical Significance Analysis</div>', unsafe_allow_html=True)

    # --- PERHITUNGAN STATISTIK ---
    # cek apakah data kosong
    tot_empty = df_tot[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    err_empty = df_error[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    ueq_empty = light_df.replace(4, np.nan).dropna(how="all").empty \
          and dark_df.replace(4, np.nan).dropna(how="all").empty

    # ======================
    # TIME ON TASK
    # ======================
    if tot_empty:
        p_tot = np.nan
    else:
        l_tot_v = df_tot[["Light_T1","Light_T2","Light_T3"]].mean(axis=1)
        d_tot_v = df_tot[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)
        t_tot, p_tot = stats.ttest_rel(l_tot_v, d_tot_v)

    # ======================
    # ERROR RATE
    # ======================
    if err_empty:
        p_err = np.nan
    else:
        l_err_v = df_error[["Light_T1","Light_T2","Light_T3"]].mean(axis=1)
        d_err_v = df_error[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)
        t_err, p_err = stats.ttest_rel(l_err_v, d_err_v)

    # ======================
    # UEQ
    # ======================
    if ueq_empty:
        p_ueq = np.nan
    else:
        l_ueq_v = preprocess_ueq(light_df).mean(axis=1)
        d_ueq_v = preprocess_ueq(dark_df).mean(axis=1)

        t_ueq, p_ueq = stats.ttest_rel(l_ueq_v, d_ueq_v)
    
    # ======================
    # SUBJECTIVE PREFERENCE
    # ======================
    p_pref = np.nan
    if len(pref_diff) > 1:
        t_pref, p_pref = stats.ttest_1samp(pref_diff, 0)

    # ======================
    # GRAND P VALUE
    # ======================
    grand_p = np.nanmean([p_tot, p_err, p_ueq, p_pref])
    p_val_final = grand_p

    # --- TAMPILAN DASHBOARD ---
    col_main, col_detail = st.columns([1.2, 2.8]) # Penyesuaian rasio kolom

    with col_main:
        is_sig = False if pd.isna(grand_p) else grand_p < 0.05
        # Warna yang lebih soft (Modern Palette)
        bg_color = "#ecfdf5" if is_sig else "#fffbeb"
        border_color = "#10b981" if is_sig else "#f59e0b"
        accent_color = "#059669" if is_sig else "#d97706"
        
        st.markdown(f"""
            <div style="background:{bg_color}; border: 2px solid {border_color}; padding: 30px 15px; border-radius: 15px; text-align: center; height: 350px; display: flex; flex-direction: column; justify-content: center; align-items: center; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                <div style="color: {accent_color}; font-size: 12px; font-weight: 800; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 10px;">Overall Significance</div>
                <div style="font-size: 48px; font-weight: 900; color: {accent_color}; line-height: 1;">p = {round(grand_p, 4)}</div>
                <div style="display: inline-block; padding: 6px 15px; background: {accent_color}; color: white; border-radius: 20px; font-size: 14px; font-weight: 700; margin-top: 20px;">
                    {"SIGNIFICANT" if is_sig else "NOT SIGNIFICANT"}
                </div>
                <div style="font-size: 11px; color: #64748b; line-height: 1.6; margin-top: 25px; max-width: 200px;">
                    Akumulasi analisis performa, akurasi, kuesioner UEQ, dan preferensi responden.
                </div>
            </div>
        """, unsafe_allow_html=True)

    with col_detail:
        # 1. Buat list untuk menampung baris-baris HTML
        rows_list = []
        
        # Data untuk diulang
        stats_data = [
            ("Time on Task (ToT)", "Paired Sample T-Test", p_tot),
            ("Error Rate (Akurasi)", "Paired Sample T-Test", p_err),
            ("UEQ Analysis", "Independent Sample T-Test", p_ueq),
            ("Subjective Preference", "1-Sample T-Test", p_pref)
        ]

        for label, method, p_val in stats_data:
            row_sig = p_val < 0.05
            p_color = "#10b981" if row_sig else "#64748b"
            bg_dot = "#10b981" if row_sig else "#e2e8f0"
            
            # Tulis HTML baris (Pastikan nempel ke kiri)
            row_html = f"""
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 12px 0; border-bottom: 1px solid #f8fafc;">
        <div style="display: flex; align-items: center;">
        <div style="width: 8px; height: 8px; background: {bg_dot}; border-radius: 50%; margin-right: 12px;"></div>
        <div style="text-align: left;">
        <div style="font-size: 13px; font-weight: 600; color: #334155;">{label}</div>
        <div style="font-size: 10px; color: #6366f1; font-weight: 500;">{method}</div>
        </div>
        </div>
        <div style="text-align: right;">
        <div style="font-size: 14px; font-weight: 800; color: {p_color};">p = {round(p_val, 4)}</div>
        <div style="font-size: 9px; color: {p_color}; font-weight: 700; text-transform: uppercase;">{"Signifikan" if row_sig else "Tidak Signifikan"}</div>
        </div>
        </div>"""
            rows_list.append(row_html)

        # 2. Gabungkan semua baris
        all_rows = "".join(rows_list)

        # 3. Bungkus dalam Card Utama (Tulis nempel ke kiri)
        from textwrap import dedent

        # 3. Bungkus dalam Card Utama
        full_card_html = dedent(f"""
        <div style="background:white;padding:25px;border-radius:15px;border:1px solid #e2e8f0;height:350px;display:flex;flex-direction:column;box-shadow:0 2px 4px rgba(0,0,0,0.05);">

        <div style="font-size:18px;font-weight:700;color:#1e293b;margin-bottom:15px;border-bottom:2px solid #f1f5f9;padding-bottom:10px;">
        Rincian Metode Analisis
        </div>

        <div style="overflow-y:auto;flex-grow:1;">
        {all_rows}
        </div>

        </div>
        """)

        # 4. Tampilkan
        st.markdown(full_card_html, unsafe_allow_html=True)
    # ======================
    # UEQ SCALE ANALYSIS
    # ======================

    result_light = calculate_ueq_tool_style(light_df)
    result_dark = calculate_ueq_tool_style(dark_df)

    ueq_scale_means = {}

    for i in range(len(result_light)):
        scale = result_light.loc[i, "Scale"]
        ueq_scale_means[scale] = {
            "light": result_light.loc[i, "Mean"],
            "dark": result_dark.loc[i, "Mean"]
        }

    # cari scale yang paling unggul
    best_ueq_scale = max(
        ueq_scale_means,
        key=lambda s: max(ueq_scale_means[s]["light"], ueq_scale_means[s]["dark"])
    )

    # ======================
    # PREFERENCE ANALYSIS
    # ======================

    pref_scores = {}

    if os.path.exists(file_pos) and os.path.exists(file_neg):

        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        df_neg_raw = pd.read_csv(file_neg).fillna(0)

        for a, cols in aspek.items():

            pos_val = df_pos_raw[cols].apply(pd.to_numeric, errors='coerce').mean().mean()
            neg_val = (8 - df_neg_raw[cols].apply(pd.to_numeric, errors='coerce')).mean().mean()

            pref_scores[a] = (pos_val + neg_val) / 2

    # aspek tertinggi dan terendah
    if len(pref_scores) > 0:
        best_pref_aspect = max(pref_scores, key=pref_scores.get)
        worst_pref_aspect = min(pref_scores, key=pref_scores.get)
    else:
        best_pref_aspect = "Belum ada data"
        worst_pref_aspect = "Belum ada data"

    # ======================
    # ANALISIS PREFERENSI PER ASPEK
    # ======================

    aspect_scores = {}

    if os.path.exists(file_pos) and os.path.exists(file_neg):

        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        df_neg_raw = pd.read_csv(file_neg).fillna(0)

        for a, cols in aspek.items():

            pos_val = df_pos_raw[cols].apply(pd.to_numeric, errors='coerce').mean().mean()
            neg_val = (8 - df_neg_raw[cols].apply(pd.to_numeric, errors='coerce')).mean().mean()

            mean_val = (pos_val + neg_val) / 2

            prefer_mode = "☀️ LIGHT" if mean_val < 4 else "🌙 DARK"

            aspect_scores[a] = {
                "score": mean_val,
                "mode": prefer_mode
            }

    if len(aspect_scores) > 0:

        best_aspect = max(aspect_scores, key=lambda x: aspect_scores[x]["score"])
        best_aspect_mode = aspect_scores[best_aspect]["mode"]

        worst_aspect = min(aspect_scores, key=lambda x: aspect_scores[x]["score"])
        worst_aspect_mode = aspect_scores[worst_aspect]["mode"]

    else:

        best_aspect = "Belum ada data"
        best_aspect_mode = "-"
        worst_aspect = "Belum ada data"
        worst_aspect_mode = "-"

    # ======================
    # FINAL INSIGHT (TOTAL RESET)
    # ======================

    st.write("")
    st.write("")
    # cek apakah data sudah diinput
    data_tot_empty = df_tot[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    data_err_empty = df_error[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    data_ueq_empty = light_df.sum().sum() == 0 and dark_df.sum().sum() == 0

    pref_empty = not (os.path.exists(file_pos) and os.path.exists(file_neg))

    if data_tot_empty or data_err_empty or data_ueq_empty or pref_empty:
        st.info("Silakan input semua data penelitian terlebih dahulu untuk menampilkan kesimpulan penelitian.")
    else:

        st.markdown('<div class="section-title">Kesimpulan Akhir Penelitian</div>', unsafe_allow_html=True)

        main_color = "#6366f1"
        accent_bg = "rgba(99, 102, 241, 0.05)"

        sig_label = "SIGNIFICANT" if p_val_final < 0.05 else "NOT SIGNIFICANT"
        sig_color = "#10b981" if p_val_final < 0.05 else "#f59e0b"
        sig_bg = "#ecfdf5" if p_val_final < 0.05 else "#fffbe6"

        win_ueq = '☀️ LIGHT' if light_ueq_mean > dark_ueq_mean else '🌙 DARK'
        win_tot = '☀️ LIGHT' if avg_light_tot < avg_dark_tot else '🌙 DARK'
        win_err = '☀️ LIGHT' if avg_light_err < avg_dark_err else '🌙 DARK'

        st.markdown(f"""
        <div style="
        background:white;
        border-radius:16px;
        border:1px solid #e2e8f0;
        padding:28px;
        box-shadow:0 6px 18px rgba(0,0,0,0.06);
        ">

        <div style="
        text-align:center;
        padding:22px;
        background:{accent_bg};
        border-radius:12px;
        border:1px solid {main_color}33;
        margin-bottom:24px;
        ">

        <div style="font-size:11px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:{main_color};">
        Recommended Interface
        </div>

        <div style="font-size:34px;font-weight:900;color:{main_color};margin-top:6px;">
        {"☀️" if best_pref == "Light Mode" else "🌙"} {best_pref.upper()}
        </div>

        </div>


        <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">


        <div style="background:#f8fafc;padding:14px;border-radius:10px;">
        <b>User Experience (UEQ)</b><br>
        Mode <b>{win_ueq}</b> menunjukkan pengalaman pengguna yang lebih baik
        pada beberapa skala UEQ, dengan skor tertinggi pada skala
        <b>{best_ueq_scale}</b>.
        </div>


        <div style="background:#f8fafc;padding:14px;border-radius:10px;">
        <b>Performance (Time on Task)</b><br>
        Responden menyelesaikan tugas lebih cepat menggunakan
        <b>{win_tot}</b>, yang menunjukkan efisiensi interaksi yang lebih baik.
        </div>


        <div style="background:#f8fafc;padding:14px;border-radius:10px;">
        <b>Accuracy (Error Rate)</b><br>
        <b>{win_err}</b> menunjukkan tingkat kesalahan yang lebih rendah,
        sehingga memberikan akurasi penggunaan yang lebih baik.
        </div>


        <div style="background:#f8fafc;padding:14px;border-radius:10px;">
        <b>Preferensi Positif</b><br>
        Pada enam aspek preferensi yaitu <i>Readability, Eye Strain,
        Usability, Battery, Efficiency</i>, dan <i>Aesthetic</i>,
        aspek dengan penilaian tertinggi ditemukan pada
        <b>{best_pref_aspect}</b> pada mode <b>{best_aspect_mode}</b>.
        </div>


        <div style="background:#f8fafc;padding:14px;border-radius:10px;">
        <b>Preferensi Negatif</b><br>
        Pada enam aspek preferensi yaitu <i>Readability, Eye Strain,
        Usability, Battery, Efficiency</i>, dan <i>Aesthetic</i>,
        aspek dengan penilaian tertinggi ditemukan pada
        <b>{worst_aspect}</b> pada mode <b>{worst_aspect_mode}</b>.
        </div>


        </div>


        <div style="
        text-align:center;
        font-size:14px;
        color:#334155;
        line-height:1.6;
        margin-top:10px;
        ">

        Secara keseluruhan, hasil penelitian menunjukkan bahwa
        <b>{best_pref}</b> memberikan pengalaman pengguna yang lebih optimal
        berdasarkan evaluasi pengalaman pengguna (UEQ), efisiensi penyelesaian
        tugas (Time on Task), tingkat kesalahan (Error Rate), serta preferensi
        responden pada enam aspek pengalaman pengguna.

        </div>


        <div style="
        margin-top:20px;
        display:flex;
        justify-content:center;
        gap:10px;
        ">

        <div style="
        padding:6px 12px;
        background:{sig_bg};
        color:{sig_color};
        border-radius:6px;
        font-size:11px;
        font-weight:800;
        ">
        {sig_label}
        </div>

        </div>

        </div>
        """, unsafe_allow_html=True)

    

# ======================
# TIME ON TASK (UI BARU - SAMA SEPERTI UEQ)
# ======================

if menu == "Time on Task":

    st.markdown("""
    <div style="font-size:28px;font-weight:700;color:#1e293b;margin-bottom:10px;">
    ⏱️ Time on Task Analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:14px;color:#6b7280;">
    Analisis menggunakan Wilcoxon Signed Ranks Test (SPSS Style)
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # DATASET MANAGER
    # ======================
    with st.expander("📁 Dataset Manager", expanded=False):
        dataset_manager(
            df_tot,
            columns,
            file_tot,
            "Dataset Time on Task",
            f"time_on_task_{app}"
        )

    # ======================
    # DATA EDITOR
    # ======================
    st.markdown("### 📊 Dataset Input")
    
    df_edit = st.data_editor(
        df_tot, 
        key="tot_editor", 
        use_container_width=True,
        column_config={
            "Responden": st.column_config.TextColumn("Responden"),
            "Light_T1": st.column_config.NumberColumn("Light T1 (detik)", min_value=0, step=0.1),
            "Light_T2": st.column_config.NumberColumn("Light T2 (detik)", min_value=0, step=0.1),
            "Light_T3": st.column_config.NumberColumn("Light T3 (detik)", min_value=0, step=0.1),
            "Dark_T1": st.column_config.NumberColumn("Dark T1 (detik)", min_value=0, step=0.1),
            "Dark_T2": st.column_config.NumberColumn("Dark T2 (detik)", min_value=0, step=0.1),
            "Dark_T3": st.column_config.NumberColumn("Dark T3 (detik)", min_value=0, step=0.1),
        }
    )
    
    # Save button
    col_save1, col_save2 = st.columns([3,1])
    with col_save2:
        if st.button("💾 Simpan Data", type="primary"):
            df_edit.to_csv(file_tot, index=False)
            st.success("✅ Data tersimpan!")
            st.rerun()
    # ======================
    # ANALYSIS BUTTON (WILCOXON)
    # ======================
    # ... bagian kode sebelumnya ...
    if st.button("🚀 ANALISIS WILCOXON TEST", type="secondary"):
        
        st.markdown("---")
        st.markdown("### 📊 Overall Metrics")
        
        # 1. Hitung rata-rata per responden (untuk distribusi/histogram)
        light_err_per_user = df_tot[["Light_T1","Light_T2","Light_T3"]].mean(axis=1)
        dark_err_per_user = df_tot[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)
        
        # 2. TAMBAHKAN INI: Hitung rata-rata total untuk perbandingan (skalar)
        avg_light_err = light_err_per_user.mean()
        avg_dark_err = dark_err_per_user.mean()
        
        # 3. Sekarang variabel avg_light_err sudah tersedia untuk baris ini
        better_mode = "Light" if avg_light_err < avg_dark_err else "Dark"
        color = "normal" if avg_light_err < avg_dark_err else "inverse"
        
        st.metric(
            label="Lowest Time on Task", 
            value=f"{better_mode}",
            delta=f"{abs(avg_light_err - avg_dark_err):.1f}s"
        )


        # Per task averages
        task_avgs = pd.DataFrame({
            "Task": ["T1", "T2", "T3"],
            "Light Mode": [
                df_tot["Light_T1"].mean(),
                df_tot["Light_T2"].mean(),
                df_tot["Light_T3"].mean()
            ],
            "Dark Mode": [
                df_tot["Dark_T1"].mean(),
                df_tot["Dark_T2"].mean(),
                df_tot["Dark_T3"].mean()
            ]
        })
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            better_mode = "Light" if avg_light_err < avg_dark_err else "Dark"
            color = "normal" if avg_light_err < avg_dark_err else "inverse"
            st.metric(
                label="Lowest Time on Task", 
                value=f"{better_mode}",
                delta=f"{abs(avg_light_err - avg_dark_err):.1f}s",
                delta_color=color
            )
        
        with col2:
            st.metric(
                label="Light Mode Avg", 
                value=f"{avg_light_err:.1f}s"
            )
            
        with col3:
            st.metric(
                label="Dark Mode Avg", 
                value=f"{avg_dark_err:.1f}s"
            )
        
        # ======================
        # TASK TABLE
        # ======================
        st.markdown("### 📋 Task Results")
        st.dataframe(task_avgs.round(2), use_container_width=True)
        
        # ======================
        # WILCOXON SPSS OUTPUT - PER TASK
        # ======================
        st.markdown("### 📈 Wilcoxon Signed Ranks Test — Per Task")

        pairs_per_task = []
        z_values = []
        p_values = []
        for i in range(1, 4):
            light = pd.to_numeric(df_tot[f"Light_T{i}"], errors="coerce")
            dark  = pd.to_numeric(df_tot[f"Dark_T{i}"],  errors="coerce")
            pd_item = compute_wilcoxon_pair(light, dark, f"Light_T{i}", f"Dark_T{i}")
            pairs_per_task.append(pd_item)
            z_values.append(pd_item["z_val"])
            p_values.append(pd_item["p_val"])

        render_spss_wilcoxon(pairs_per_task)

        # ======================
        # OVERALL WILCOXON (MEAN PER USER)
        # ======================
        st.markdown("### 📊 Overall Wilcoxon Test (Mean per User)")
        overall_item = compute_wilcoxon_pair(
            light_err_per_user, dark_err_per_user,
            "Light (mean)", "Dark (mean)"
        )
        render_spss_wilcoxon([overall_item])
        
        # ======================
        # VISUALIZATION
        # ======================
        st.markdown("### 🎯 Visual Comparison")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Time on Task Analysis", fontsize=16, fontweight='bold')
        
        # 1. Per task comparison
        x = np.arange(3)
        width = 0.35
        ax1.bar(x - width/2, task_avgs["Light Mode"], width, label='Light', color="#6366f1", alpha=0.8)
        ax1.bar(x + width/2, task_avgs["Dark Mode"], width, label='Dark', color="#1e293b", alpha=0.8)
        ax1.set_title("Per Task Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["T1", "T2", "T3"])
        ax1.set_ylabel("Time (detik)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        

        
        # 2. Overall distribution (MEAN PER USER)
        ax2.hist(light_err_per_user.dropna(), bins=15, alpha=0.7, color="#6366f1", label='Light', density=True)
        ax2.hist(dark_err_per_user.dropna(), bins=15, alpha=0.7, color="#1e293b", label='Dark', density=True)
        ax2.set_title("Distribution (Mean per User)")
        ax2.set_xlabel("Time (detik)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Wilcoxon Z-scores
        tasks = [f"T{i}" for i in range(1,4)]
        colors_sig = ["#10b981" if p < 0.05 else "#ef4444" for p in p_values]
        ax3.bar(tasks, z_values, color=colors_sig, alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title("Wilcoxon Z-Scores")
        ax3.grid(True, alpha=0.3)
        
        # 4. P-values
        ax4.bar(tasks, p_values, color=colors_sig, alpha=0.8)
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax4.set_title("P-Values")
        ax4.set_ylim(0, max(0.3, max(p_values)*1.1))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ======================
        # BENCHMARK CARDS
        # ======================
        st.markdown("### 🏆 Benchmark Results")
        
        benchmarks = [
            (avg_light_err, "Light Mode", "#6366f1"),
            (avg_dark_err, "Dark Mode", "#1e293b")
        ]
        
        col_b1, col_b2 = st.columns(2)
        
        for i, (avg_time, label, color) in enumerate(benchmarks):
            col = col_b1 if i == 0 else col_b2
            with col:
                st.markdown(f"""
                    <div style="
                        display: flex; flex-direction: column; align-items: center;
                        padding: 25px; background: white; border-radius: 16px; 
                        border: 2px solid {color}20; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                        height: 140px; justify-content: center;
                    ">
                        <div style="font-size: 32px; font-weight: 900; color: {color}; margin-bottom: 8px;">
                            {avg_time:.1f}s
                        </div>
                        <div style="font-size: 14px; color: {color}; font-weight: 600;">
                            {label}
                        </div>
                        <div style="margin-top: 12px; font-size: 12px; color: #10b981; font-weight: 700;">
                            {'🏆 FASTEST' if avg_time == min([avg_light_err, avg_dark_err]) else 'Slower'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # ======================
        # STATISTICAL SUMMARY
        # ======================
        significant_tasks = sum(p < 0.05 for p in p_values)
        overall_p   = overall_item["p_val"]
        overall_z   = overall_item["z_val"]
        overall_sig = "✅" if overall_p < 0.05 else "❌"

        st.markdown("### 📋 Statistical Summary")

        st.markdown(f"""
        <div style="
            background: #f8fafc; padding: 24px; border-radius: 12px;
            border-left: 4px solid #6366f1;
        ">
            <div style="font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 12px;">
                Overall Findings
            </div>
            <ul style="font-size: 14px; color: #374151; line-height: 1.8; margin: 0;">
                <li><b>{significant_tasks}/3 tasks</b> menunjukkan perbedaan signifikan (p &lt; 0.05)</li>
                <li><b>Overall Wilcoxon:</b> Z={overall_z:.3f}, p={overall_p:.3f} {overall_sig}</li>
                <li>Mean Light Mode: <b>{avg_light_err:.1f}s</b></li>
                <li>Mean Dark Mode: <b>{avg_dark_err:.1f}s</b></li>
                <li>{'Light Mode lebih cepat' if avg_light_err < avg_dark_err else 'Dark Mode lebih cepat'} secara deskriptif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("*Wilcoxon Signed Ranks Test • SPSS Compatible Output • Mean per User*")

# ======================
# ERROR RATE (WILCOXON - FIXED)
# ======================

if menu == "Error Rate":

    st.markdown("""
    <div style="font-size:28px;font-weight:700;color:#1e293b;margin-bottom:10px;">
    🚨 Error Rate Analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="font-size:14px;color:#6b7280;">
    Analisis menggunakan Wilcoxon Signed Ranks Test (SPSS Style)
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # DATASET MANAGER
    # ======================
    with st.expander("📁 Dataset Manager", expanded=False):
        dataset_manager(
            df_error,
            columns,
            file_error,
            "Dataset Error Rate",
            f"error_rate_{app}"
        )

    # ======================
    # DATA EDITOR
    # ======================
    st.markdown("### 📊 Dataset Input")
    
    df_edit = st.data_editor(
        df_error, 
        key="error_editor", 
        use_container_width=True,
        column_config={
            "Responden": st.column_config.TextColumn("Responden"),
            "Light_T1": st.column_config.NumberColumn("Light T1 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            "Light_T2": st.column_config.NumberColumn("Light T2 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            "Light_T3": st.column_config.NumberColumn("Light T3 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            "Dark_T1": st.column_config.NumberColumn("Dark T1 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            "Dark_T2": st.column_config.NumberColumn("Dark T2 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
            "Dark_T3": st.column_config.NumberColumn("Dark T3 (error %)", min_value=0, max_value=100, step=0.1, format="%.1f%%"),
        }
    )
    
    # Save button - FIXED WITH UNIQUE KEY
    col_save1, col_save2 = st.columns([3,1])
    with col_save2:
        if st.button("💾 Simpan Data", type="primary", key="save_error_data"):
            df_edit.to_csv(file_error, index=False)
            st.success("✅ Data tersimpan!")
            st.rerun()

    # ======================
    # ANALYSIS BUTTON - FIXED WITH UNIQUE KEY
    # ======================
    if st.button("🚀 ANALISIS WILCOXON TEST", type="secondary", key="analyze_error_rate"):
        
        st.markdown("---")
        st.markdown("### 📊 Overall Metrics")
        
        # Calculate averages
        avg_light_err = df_edit[["Light_T1","Light_T2","Light_T3"]].mean().mean()
        avg_dark_err = df_edit[["Dark_T1","Dark_T2","Dark_T3"]].mean().mean()
        
        # Per task averages
        task_avgs = pd.DataFrame({
            "Task": ["T1", "T2", "T3"],
            "Light Mode": [
                df_edit["Light_T1"].mean(),
                df_edit["Light_T2"].mean(),
                df_edit["Light_T3"].mean()
            ],
            "Dark Mode": [
                df_edit["Dark_T1"].mean(),
                df_edit["Dark_T2"].mean(),
                df_edit["Dark_T3"].mean()
            ]
        })
        
        # Metrics cards
        col1, col2, col3 = st.columns(3)
        
        with col1:
            better_mode = "Light" if avg_light_err < avg_dark_err else "Dark"
            color = "normal" if avg_light_err < avg_dark_err else "inverse"
            st.metric(
                label="Lowest Error Rate", 
                value=f"{better_mode}",
                delta=f"{abs(avg_light_err - avg_dark_err):.1f}%",
                delta_color=color
            )
        
        with col2:
            st.metric(
                label="Light Mode Avg", 
                value=f"{avg_light_err:.1f}%"
            )
            
        with col3:
            st.metric(
                label="Dark Mode Avg", 
                value=f"{avg_dark_err:.1f}%"
            )
        
        # ======================
        # TASK TABLE
        # ======================
        st.markdown("### 📋 Task Results")
        st.dataframe(task_avgs.round(2), use_container_width=True)
        
        # ======================
        # WILCOXON SPSS OUTPUT - PER TASK
        # ======================
        st.markdown("### 📈 Wilcoxon Signed Ranks Test — Per Task")

        pairs_per_task = []
        z_values = []
        p_values = []
        for i in range(1, 4):
            light = pd.to_numeric(df_edit[f"Light_T{i}"], errors="coerce")
            dark  = pd.to_numeric(df_edit[f"Dark_T{i}"],  errors="coerce")
            pd_item = compute_wilcoxon_pair(light, dark, f"Light_T{i}", f"Dark_T{i}")
            pairs_per_task.append(pd_item)
            z_values.append(pd_item["z_val"])
            p_values.append(pd_item["p_val"])

        render_spss_wilcoxon(pairs_per_task)

        # ======================
        # OVERALL WILCOXON (MEAN PER USER)
        # ======================
        st.markdown("### 📊 Overall Wilcoxon Test (Mean per User)")

        light_err_per_user = df_edit[["Light_T1","Light_T2","Light_T3"]].mean(axis=1)
        dark_err_per_user  = df_edit[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)

        overall_item = compute_wilcoxon_pair(
            light_err_per_user, dark_err_per_user,
            "Light (mean)", "Dark (mean)"
        )
        render_spss_wilcoxon([overall_item])
        
        # ======================
        # VISUALIZATION
        # ======================
        st.markdown("### 🎯 Visual Comparison")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Error Rate Analysis", fontsize=16, fontweight='bold')
        
        # 1. Per task comparison
        x = np.arange(3)
        width = 0.35
        ax1.bar(x - width/2, task_avgs["Light Mode"], width, label='Light', color="#6366f1", alpha=0.8)
        ax1.bar(x + width/2, task_avgs["Dark Mode"], width, label='Dark', color="#1e293b", alpha=0.8)
        ax1.set_title("Per Task Comparison")
        ax1.set_xticks(x)
        ax1.set_xticklabels(["T1", "T2", "T3"])
        ax1.set_ylabel("Error Rate (%)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Overall distribution (MEAN PER USER)
        ax2.hist(light_err_per_user, bins=15, alpha=0.7, color="#6366f1", label='Light', density=True)
        ax2.hist(dark_err_per_user, bins=15, alpha=0.7, color="#1e293b", label='Dark', density=True)
        ax2.set_title("Distribution (Mean per User)")
        ax2.set_xlabel("Error Rate (%)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Wilcoxon Z-scores
        tasks = [f"T{i}" for i in range(1,4)]
        colors_sig = ["#10b981" if p < 0.05 else "#ef4444" for p in p_values]
        ax3.bar(tasks, z_values, color=colors_sig, alpha=0.8)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax3.set_title("Wilcoxon Z-Scores")
        ax3.grid(True, alpha=0.3)
        
        # 4. P-values
        ax4.bar(tasks, p_values, color=colors_sig, alpha=0.8)
        ax4.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='α=0.05')
        ax4.set_title("P-Values")
        ax4.set_ylim(0, max(0.3, max(p_values)*1.1))
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # ======================
        # BENCHMARK CARDS
        # ======================
        st.markdown("### 🏆 Benchmark Results")
        
        benchmarks = [
            (avg_light_err, "Light Mode", "#6366f1"),
            (avg_dark_err, "Dark Mode", "#1e293b")
        ]
        
        col_b1, col_b2 = st.columns(2)
        
        for i, (error_rate, label, color) in enumerate(benchmarks):
            col = col_b1 if i == 0 else col_b2
            with col:
                st.markdown(f"""
                <div style="
                    display: flex; flex-direction: column; align-items: center;
                    padding: 25px; background: white; border-radius: 16px; 
                    border: 2px solid {color}20; box-shadow: 0 4px 12px rgba(0,0,0,0.08);
                    height: 140px; justify-content: center;
                ">
                    <div style="font-size: 32px; font-weight: 900; color: {color}; margin-bottom: 8px;">
                        {error_rate:.1f}%
                    </div>
                    <div style="font-size: 14px; color: {color}; font-weight: 600;">
                        {label}
                    </div>
                    <div style="margin-top: 12px; font-size: 12px; color: #10b981; font-weight: 700;">
                        {'🏆 LOWEST ERROR' if error_rate == min([avg_light_err, avg_dark_err]) else 'Higher Error'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # ======================
        # STATISTICAL SUMMARY
        # ======================
        significant_tasks = sum(p < 0.05 for p in p_values)
        overall_p   = overall_item["p_val"]
        overall_z   = overall_item["z_val"]
        overall_sig = "✅" if overall_p < 0.05 else "❌"

        st.markdown("### 📋 Statistical Summary")

        st.markdown(f"""
        <div style="
            background: #f8fafc; padding: 24px; border-radius: 12px;
            border-left: 4px solid #6366f1;
        ">
            <div style="font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 12px;">
                Overall Findings
            </div>
            <ul style="font-size: 14px; color: #374151; line-height: 1.8; margin: 0;">
                <li><b>{significant_tasks}/3 tasks</b> menunjukkan perbedaan signifikan (p < 0.05)</li>
                <li><b>Overall Wilcoxon:</b> Z={overall_z:.3f}, p={overall_p:.3f} {overall_sig}</li>
                <li>Mean Light Mode: <b>{avg_light_err:.1f}%</b></li>
                <li>Mean Dark Mode: <b>{avg_dark_err:.1f}%</b></li>
                <li>{'Light Mode lebih akurat' if avg_light_err < avg_dark_err else 'Dark Mode lebih akurat'} secara deskriptif</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("*Wilcoxon Signed Ranks Test • SPSS Compatible Output • Mean per User*")
        
# ======================
# UEQ ANALYSIS
# ======================

if menu == "UEQ Analysis":
    st.header("🎯 UEQ Analysis - UEQ Tools Excel")
    st.caption("100% Identik dengan UEQ Tools Excel (AVERAGE per scale)")
    
    # Dataset managers
    with st.expander("📁 Dataset Manager Light Mode", expanded=False):
        dataset_manager(light_df, items, file_ueq_light, "UEQ Light Mode", f"ueq_light_{app}")
    
    st.subheader("☀️ Light Mode Data")
    light_df_edit = st.data_editor(light_df, key="light_ueq", use_container_width=True)
    
    with st.expander("📁 Dataset Manager Dark Mode", expanded=False):
        dataset_manager(dark_df, items, file_ueq_dark, "UEQ Dark Mode", f"ueq_dark_{app}")
    
    st.subheader("🌙 Dark Mode Data")
    dark_df_edit = st.data_editor(dark_df, key="dark_ueq", use_container_width=True)
    
    # Save button
    col_save1, col_save2 = st.columns([3,1])
    with col_save2:
        if st.button("💾 Simpan Semua", type="primary"):
            light_df_edit.to_csv(file_ueq_light, index=False)
            dark_df_edit.to_csv(file_ueq_dark, index=False)
            st.success("✅ Data tersimpan!")
            st.rerun()
    
    # ANALYSIS BUTTON
    if st.button("🚀 ANALISIS UEQ TOOLS", type="secondary"):
        
        # Calculate EXACT UEQ Tools
        light_results = calculate_ueq_tool_style(light_df_edit)
        dark_results = calculate_ueq_tool_style(dark_df_edit)
        
        # Overall scores
        light_overall = light_results["Mean"].mean()
        dark_overall = dark_results["Mean"].mean()
        
        # Combined table
        ueq_table = pd.DataFrame({
            "Scale": light_results["Scale"].values,
            "Light Mean": light_results["Mean"].values,
            "Light Var": light_results["Variance"].values,
            "Dark Mean": dark_results["Mean"].values,
            "Dark Var": dark_results["Variance"].values,
            "Light Overall": light_overall,
            "Dark Overall": dark_overall
        })
        
        # DISPLAY 1: Overall Metrics
        st.markdown("---")
        st.markdown("### 📊 UEQ Overall Scores")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label="Light Mode", 
                value=f"{light_overall:.3f}",
                delta=interpret_ueq(light_overall)
            )
        with col2:
            st.metric(
                label="Dark Mode", 
                value=f"{dark_overall:.3f}",
                delta=interpret_ueq(dark_overall)
            )
        with col3:
            winner = "Light" if light_overall > dark_overall else "Dark"
            color = "normal" if light_overall > dark_overall else "inverse"
            st.metric("Winner", winner, delta_color=color)
        
        # DISPLAY 2: Scale Table
        st.markdown("### 📋 Scale Results (UEQ Tools Excel)")
        st.dataframe(ueq_table.round(3), use_container_width=True)
        
        # DISPLAY 3: Bar Chart
        st.markdown("### 📈 Visual Comparison")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left: Light scales
        scales = ueq_table["Scale"].tolist()
        x = np.arange(len(scales))
        ax1.bar(x, ueq_table["Light Mean"], color="#6366f1", alpha=0.8)
        ax1.set_title("Light Mode")
        ax1.set_xticks(x)
        ax1.set_xticklabels(scales, rotation=45, ha='right')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.grid(True, alpha=0.3)
        
        # Right: Dark scales  
        ax2.bar(x, ueq_table["Dark Mean"], color="#1e293b", alpha=0.8)
        ax2.set_title("Dark Mode")
        ax2.set_xticks(x)
        ax2.set_xticklabels(scales, rotation=45, ha='right')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # DISPLAY 4: Benchmark Cards
        st.markdown("### 🏆 Benchmark UEQ Tools")
        benchmarks = [
            (light_overall, "Light Mode", "#6366f1"),
            (dark_overall, "Dark Mode", "#1e293b")
        ]
        
        for score, label, color in benchmarks:
            rating = interpret_ueq(score)
            rating_color = {
                "Excellent": "#10b981", "Good": "#059669",
                "Above Average": "#f59e0b", "Below Average": "#d97706", "Bad": "#ef4444"
            }.get(rating, "#6b7280")
            
            st.markdown(f"""
            <div style="
                display: flex; align-items: center; 
                padding: 20px; background: white; 
                border-radius: 12px; border: 2px solid {color}20;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            ">
                <div style="flex: 1;">
                    <div style="font-size: 28px; font-weight: 900; color: {color};">{score:.3f}</div>
                    <div style="font-size: 14px; color: {color}; font-weight: 600; margin-top: 4px;">
                        {label}
                    </div>
                </div>
                <div style="
                    padding: 8px 16px; background: {rating_color}20; 
                    color: {rating_color}; border-radius: 20px; 
                    font-weight: 700; font-size: 12px;
                ">
                    {rating}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.caption("*Perhitungan menggunakan AVERAGE() & VAR.P() Excel seperti UEQ Tools*")

                

            

# ======================
# PREFERENSI RESPONDEN
# ======================

if menu == "Preferensi Responden":

    st.header("Preferensi Responden")

    st.caption("Metode Analisis: Mean Preference Analysis menggunakan skala Likert 1–7 untuk menentukan kecenderungan preferensi Light Mode dan Dark Mode")
    
    columns_pref = [
    "Responden",
    "R1","R2","R3","R4",
    "ES1","ES2","ES3","ES4",
    "U1","U2","U3","U4",
    "B1","B2","B3","B4",
    "E1","E2","E3","E4",
    "ED1","ED2","ED3","ED4"
    ]

    # ======================
    # FILE
    # ======================

    file_pos = os.path.join(BASE_DIR, f"preferensi_positif_{app}.csv")
    file_neg = os.path.join(BASE_DIR, f"preferensi_negatif_{app}.csv")

    # ======================
    # DATAFRAME POSITIF
    # ======================

    if os.path.exists(file_pos):
        df_pos = pd.read_csv(file_pos)
    else:
        df_pos = pd.DataFrame(columns=columns_pref)

    df_pos = adjust_dataframe(df_pos,n)

    for c in columns_pref[1:]:
        if c not in df_pos:
            df_pos[c] = 0

    dataset_manager(
        df_pos,
        columns_pref,
        file_pos,
        "Dataset Preferensi Positif",
        f"preferensi_positif_{app}"
    )

    st.subheader("Tabel Rekap - Preferensi Positif")

    df_pos = st.data_editor(df_pos, key="pos_editor")

    if st.button("Simpan Data Positif"):
        df_pos.to_csv(file_pos,index=False)
        st.success("Data preferensi positif tersimpan")

    # ======================
    # DATAFRAME NEGATIF
    # ======================

    if os.path.exists(file_neg):
        df_neg = pd.read_csv(file_neg)
    else:
        df_neg = pd.DataFrame(columns=columns_pref)

    df_neg = adjust_dataframe(df_neg,n)

    for c in columns_pref[1:]:
        if c not in df_neg:
            df_neg[c] = 0
    
    dataset_manager(
        df_neg,
        columns_pref,
        file_neg,
        "Dataset Preferensi Negatif",
        f"preferensi_negatif_{app}"
    )

    st.subheader("Tabel Rekap - Preferensi Negatif")

    df_neg = st.data_editor(df_neg, key="neg_editor")

    if st.button("Simpan Data Negatif"):
        df_neg.to_csv(file_neg,index=False)
        st.success("Data preferensi negatif tersimpan")

    # ======================
    # ANALISIS
    # ======================

    if st.button("Analisis Preferensi"):

        aspek = {

        "Keterbacaan (Readability)" : ["R1","R2","R3","R4"],
        "Kelelahan Mata (Eye Strain)" : ["ES1","ES2","ES3","ES4"],
        "Usability" : ["U1","U2","U3","U4"],
        "Konsumsi Baterai" : ["B1","B2","B3","B4"],
        "Efisien Kinerja" : ["E1","E2","E3","E4"],
        "Estetika & Daya Tarik" : ["ED1","ED2","ED3","ED4"]

        }

    # ======================
    # POSITIF
    # ======================

        pos_results = []

        for a,cols in aspek.items():

            pos_numeric = df_pos[cols].apply(pd.to_numeric, errors="coerce")

            mean_val = pos_numeric.mean().mean()

            if mean_val < 4:
                prefer = "Light Mode"
            elif mean_val > 4:
                prefer = "Dark Mode"
            else:
                prefer = "Netral"

            pos_results.append({
                "Aspek":a,
                "Mean":round(mean_val,2),
                "Preferensi":prefer
            })

        pos_df = pd.DataFrame(pos_results)

        st.subheader("Tabel Rekap - Preferensi Positif")

        st.dataframe(pos_df,use_container_width=True)

    # ======================
    # NEGATIF
    # ======================

        neg_results = []

        for a,cols in aspek.items():

            neg_numeric = df_neg[cols].apply(pd.to_numeric, errors="coerce")

            mean_val = neg_numeric.mean().mean()

            if mean_val < 4:
                prefer = "Light Mode"
            elif mean_val > 4:
                prefer = "Dark Mode"
            else:
                prefer = "Netral"

            neg_results.append({
                "Aspek":a,
                "Mean":round(mean_val,2),
                "Preferensi":prefer
            })

        neg_df = pd.DataFrame(neg_results)

        st.subheader("Tabel Rekap - Preferensi Negatif")

        st.dataframe(neg_df,use_container_width=True)

    # ======================
    # CHART
    # ======================

        st.subheader("Visualisasi Preferensi")

        light = sum(pos_df["Preferensi"]=="Light Mode") + sum(neg_df["Preferensi"]=="Light Mode")
        dark = sum(pos_df["Preferensi"]=="Dark Mode") + sum(neg_df["Preferensi"]=="Dark Mode")

        fig,ax = plt.subplots(figsize=(6,4))

        ax.bar(["Light Mode","Dark Mode"],[light,dark])

        ax.set_ylabel("Jumlah Aspek")
        ax.set_title("Preferensi Mode Berdasarkan 6 Aspek")

        st.pyplot(fig)

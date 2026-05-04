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
    Menampilkan tabel Ranks dan Test Statistics.
    """
    ranks_rows = ""
    footnotes = []
    abc = "abcdefghijklmnopqrstuvwxyz"
    fn_idx = 0

    for pd_item in pairs_data:
        vn = pd_item["var_name"]
        l_lbl = pd_item["light_lbl"]
        d_lbl = pd_item["dark_lbl"]

        labels = ["", "", ""]
        hubungan = [
            f"{d_lbl} < {l_lbl}",
            f"{d_lbl} > {l_lbl}",
            f"{l_lbl} = {d_lbl}"
        ]

        n_vals = [pd_item['neg_n'], pd_item['pos_n'], pd_item['ties_n']]

        for i in range(3):
            current_letter = abc[fn_idx]
            if n_vals[i] > 0:
                labels[i] = f"<sup>{current_letter}</sup>"
                footnotes.append(f"{current_letter}. {hubungan[i]}")
            fn_idx += 1

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

    footnote_html = "<br>".join(footnotes) if footnotes else ""

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

    # ======================
    # TEST STATISTICS (SPSS STYLE)
    # ======================
    def get_z_superscript(pd_item):
        """
        Tentukan superscript Z sesuai SPSS:
        - Z negatif → based on negative ranks → superscript 'b'
        - Z positif → based on positive ranks → superscript 'c'
        """
        z = pd_item["z_val"]
        neg_n = pd_item["neg_n"]
        pos_n = pd_item["pos_n"]
        if neg_n == 0 and pos_n == 0:
            return "", ""
        elif z <= 0:
            return "b", "b. Based on negative ranks."
        else:
            return "c", "c. Based on positive ranks."

    # Header kolom: nama variabel tiap pasangan
    test_stat_headers = "".join([
        f'<th style="border:1px solid #aaa;padding:7px 12px;text-align:center;font-size:12px;">{p["var_name"]}</th>'
        for p in pairs_data
    ])

    # Baris Z dengan superscript
    z_footnotes_dict = {}
    z_cells = ""
    for p in pairs_data:
        sup, note = get_z_superscript(p)
        if sup and sup not in z_footnotes_dict:
            z_footnotes_dict[sup] = note
        z_cells += f'<td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{p["z_val"]:.3f}<sup>{sup}</sup></td>'

    # Baris Asymp. Sig
    p_cells = "".join([
        f'<td style="border:1px solid #bbb;padding:7px 12px;text-align:right;">{p["p_val"]:.3f}</td>'
        for p in pairs_data
    ])

    # Footnote Z
    z_footnote_html = "<br>".join(z_footnotes_dict.values())

    test_stats_html = f"""
    <div style="margin:16px 0 24px 0;">
        <div style="font-weight:700;font-size:14px;border-bottom:2px solid #333;padding-bottom:4px;margin-bottom:0;">
            Test Statistics<sup>a</sup>
        </div>
        <table style="border-collapse:collapse;font-size:13px;font-family:Arial,sans-serif;width:100%;">
            <thead>
                <tr style="background:#d9d9d9;">
                    <th style="border:1px solid #aaa;padding:7px 12px;text-align:left;"></th>
                    {test_stat_headers}
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="border:1px solid #bbb;padding:7px 12px;font-weight:600;background:#f5f5f5;">Z</td>
                    {z_cells}
                </tr>
                <tr>
                    <td style="border:1px solid #bbb;padding:7px 12px;font-weight:600;background:#eaf4ff;">Asymp. Sig. (2-tailed)</td>
                    {p_cells}
                </tr>
            </tbody>
        </table>
        <div style="font-size:11px;color:#444;margin-top:5px;font-style:italic;line-height:1.8;">
            a. Wilcoxon Signed Ranks Test<br>
            {z_footnote_html}
        </div>
    </div>"""

    st.markdown(ranks_html + test_stats_html, unsafe_allow_html=True)


def dataset_manager(df, expected_columns, save_path, title, filename_base):

    st.markdown(f"""
    <div style="font-size:16px;font-weight:600;color:#1e293b;margin-bottom:8px;">
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
            ["Excel (.xlsx)", "CSV (.csv)", "PDF (.pdf)"],
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
            elements.append(Spacer(1, 20))
            data = [df.columns.tolist()] + df.values.tolist()
            table = Table(data)
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTSIZE", (0, 0), (-1, -1), 8)
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
    elif action == "Import Dataset":

        # Inisialisasi session state untuk status import
        import_key = f"import_status_{filename_base}"
        if import_key not in st.session_state:
            st.session_state[import_key] = None  # None | "success" | "error"

        # Tampilkan pesan hasil import jika ada
        if st.session_state[import_key] == "success":
            st.success("Dataset berhasil diimport!")
            if st.button("Upload File Lain", key=f"reset_import_{filename_base}"):
                st.session_state[import_key] = None
                st.rerun()
            return  # Hentikan render form upload

        uploaded_file = st.file_uploader(
            "Upload file dataset",
            type=["xlsx", "csv"],
            key=f"upload_{filename_base}"
        )

        if uploaded_file is not None:

            if uploaded_file.name.endswith(".xlsx"):
                df_new = pd.read_excel(uploaded_file)
            else:
                df_new = pd.read_csv(uploaded_file)

            if list(df_new.columns) != expected_columns:
                st.error("Struktur dataset tidak sesuai. Pastikan kolom file sama persis dengan template.")
            else:
                st.markdown("**Preview Data:**")
                st.dataframe(df_new.head(5), use_container_width=True)

                if st.button(
                    "Konfirmasi Import",
                    type="primary",
                    use_container_width=True,
                    key=f"confirm_import_{filename_base}"
                ):
                    try:
                        df_new.to_csv(save_path, index=False)
                        st.session_state[import_key] = "success"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Gagal menyimpan: {e}")
            



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
    background-color: {bg_sidebar} !important;
    border-right: 1px solid {border} !important;
}
[data-testid="stSidebar"] .stMarkdown p, 
[data-testid="stSidebar"] label, 
[data-testid="stSidebar"] .sidebar-title {
    color: {text_sidebar} !important;
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
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
            
.stApp {
    background-color: {bg_main};
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

* {
    transition: background 0.3s ease, color 0.3s ease;
}

[data-testid="stMetricV2"] {
    background-color: {bg_card};
    color:{text_main};
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
    background: {bg_card} !important;
    padding: 24px;
    border-radius: 20px;
    border: 1px solid {border} !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    transition: all 0.3s ease;
    height: 100%;
    color: {text_main} !important;
}

body, p, span, div {
    color: {text_main} !important;
}
            
.stAlert {
    background-color: {bg_insight} !important;
    color: {text_main} !important;
}

details {
    background: {bg_card} !important;
    border: 1px solid {border} !important;
    border-radius: 8px;
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
    color: {text_main} !important;
    line-height: 1.2;
}

[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #f8fafc !important;
    color: #1e293b !important;
    border: 1px solid #e2e8f0 !important;
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
    background-color: {bg_card};
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

# ======================
# SIDEBAR UI (MODERN VERSION)
# ======================
with st.sidebar:
    # Branding Header
    st.markdown(f"""
        <div style="text-align: center; padding: 10px 0 25px 0;">
            <h1 style='font-size: 24px; color: #6366f1; margin-bottom: 0;'>UX Analytics</h1>
            <p style='font-size: 10px; color: #94a3b8; text-transform: uppercase; letter-spacing: 2px; font-weight: 600;'>Universitas Islam Indonesia</p>
        </div>
    """, unsafe_allow_html=True)

    # --- SECTION 1: RESEARCH OBJECT ---
    st.markdown('<p class="menu-label">Research Object</p>', unsafe_allow_html=True)
    
    # Pilih aplikasi aktif
    app = st.selectbox("Aplikasi Analisis", st.session_state.app_list, label_visibility="collapsed")
    
    # Kelola Aplikasi (Pop-over style expander)
    with st.expander("Manage Applications", expanded=False):
        new_app = st.text_input("Nama Aplikasi Baru", placeholder="Contoh: Instagram")
        if st.button("Add Object", use_container_width=True):
            if new_app and new_app.strip() not in st.session_state.app_list:
                st.session_state.app_list.append(new_app.strip())
                pd.Series(st.session_state.app_list).to_json(APP_FILE)
                st.rerun()
        
        if st.session_state.app_list:
            st.markdown("---")
            app_delete = st.selectbox("Hapus Aplikasi", st.session_state.app_list, key="del_select")
            if st.button("Delete Object", use_container_width=True, type="secondary"):
                st.session_state.app_list.remove(app_delete)
                pd.Series(st.session_state.app_list).to_json(APP_FILE)
                st.rerun()

    st.markdown("<div style='margin: 10px 0;'></div>", unsafe_allow_html=True)

    # --- SECTION 2: MAIN NAVIGATION ---
    st.markdown('<p class="menu-label">Main Navigation</p>', unsafe_allow_html=True)
    menu = st.selectbox(
        "Pilih Menu", 
        ["Home", "Overview", "Time on Task", "Error Rate", "UEQ Analysis", "Preferensi Responden"],
        label_visibility="collapsed"
    )

    # --- SECTION 3: PARAMETERS ---
    st.markdown('<p class="menu-label">Study Parameters</p>', unsafe_allow_html=True)
    n = st.number_input("Sample Size (N)", min_value=1, max_value=100, value=25, help="Jumlah responden dalam penelitian ini")

    # Info Card Aktif
    st.markdown(f"""
        <div style="background-color: #f1f5f9; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 20px;">
            <div style="font-size: 9px; color: #6366f1; font-weight: 800; text-transform: uppercase; margin-bottom: 5px;">Project Insight</div>
            <div style="font-size: 13px; font-weight: 700; color: #1e293b;">{app if app else "No App"} Study</div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Status: <span style="color:#10b981; font-weight:600;">Active Analysis</span></div>
        </div>
    """, unsafe_allow_html=True)

    # --- SECTION 4: SYSTEM ---
    st.markdown("<div style='flex-grow: 1;'></div>", unsafe_allow_html=True) # Push reset button to bottom
    st.markdown("---")
    
    if not st.session_state.confirm_reset:
        if st.button("RESET SYSTEM DATA", use_container_width=True):
            st.session_state.confirm_reset = True
            st.rerun()
    else:
        st.markdown('<p style="font-size:11px; color:#ef4444; text-align:center; margin-bottom:5px;"><b>Konfirmasi Hapus?</b></p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Batal", use_container_width=True):
                st.session_state.confirm_reset = False
                st.rerun()
        with c2:
            if st.button("Ya", type="primary", use_container_width=True):
                # ... (logika hapus file kamu di sini) ...
                st.session_state.confirm_reset = False
                st.success("Cleaned!")
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
    df = df.copy().apply(pd.to_numeric, errors='coerce')
    df_transformed = df - 4
    
    # BENAR: hilangkan 10 dan 24 dari list
    reverse_items = [3, 4, 5, 9, 12, 17, 18, 19, 21, 23, 25]
    for i in reverse_items:
        col = f"I{i}"
        if col in df_transformed.columns:
            df_transformed[col] = -df_transformed[col]
    return df_transformed


def ueq_transform(df):
    """
    Transformasi data mentah (1–7) ke skala -3 s.d. +3.
    Identik dengan sheet DT di UEQ Tools Excel v13.
    """
    df = df.copy().apply(pd.to_numeric, errors="coerce")
    df_t = df - 4
    for i in REVERSE_ITEMS:
        col = f"I{i}"
        if col in df_t.columns:
            df_t[col] = -df_t[col]
    return df_t

def calculate_ueq_tool_style(df):
    """LOGIKA FINAL: Menghitung mean persis UEQ Analysis Tool"""
    df_proc = preprocess_ueq(df)
    
    # Mapping item ke skala sesuai standar UEQ
    scales_map = {
        "Daya tarik": [1, 12, 14, 16, 24, 25],
        "Kejelasan": [2, 4, 13, 21],
        "Efisiensi": [9, 20, 22, 23],
        "Ketepatan": [8, 11, 17, 19],
        "Stimulasi": [5, 6, 7, 18],
        "Kebaruan": [3, 10, 15, 26]
    }
    
    results = []
    for scale_name, item_indices in scales_map.items():
        cols = [f"I{i}" for i in item_indices]
        
        # LOGIKA KRUSIAL: 
        # 1. Hitung rata-rata per baris (per responden) terlebih dahulu
        # 2. Kemudian hitung rata-rata dari hasil per responden tersebut
        scale_means_per_person = df_proc[cols].mean(axis=1)
        final_scale_mean = scale_means_per_person.mean()
        
        # Hitung varians (menggunakan ddof=1 sesuai statistik sampel)
        final_variance = scale_means_per_person.var()
        
        results.append({
            "Scale": scale_name,
            "Mean": round(final_scale_mean, 6),
            "Variance": round(final_variance, 6)
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
    # HERO SECTION (Visi Platform Masa Depan)
    # ======================
    st.markdown(f"""
    <div style="
    background: linear-gradient(135deg,#4f46e5,#6366f1);
    padding:40px;
    border-radius:20px;
    color:white;
    margin-bottom:30px;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
    ">
        <div style="font-size:28px;font-weight:800;letter-spacing:-0.5px;">
            UX Research Analytics Platform v1.0
        </div>
        <div style="font-size:14px;margin-top:12px;max-width:700px;line-height:1.7;opacity:0.9;">
            Platform ini dikembangkan sebagai solusi Standardisasi Analisis UX untuk penelitian selanjutnya. 
            Mengintegrasikan metodologi Within-Subject Design dengan pengolahan data otomatis berbasis 
            Python Streamlit untuk menghasilkan insight yang cepat, akurat, dan interaktif.
        </div>
        <div style="margin-top:20px; display: flex; gap: 15px;">
            <div style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 30px; font-size: 11px; font-weight: 600;">
                Current Object: {app if app else "None"}
            </div>
            <div style="background: rgba(255,255,255,0.2); padding: 8px 15px; border-radius: 30px; font-size: 11px; font-weight: 600;">
                Sample Size: {n}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # VISI & TUJUAN PLATFORM
    # ======================
    st.markdown("### Future Research Foundation")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:14px; font-weight:700; color:#4f46e5; margin-bottom:10px;">Automated Standardization</div>
            <div style="font-size:12px; line-height:1.6; color:{text_main};">
                Menghilangkan proses manual dalam perhitungan statistik Wilcoxon dan UEQ Benchmark. 
                Platform ini memastikan bahwa penelitian UX di masa mendatang memiliki standar perhitungan yang konsisten dan meminimalisir human error.
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown(f"""
        <div class="card">
            <div style="font-size:14px; font-weight:700; color:#4f46e5; margin-bottom:10px;">Interactivity & Visual Insight</div>
            <div style="font-size:12px; line-height:1.6; color:{text_main};">
                Bukan sekadar angka statis, platform ini menyediakan visualisasi dinamis yang mempermudah peneliti 
                dalam melakukan interpretasi data secara mendalam terhadap perilaku responden.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ======================
    # INTEGRATED METRICS
    # ======================
    st.markdown("### Integrated Analysis Modules", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    
    modules = [
        ("Time on Task", "Efisiensi kinerja pengguna dalam satuan waktu detik."),
        ("Error Rate", "Akurasi interaksi dan identifikasi hambatan sistem."),
        ("UEQ Standard", "Metrik kepuasan fungsional dan emosional pengguna."),
        ("Preference", "Analisis kecenderungan pilihan subjektif responden.")
    ]

    for col, (title, desc) in zip([col1, col2, col3, col4], modules):
        col.markdown(f"""
        <div class="card" style="text-align:center;">
            <div style="font-size:13px; font-weight:800; color:{text_main}; margin-bottom:5px;">{title}</div>
            <div style="font-size:10px; color:{text_soft};">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ======================
    # RESEARCH FOOTER
    # ======================
    st.markdown("<br>", unsafe_allow_html=True)
    st.info(f"""
    Panduan Penggunaan:
    1. Tentukan Research Object pada sidebar (saat ini: {app}).
    2. Masukkan data mentah pada masing-masing modul (ToT, Error, UEQ, atau Preferensi).
    3. Lihat hasil akumulasi signifikansi penelitian pada menu Overview untuk mendapatkan kesimpulan akhir secara otomatis.
    """)


# ======================
# OVERVIEW
# ======================

if menu == "Overview":

    # ======================
    # HITUNG SEMUA METRICS DULU
    # ======================
    avg_light_tot = df_tot[["Light_T1","Light_T2","Light_T3"]].mean().mean()
    avg_dark_tot  = df_tot[["Dark_T1","Dark_T2","Dark_T3"]].mean().mean()
    avg_light_err = df_error[["Light_T1","Light_T2","Light_T3"]].mean().mean()
    avg_dark_err  = df_error[["Dark_T1","Dark_T2","Dark_T3"]].mean().mean()

    light_ueq_mean = calculate_ueq_tool_style(light_df)["Mean"].mean()
    dark_ueq_mean  = calculate_ueq_tool_style(dark_df)["Mean"].mean()

    file_pos = os.path.join(BASE_DIR, f"preferensi_positif_{app}.csv")
    file_neg = os.path.join(BASE_DIR, f"preferensi_negatif_{app}.csv")

    aspek = {
        "Readability": ["R1","R2","R3","R4"],
        "Eye Strain":  ["ES1","ES2","ES3","ES4"],
        "Usability":   ["U1","U2","U3","U4"],
        "Battery":     ["B1","B2","B3","B4"],
        "Efficiency":  ["E1","E2","E3","E4"],
        "Aesthetic":   ["ED1","ED2","ED3","ED4"],
    }

    aspek_result = []
    if os.path.exists(file_pos) and os.path.exists(file_neg):
        df_pos = pd.read_csv(file_pos).fillna(0)
        df_neg = pd.read_csv(file_neg).fillna(0)
        for a, cols in aspek.items():
            pos_val = df_pos[cols].mean().mean()
            neg_val = (8 - df_neg[cols]).mean().mean()
            if pd.isna(pos_val) or pd.isna(neg_val):
                continue
            aspek_result.append("Light Mode" if (pos_val + neg_val) / 2 < 4 else "Dark Mode")

    light_pref = aspek_result.count("Light Mode")
    dark_pref  = aspek_result.count("Dark Mode")
    best_pref  = "Light Mode" if light_pref >= dark_pref else "Dark Mode"

    # ======================
    # HEADER
    # ======================
    st.markdown(f"""
    <div style="margin-bottom:28px;">
        <div style="font-size:24px;font-weight:700;color:#1E293B;letter-spacing:-0.3px;">
            Research Overview
        </div>
        <div style="font-size:13px;color:#64748B;margin-top:3px;">
            {app} &nbsp;·&nbsp; {n} responden &nbsp;·&nbsp; Within-Subject Design
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # KPI CARDS — 4 METRIK + 1 KESIMPULAN
    # ======================
    def _winner_badge(wins):
        if wins:
            return '<span style="font-size:9px;font-weight:600;background:#EEF2FF;color:#4338CA;padding:2px 7px;border-radius:20px;margin-left:5px;vertical-align:middle;">BEST</span>'
        return ""

    def _kpi(title, l_val, d_val, unit, lower_is_better=False):
        l_wins = (l_val < d_val) if lower_is_better else (l_val > d_val)
        d_wins = not l_wins
        return f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;
            padding:20px 18px;height:100%;">
            <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:16px;">{title}</div>
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-size:12px;color:#64748B;display:flex;align-items:center;gap:6px;">
                        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                            background:#6366F1;flex-shrink:0;"></span>Light
                    </div>
                    <div style="font-size:16px;font-weight:700;color:#4338CA;">
                        {round(l_val, 2)}{unit}{_winner_badge(l_wins)}
                    </div>
                </div>
                <div style="height:1px;background:#F1F5F9;"></div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="font-size:12px;color:#64748B;display:flex;align-items:center;gap:6px;">
                        <span style="display:inline-block;width:6px;height:6px;border-radius:50%;
                            background:#334155;flex-shrink:0;"></span>Dark
                    </div>
                    <div style="font-size:16px;font-weight:700;color:#334155;">
                        {round(d_val, 2)}{unit}{_winner_badge(d_wins)}
                    </div>
                </div>
            </div>
        </div>"""

    col_a, col_b, col_c, col_d, col_e = st.columns(5)

    with col_a:
        st.markdown(_kpi("UEQ Score", light_ueq_mean, dark_ueq_mean, ""), unsafe_allow_html=True)
    with col_b:
        st.markdown(_kpi("Time on Task", avg_light_tot, avg_dark_tot, "s", lower_is_better=True), unsafe_allow_html=True)
    with col_c:
        st.markdown(_kpi("Error Rate", avg_light_err, avg_dark_err, "%", lower_is_better=True), unsafe_allow_html=True)
    with col_d:
        pref_color = "#4338CA" if best_pref == "Light Mode" else "#1E293B"
        pref_bg    = "#EEF2FF" if best_pref == "Light Mode" else "#F1F5F9"
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;
            padding:20px 18px;height:100%;">
            <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:16px;">Best Preference</div>
            <div style="font-size:18px;font-weight:700;color:{pref_color};margin-bottom:8px;">
                {best_pref}
            </div>
            <div style="display:inline-block;font-size:10px;font-weight:600;
                background:{pref_bg};color:{pref_color};
                padding:3px 10px;border-radius:20px;">
                {light_pref} Light &nbsp;·&nbsp; {dark_pref} Dark
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col_e:
        st.markdown(f"""
        <div style="background:#6366F1;border-radius:14px;padding:20px 18px;height:100%;color:white;">
            <div style="font-size:10px;font-weight:700;text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:16px;opacity:0.7;">Objek Studi</div>
            <div style="font-size:20px;font-weight:700;margin-bottom:6px;">{app}</div>
            <div style="font-size:11px;opacity:0.7;">N = {n} responden</div>
            <div style="font-size:11px;opacity:0.7;">3 tugas per mode</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ======================
    # CHART ROW — ToT, Error, UEQ side by side (Plotly, bersih)
    # ======================
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:12px;">Perbandingan Metrik</div>
    """, unsafe_allow_html=True)

    col_g1, col_g2, col_g3 = st.columns(3)

    def _bar_chart(title, l_val, d_val, unit):
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Light", "Dark"],
            y=[l_val, d_val],
            marker_color=["#6366F1", "#334155"],
            text=[f"{round(l_val,1)}{unit}", f"{round(d_val,1)}{unit}"],
            textposition="outside",
            textfont=dict(size=12, color=["#4338CA", "#334155"]),
            width=0.45,
        ))
        fig.update_layout(
            title=dict(text=title, font=dict(size=12, color="#64748B"), x=0, xanchor="left"),
            yaxis=dict(showgrid=True, gridcolor="#F1F5F9", zeroline=False,
                       tickfont=dict(size=10, color="#94A3B8"), showline=False),
            xaxis=dict(tickfont=dict(size=11, color="#374151"), showline=False),
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=36, b=16, l=8, r=8),
            height=220,
            showlegend=False,
        )
        return fig

    with col_g1:
        st.plotly_chart(_bar_chart("Time on Task (detik)", avg_light_tot, avg_dark_tot, "s"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_g2:
        st.plotly_chart(_bar_chart("Error Rate (%)", avg_light_err, avg_dark_err, "%"),
                        use_container_width=True, config={"displayModeBar": False})
    with col_g3:
        st.plotly_chart(_bar_chart("UEQ Score", light_ueq_mean, dark_ueq_mean, ""),
                        use_container_width=True, config={"displayModeBar": False})

    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)

    # ======================
    # PREFERENSI DONUT — POSITIF & NEGATIF
    # ======================
    pos_percent = {a: 0.0 for a in aspek.keys()}
    neg_percent = {a: 0.0 for a in aspek.keys()}

    if os.path.exists(file_pos):
        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        raw_p = {}
        for a, cols in aspek.items():
            ec = [c for c in cols if c in df_pos_raw.columns]
            if ec:
                v = df_pos_raw[ec].apply(pd.to_numeric, errors="coerce").mean().mean()
                raw_p[a] = v if not pd.isna(v) and v > 0 else 0
        tot = sum(raw_p.values())
        if tot > 0:
            pos_percent = {k: (v / tot) * 100 for k, v in raw_p.items()}

    if os.path.exists(file_neg):
        df_neg_raw = pd.read_csv(file_neg).fillna(0)
        raw_n = {}
        for a, cols in aspek.items():
            ec = [c for c in cols if c in df_neg_raw.columns]
            if ec:
                v = df_neg_raw[ec].apply(pd.to_numeric, errors="coerce").mean().mean()
                raw_n[a] = v if not pd.isna(v) and v > 0 else 0
        tot = sum(raw_n.values())
        if tot > 0:
            neg_percent = {k: (v / tot) * 100 for k, v in raw_n.items()}

    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:12px;">Distribusi Preferensi</div>
    """, unsafe_allow_html=True)

    col_p1, col_p2 = st.columns(2)

    def _donut_section(col, label, percent_dict, palette):
        with col:
            with st.container(border=True):
                st.markdown(f"""
                <div style="font-size:13px;font-weight:600;color:#1E293B;
                    margin-bottom:12px;">{label}</div>
                """, unsafe_allow_html=True)
                if any(v > 0 for v in percent_dict.values()):
                    c_left, c_right = st.columns([1, 1.3])
                    with c_left:
                        fig = go.Figure(data=[go.Pie(
                            labels=list(percent_dict.keys()),
                            values=list(percent_dict.values()),
                            hole=0.62,
                            marker=dict(colors=palette, line=dict(color="#FFFFFF", width=2)),
                            textinfo="none",
                            showlegend=False,
                            hoverinfo="label+percent",
                        )])
                        fig.update_layout(
                            margin=dict(t=0, b=0, l=0, r=0),
                            height=160,
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                        )
                        st.plotly_chart(fig, use_container_width=True,
                                        config={"displayModeBar": False})
                    with c_right:
                        legend = ""
                        for i, (name, val) in enumerate(percent_dict.items()):
                            c = palette[i % len(palette)]
                            legend += f"""
                            <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:7px;">
                                <div style="display:flex;align-items:center;gap:7px;">
                                    <div style="width:7px;height:7px;border-radius:50%;
                                        background:{c};flex-shrink:0;"></div>
                                    <span style="font-size:11px;color:#374151;">{name}</span>
                                </div>
                                <span style="font-size:11px;font-weight:600;color:#1E293B;">
                                    {round(val,1)}%
                                </span>
                            </div>"""
                        st.markdown(legend, unsafe_allow_html=True)
                else:
                    st.caption("Data belum tersedia.")

    _donut_section(col_p1, "Preferensi Positif",
                   pos_percent,
                   ["#4338CA","#4F46E5","#6366F1","#818CF8","#A5B4FC","#C7D2FE"])
    _donut_section(col_p2, "Preferensi Negatif",
                   neg_percent,
                   ["#1E3A8A","#1E40AF","#1D4ED8","#2563EB","#3B82F6","#60A5FA"])

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ======================
    # STATISTICAL SIGNIFICANCE
    # ======================
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;
        letter-spacing:0.08em;margin-bottom:12px;">Signifikansi Statistik</div>
    """, unsafe_allow_html=True)

    tot_empty = df_tot[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    err_empty = df_error[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    ueq_empty = light_df.replace(4, np.nan).dropna(how="all").empty and dark_df.replace(4, np.nan).dropna(how="all").empty

    p_tot = np.nan
    if not tot_empty:
        _, p_tot = stats.ttest_rel(
            df_tot[["Light_T1","Light_T2","Light_T3"]].mean(axis=1),
            df_tot[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)
        )

    p_err = np.nan
    if not err_empty:
        _, p_err = stats.ttest_rel(
            df_error[["Light_T1","Light_T2","Light_T3"]].mean(axis=1),
            df_error[["Dark_T1","Dark_T2","Dark_T3"]].mean(axis=1)
        )

    p_ueq = np.nan
    if not ueq_empty:
        _, p_ueq = stats.ttest_rel(
            preprocess_ueq(light_df).mean(axis=1),
            preprocess_ueq(dark_df).mean(axis=1)
        )

    pref_diff = []
    if os.path.exists(file_pos) and os.path.exists(file_neg):
        df_pos_raw = pd.read_csv(file_pos).fillna(0)
        df_neg_raw = pd.read_csv(file_neg).fillna(0)
        cols_all = [c for cols in aspek.values() for c in cols]
        ep = [c for c in cols_all if c in df_pos_raw.columns]
        en = [c for c in cols_all if c in df_neg_raw.columns]
        if ep and en:
            pv = df_pos_raw[ep].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            nv = (8 - df_neg_raw[en].apply(pd.to_numeric, errors="coerce")).mean(axis=1)
            pref_diff = (pv - nv).dropna().tolist()

    p_pref = np.nan
    if len(pref_diff) > 1:
        _, p_pref = stats.ttest_1samp(pref_diff, 0)

    grand_p = np.nanmean([p_tot, p_err, p_ueq, p_pref])
    p_val_final = grand_p

    is_sig       = not pd.isna(grand_p) and grand_p < 0.05
    grand_color  = "#166534" if is_sig else "#92400E"
    grand_bg     = "#F0FDF4" if is_sig else "#FFFBEB"
    grand_border = "#BBF7D0" if is_sig else "#FDE68A"
    grand_label  = "Signifikan" if is_sig else "Tidak Signifikan"

    stats_rows = [
        ("Time on Task",        "Paired T-Test",   p_tot),
        ("Error Rate",          "Paired T-Test",   p_err),
        ("UEQ Analysis",        "Paired T-Test",   p_ueq),
        ("Preferensi Subjektif","1-Sample T-Test", p_pref),
    ]

    col_grand, col_detail = st.columns([1, 2.6])

    with col_grand:
        p_display = f"{round(grand_p, 4)}" if not pd.isna(grand_p) else "—"
        st.markdown(f"""
        <div style="background:{grand_bg};border:1px solid {grand_border};border-radius:14px;
            padding:28px 20px;text-align:center;height:100%;display:flex;flex-direction:column;
            justify-content:center;align-items:center;min-height:200px;">
            <div style="font-size:10px;font-weight:700;color:{grand_color};text-transform:uppercase;
                letter-spacing:0.08em;margin-bottom:10px;">Overall p-value</div>
            <div style="font-size:38px;font-weight:700;color:{grand_color};line-height:1;">
                {p_display}
            </div>
            <div style="margin-top:14px;display:inline-block;font-size:11px;font-weight:600;
                background:{grand_color};color:white;padding:5px 16px;border-radius:20px;">
                {grand_label}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_detail:
        rows_html = ""
        for label, method, pv in stats_rows:
            if pd.isna(pv):
                sig_color = "#94A3B8"
                p_text    = "&#8212;"
                sig_text  = "Belum ada data"
                dot_bg    = "#E2E8F0"
            elif pv < 0.05:
                sig_color = "#166534"
                p_text    = str(round(pv, 4))
                sig_text  = "Signifikan"
                dot_bg    = "#86EFAC"
            else:
                sig_color = "#94A3B8"
                p_text    = str(round(pv, 4))
                sig_text  = "Tidak Signifikan"
                dot_bg    = "#E2E8F0"

            rows_html += (
                '<div style="display:flex;justify-content:space-between;align-items:center;'
                'padding:11px 0;border-bottom:1px solid #F8FAFC;">'
                  '<div style="display:flex;align-items:center;gap:10px;">'
                    '<div style="width:7px;height:7px;border-radius:50%;'
                    'background:' + dot_bg + ';flex-shrink:0;"></div>'
                    '<div>'
                      '<div style="font-size:13px;font-weight:600;color:#334155;">' + label + '</div>'
                      '<div style="font-size:10px;color:#6366F1;margin-top:1px;">' + method + '</div>'
                    '</div>'
                  '</div>'
                  '<div style="text-align:right;">'
                    '<div style="font-size:14px;font-weight:700;color:' + sig_color + ';">p = ' + p_text + '</div>'
                    '<div style="font-size:9px;color:' + sig_color + ';font-weight:600;'
                    'text-transform:uppercase;margin-top:1px;">' + sig_text + '</div>'
                  '</div>'
                '</div>'
            )

        card_html = (
            '<div style="border:1px solid #E2E8F0;border-radius:14px;padding:20px 22px;height:100%;">'
              '<div style="font-size:12px;font-weight:600;color:#374151;margin-bottom:12px;">'
                'Rincian Metode Analisis'
              '</div>'
            + rows_html +
            '</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

    st.markdown("<div style='margin-top:28px;'></div>", unsafe_allow_html=True)

    # ======================
    # KESIMPULAN AKHIR
    # ======================
    data_tot_empty = df_tot[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    data_err_empty = df_error[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0
    data_ueq_empty = light_df.sum().sum() == 0 and dark_df.sum().sum() == 0
    pref_empty     = not (os.path.exists(file_pos) and os.path.exists(file_neg))

    if data_tot_empty or data_err_empty or data_ueq_empty or pref_empty:
        st.info("Silakan input semua data penelitian terlebih dahulu untuk menampilkan kesimpulan.")
    else:
        result_light   = calculate_ueq_tool_style(light_df)
        result_dark    = calculate_ueq_tool_style(dark_df)
        ueq_scale_m    = {result_light.loc[i,"Scale"]: {"l": result_light.loc[i,"Mean"], "d": result_dark.loc[i,"Mean"]} for i in range(len(result_light))}
        best_ueq_scale = max(ueq_scale_m, key=lambda s: max(ueq_scale_m[s]["l"], ueq_scale_m[s]["d"]))

        win_ueq = "Light" if light_ueq_mean > dark_ueq_mean else "Dark"
        win_tot = "Light" if avg_light_tot < avg_dark_tot else "Dark"
        win_err = "Light" if avg_light_err < avg_dark_err else "Dark"

        pref_scores = {}
        if os.path.exists(file_pos) and os.path.exists(file_neg):
            df_pos_r = pd.read_csv(file_pos).fillna(0)
            df_neg_r = pd.read_csv(file_neg).fillna(0)
            for a, cols in aspek.items():
                pv2 = df_pos_r[cols].apply(pd.to_numeric, errors="coerce").mean().mean()
                nv2 = (8 - df_neg_r[cols].apply(pd.to_numeric, errors="coerce")).mean().mean()
                pref_scores[a] = (pv2 + nv2) / 2

        best_pref_aspect  = max(pref_scores, key=pref_scores.get) if pref_scores else "—"
        worst_pref_aspect = min(pref_scores, key=pref_scores.get) if pref_scores else "—"

        aspect_scores = {}
        if pref_scores:
            for a, score in pref_scores.items():
                aspect_scores[a] = {"score": score, "mode": "LIGHT" if score < 4 else "DARK"}
        best_aspect       = max(aspect_scores, key=lambda x: aspect_scores[x]["score"]) if aspect_scores else "—"
        best_aspect_mode  = aspect_scores[best_aspect]["mode"] if aspect_scores else "—"
        worst_aspect      = min(aspect_scores, key=lambda x: aspect_scores[x]["score"]) if aspect_scores else "—"
        worst_aspect_mode = aspect_scores[worst_aspect]["mode"] if aspect_scores else "—"

        sig_label  = "Signifikan" if p_val_final < 0.05 else "Tidak Signifikan"
        sig_color  = "#166534" if p_val_final < 0.05 else "#92400E"
        sig_bg     = "#F0FDF4" if p_val_final < 0.05 else "#FFFBEB"
        rec_color  = "#4338CA" if best_pref == "Light Mode" else "#1E293B"
        rec_bg     = "#EEF2FF" if best_pref == "Light Mode" else "#F8FAFC"
        rec_border = "#C7D2FE" if best_pref == "Light Mode" else "#E2E8F0"

        # ── build grid cards ──────────────────────────────────────────
        def _grid_card(subtitle, body):
            return (
                '<div style="background:#F8FAFC;padding:14px 16px;border-radius:10px;">'
                  '<div style="font-size:11px;font-weight:600;color:#64748B;margin-bottom:4px;">'
                    + subtitle +
                  '</div>'
                  '<div style="font-size:13px;color:#334155;line-height:1.5;">'
                    + body +
                  '</div>'
                '</div>'
            )

        grid_html = (
            _grid_card("User Experience (UEQ)",
                       "Mode <strong>" + win_ueq + "</strong> lebih unggul pada UEQ, "
                       "skor tertinggi pada skala <strong>" + best_ueq_scale + "</strong>.") +
            _grid_card("Time on Task",
                       "Responden lebih cepat menggunakan mode <strong>" + win_tot + "</strong>.") +
            _grid_card("Error Rate",
                       "Tingkat kesalahan lebih rendah pada mode <strong>" + win_err + "</strong>.") +
            _grid_card("Preferensi Positif",
                       "Skor tertinggi pada aspek <strong>" + best_pref_aspect + "</strong> "
                       "(mode <strong>" + best_aspect_mode + "</strong>).") +
            _grid_card("Preferensi Negatif",
                       "Skor tertinggi pada aspek <strong>" + worst_aspect + "</strong> "
                       "(mode <strong>" + worst_aspect_mode + "</strong>).")
        )

        kesimpulan_html = (
            '<div style="font-size:11px;font-weight:700;color:#94A3B8;text-transform:uppercase;'
            'letter-spacing:0.08em;margin-bottom:12px;">Kesimpulan Akhir Penelitian</div>'

            '<div style="border:1px solid #E2E8F0;border-radius:16px;padding:28px;">'

              '<div style="background:' + rec_bg + ';border:1px solid ' + rec_border + ';border-radius:12px;'
              'padding:20px;text-align:center;margin-bottom:24px;">'
                '<div style="font-size:10px;font-weight:700;color:' + rec_color + ';text-transform:uppercase;'
                'letter-spacing:0.08em;margin-bottom:6px;">Rekomendasi Antarmuka</div>'
                '<div style="font-size:28px;font-weight:700;color:' + rec_color + ';">' + best_pref + '</div>'
              '</div>'

              '<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px;">'
                + grid_html +
              '</div>'

              '<div style="font-size:13px;color:#475569;line-height:1.7;text-align:center;'
              'padding-top:16px;border-top:1px solid #F1F5F9;">'
                'Secara keseluruhan, <strong>' + best_pref + '</strong> memberikan pengalaman pengguna '
                'yang lebih optimal berdasarkan evaluasi UEQ, Time on Task, Error Rate, '
                'dan preferensi responden pada enam aspek.'
              '</div>'

              '<div style="text-align:center;margin-top:16px;">'
                '<span style="display:inline-block;font-size:11px;font-weight:600;'
                'background:' + sig_bg + ';color:' + sig_color + ';'
                'padding:5px 16px;border-radius:20px;">'
                  + sig_label + ' (p = ' + str(round(grand_p, 4)) + ')'
                '</span>'
              '</div>'

            '</div>'
        )

        st.markdown(kesimpulan_html, unsafe_allow_html=True)
    

# ======================
# TIME ON TASK (UI BARU - SAMA SEPERTI UEQ)
# ======================

if menu == "Time on Task":

    st.markdown(f"""
    <div style="font-size:28px;font-weight:700;color:#1e293b;margin-bottom:10px;">
    Time on Task Analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div style="font-size:14px;color:#6b7280;">
    Analisis menggunakan Wilcoxon Signed Ranks Test (SPSS Style)
    </div>
    """, unsafe_allow_html=True)

    # ======================
    # DATASET MANAGER
    # ======================
    with st.expander("Dataset Manager", expanded=False):
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
    st.markdown("### Dataset Input")
    
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
    if st.button("Simpan Data Time on Task", type="primary", use_container_width=True):
        df_edit.to_csv(file_tot, index=False)
        st.success("Data tersimpan!")
        st.rerun()
    # ======================
    # ANALYSIS BUTTON (WILCOXON)
    # ======================
    # ... bagian kode sebelumnya ...
    data_kosong = df_edit[["Light_T1","Light_T2","Light_T3","Dark_T1","Dark_T2","Dark_T3"]].sum().sum() == 0

    if st.button("ANALISIS WILCOXON TEST", type="secondary"):
        if data_kosong:
            st.warning("Data masih kosong. Silakan isi data terlebih dahulu sebelum melakukan analisis.")
            st.stop()
        
        st.markdown("---")
        st.markdown("### Overall Metrics")
        
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
        st.markdown("### Task Results")
        st.dataframe(task_avgs.round(2), use_container_width=True)
        
        # ======================
        # WILCOXON SPSS OUTPUT - PER TASK
        # ======================
        st.markdown("### Wilcoxon Signed Ranks Test — Per Task")

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
        st.markdown("### Overall Wilcoxon Test (Mean per User)")
        overall_item = compute_wilcoxon_pair(
            light_err_per_user, dark_err_per_user,
            "Light (mean)", "Dark (mean)"
        )
        render_spss_wilcoxon([overall_item])
        
        # ======================
        # VISUALIZATION
        # ======================
        st.markdown("### Visual Comparison")
        
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
        st.markdown("### Benchmark Results")
        
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
                        padding: 25px; background: {bg_card}; color: {text_main}; border-radius: 16px; 
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
                            {'FASTEST' if avg_time == min([avg_light_err, avg_dark_err]) else 'Slower'}
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

        st.markdown("### Statistical Summary")

        st.markdown(f"""
        <div style="
            background: #f8fafc; padding: 24px; border-radius: 12px;
            border-left: 4px solid #6366f1;
        ">
            <div style="font-size: 16px; font-weight: 700; color:{text_main}; margin-bottom: 12px;">
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

    st.markdown(f"""
    <div style="font-size:28px;font-weight:700;color:#1e293b;margin-bottom:10px;">
    Error Rate Analysis
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
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
    st.markdown("### Dataset Input")
    
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
    if st.button("Simpan Data Error rate", type="primary", use_container_width=True):
        df_edit.to_csv(file_error, index=False)
        st.success("Data tersimpan!")
        st.rerun()


    # ======================
    # ANALYSIS BUTTON - FIXED WITH UNIQUE KEY
    # ======================
    if st.button("ANALISIS WILCOXON TEST", type="secondary", key="analyze_error_rate"):
        
        st.markdown("---")
        st.markdown("### Overall Metrics")
        
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
        st.markdown("### Task Results")
        st.dataframe(task_avgs.round(2), use_container_width=True)
        
        # ======================
        # WILCOXON SPSS OUTPUT - PER TASK
        # ======================
        st.markdown("### Wilcoxon Signed Ranks Test — Per Task")

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
        st.markdown("### Overall Wilcoxon Test (Mean per User)")

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
        st.markdown("### Visual Comparison")
        
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
        st.markdown("### Benchmark Results")
        
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
                    padding: 25px; background: {bg_card}; color: {text_main}; border-radius: 16px; 
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

        st.markdown("### Statistical Summary")

        st.markdown(f"""
        <div style="
            background: #f8fafc; padding: 24px; border-radius: 12px;
            border-left: 4px solid #6366f1;
        ">
            <div style="font-size: 16px; font-weight: 700; color:{text_main}; margin-bottom: 12px;">
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
        
# ==============================================================================
# MENU: UEQ ANALYSIS — STANDAR UEQ DATA ANALYSIS TOOL VERSION 13
# Logika identik dengan UEQ_Data_Analysis_Tool_Version13_Light_FB.xlsx
# Bahasa: Indonesia (sesuai pilihan bahasa di tool)
# ==============================================================================
 
if menu == "UEQ Analysis":

    st.markdown(f"""
    <div style="font-size:24px;font-weight:700;color:#1e293b;margin-bottom:4px;">
    Analisis UEQ (User Experience Questionnaire) — {app}
    </div>
    <div style="font-size:13px;color:#6b7280;margin-bottom:20px;">
    Logika identik UEQ Data Analysis Tool Version 13
    </div>
    """, unsafe_allow_html=True)

    # =========================================================================
    # KONSTANTA — BENAR SESUAI UEQ TOOLS V13
    # =========================================================================

    # Item negatif yang di-reverse: terverifikasi dari UEQ Tools Excel
    REVERSE_ITEMS = {3, 4, 5, 9, 10, 12, 17, 18, 19, 21, 23, 24, 25}

    SKALA_MAP = {
        "Daya Tarik":  [1, 12, 14, 16, 24, 25],
        "Kejelasan":   [2, 4, 13, 21],
        "Efisiensi":   [9, 20, 22, 23],
        "Ketepatan":   [8, 11, 17, 19],
        "Stimulasi":   [5, 6, 7, 18],
        "Kebaruan":    [3, 10, 15, 26],
    }

    LABEL_KIRI = [
        "menyusahkan","tak dapat dipahami","kreatif","mudah dipelajari",
        "bermanfaat","membosankan","tidak menarik","tak dapat diprediksi",
        "cepat","berdaya cipta","menghalangi","baik",
        "rumit","tidak disukai","lazim","tidak nyaman",
        "aman","memotivasi","memenuhi ekspektasi","tidak efisien",
        "jelas","tidak praktis","terorganisasi","atraktif",
        "ramah pengguna","konservatif",
    ]
    LABEL_KANAN = [
        "menyenangkan","dapat dipahami","monoton","sulit dipelajari",
        "kurang bermanfaat","mengasyikkan","menarik","dapat diprediksi",
        "lambat","konvensional","mendukung","buruk",
        "sederhana","menggembirakan","terdepan","nyaman",
        "tidak aman","tidak memotivasi","tidak memenuhi ekspektasi","efisien",
        "membingungkan","praktis","berantakan","tidak atraktif",
        "tidak ramah pengguna","inovatif",
    ]

    BENCHMARK = {
        "Daya Tarik": {"p25":0.69,"p50":1.18,"p75":1.58,"p90":1.84},
        "Kejelasan":  {"p25":0.72,"p50":1.20,"p75":1.73,"p90":2.00},
        "Efisiensi":  {"p25":0.60,"p50":1.05,"p75":1.50,"p90":1.88},
        "Ketepatan":  {"p25":0.78,"p50":1.14,"p75":1.48,"p90":1.70},
        "Stimulasi":  {"p25":0.50,"p50":1.00,"p75":1.35,"p90":1.70},
        "Kebaruan":   {"p25":0.16,"p50":0.70,"p75":1.12,"p90":1.60},
    }

    # =========================================================================
    # FUNGSI INTI — IDENTIK UEQ TOOLS V13
    # =========================================================================

    def ueq_transform(df):
        """
        Transformasi raw (1–7) → -3 s.d. +3.
        value - 4, lalu reverse items tertentu dikali -1.
        Identik sheet DT di UEQ Tools Excel.
        """
        dt = df.copy().apply(pd.to_numeric, errors="coerce") - 4
        for i in REVERSE_ITEMS:
            col = f"I{i}"
            if col in dt.columns:
                dt[col] = -dt[col]
        return dt

    def ueq_scale_stats(df_raw):
        """
        Hitung mean & varians per skala — identik Results & Confidence_Intervals.
        Mean skala = rata-rata dari item means (mean per kolom dalam skala).
        Varians skala = var dari item means tersebut (ddof=1).
        CI menggunakan std dev of per-person scale means.
        """
        from scipy.stats import t as t_dist
        dt = ueq_transform(df_raw)
        results = []
        for sk, items in SKALA_MAP.items():
            cols = [f"I{i}" for i in items]
            item_means   = dt[cols].mean(axis=0)          # mean per item (column)
            scale_mean   = item_means.mean()               # mean of item means
            scale_var    = item_means.var(ddof=1)          # var of item means
            per_resp     = dt[cols].mean(axis=1).dropna()  # per-person scale mean
            n_r          = len(per_resp)
            std_resp     = per_resp.std(ddof=1)
            t_crit       = t_dist.ppf(0.975, df=n_r-1) if n_r > 1 else 1.96
            ci           = t_crit * std_resp / np.sqrt(n_r) if n_r > 0 else np.nan
            results.append({
                "Skala":          sk,
                "N":              n_r,
                "Mean":           round(float(scale_mean), 4),
                "Varians":        round(float(scale_var),  4),
                "Std. Dev.":      round(float(item_means.std(ddof=1)), 4),
                "Confidence (±)": round(float(ci), 4),
                "CI Bawah":       round(float(scale_mean - ci), 4),
                "CI Atas":        round(float(scale_mean + ci), 4),
            })
        return pd.DataFrame(results)

    def ueq_item_stats(df_raw):
        """Mean & varians per item — identik Results sheet."""
        from scipy.stats import t as t_dist
        dt = ueq_transform(df_raw)
        rows = []
        for i in range(1, 27):
            col  = f"I{i}"
            vals = dt[col].dropna() if col in dt.columns else pd.Series(dtype=float)
            n    = len(vals)
            mean = float(vals.mean()) if n > 0 else np.nan
            var  = float(vals.var(ddof=1)) if n > 1 else 0.0
            std  = var ** 0.5
            t_cr = t_dist.ppf(0.975, df=n-1) if n > 1 else 1.96
            ci   = t_cr * std / np.sqrt(n) if n > 0 else np.nan
            rows.append({
                "Item":           i,
                "Kiri":           LABEL_KIRI[i-1],
                "Kanan":          LABEL_KANAN[i-1],
                "Skala":          next((s for s, it in SKALA_MAP.items() if i in it), "-"),
                "Mean":           round(mean, 2),
                "Varians":        round(var,  2),
                "Std. Dev.":      round(std,  2),
                "N":              n,
                "Confidence (±)": round(ci,   3),
                "CI Bawah":       round(mean - ci, 3),
                "CI Atas":        round(mean + ci, 3),
            })
        return pd.DataFrame(rows)

    def benchmark_kategori(mean, skala):
        b = BENCHMARK[skala]
        if mean >= b["p90"]: return "Excellent"
        elif mean >= b["p75"]: return "Good"
        elif mean >= b["p50"]: return "Above Average"
        elif mean >= b["p25"]: return "Below Average"
        else: return "Bad"

    def benchmark_interpretasi(k):
        return {
            "Excellent":     "10% hasil lebih baik, 90% lebih buruk",
            "Good":          "25% hasil lebih baik, 75% lebih buruk",
            "Above Average": "25% hasil lebih baik, 50% lebih buruk",
            "Below Average": "50% hasil lebih baik, 25% lebih buruk",
            "Bad":           "75% atau lebih hasil lebih baik",
        }.get(k, "")

    def interpret_category(score):
        if score > 1.5:  return "Excellent"
        elif score > 0.8: return "Good"
        elif score > 0.0: return "Above Average"
        elif score > -0.8: return "Below Average"
        else: return "Bad"

    def inconsistency_check(df_raw):
        dt = ueq_transform(df_raw)
        raw_num = df_raw.apply(pd.to_numeric, errors="coerce")
        hasil = []
        for idx, row in dt.iterrows():
            crit = sum(
                1 for sk, items in SKALA_MAP.items()
                if len(vals := row[[f"I{i}" for i in items]].dropna()) >= 2
                and (vals.max() - vals.min()) > 3
            )
            raw_row = raw_num.iloc[idx] if idx < len(raw_num) else pd.Series()
            same = int(raw_row.value_counts().max()) if len(raw_row) > 0 else 0
            hasil.append({
                "Responden":     f"R{idx+1}",
                "Skala Kritis":  crit,
                "Perlu Dihapus?": "Ya ⚠️" if crit >= 2 else "Tidak",
                "Jawaban Identik": same,
                "Critical Length": "Ya ⚠️" if same > 15 else "Tidak",
            })
        return pd.DataFrame(hasil)

    # =========================================================================
    # LOAD DATA
    # =========================================================================
    items_label = [f"I{i}" for i in range(1, 27)]

    if os.path.exists(file_ueq_light):
        u_light = pd.read_csv(file_ueq_light)
        for col in items_label:
            if col not in u_light.columns:
                u_light[col] = 4
        u_light = u_light[items_label].iloc[:n].reset_index(drop=True)
    else:
        u_light = pd.DataFrame(4, index=range(n), columns=items_label)

    if os.path.exists(file_ueq_dark):
        u_dark = pd.read_csv(file_ueq_dark)
        for col in items_label:
            if col not in u_dark.columns:
                u_dark[col] = 4
        u_dark = u_dark[items_label].iloc[:n].reset_index(drop=True)
    else:
        u_dark = pd.DataFrame(4, index=range(n), columns=items_label)

    u_light_disp = u_light.copy()
    u_light_disp.insert(0, "Responden", [f"R{i+1}" for i in range(len(u_light_disp))])
    u_dark_disp = u_dark.copy()
    u_dark_disp.insert(0, "Responden", [f"R{i+1}" for i in range(len(u_dark_disp))])

    # =========================================================================
    # TAB NAVIGASI
    # =========================================================================
    tab_input, tab_dt, tab_hasil, tab_ci, tab_dist, tab_bench, tab_inkonsisten = st.tabs([
        "Data Mentah", "Data Transformation (DT)", "Hasil Skala",
        "Confidence Interval", "Distribusi Jawaban", "Benchmark", "Deteksi Inkonsistensi",
    ])

    # ------------------------------------------------------------------
    # TAB 1: DATA MENTAH
    # ------------------------------------------------------------------
    with tab_input:
        st.markdown("### Input Data Skor Kuesioner (Skala 1–7)")
        st.caption("1 = alternatif paling kiri, 7 = alternatif paling kanan.")

        col_l, col_d = st.columns(2)
        with col_l:
            st.markdown(f"**Light Mode** (n={n})")
            edit_l = st.data_editor(
                u_light_disp, key="ueq_raw_light", use_container_width=True,
                column_config={
                    "Responden": st.column_config.TextColumn(disabled=True),
                    **{f"I{i}": st.column_config.NumberColumn(f"I{i}", min_value=1, max_value=7, step=1)
                       for i in range(1, 27)}
                }
            )
        with col_d:
            st.markdown(f"**Dark Mode** (n={n})")
            edit_d = st.data_editor(
                u_dark_disp, key="ueq_raw_dark", use_container_width=True,
                column_config={
                    "Responden": st.column_config.TextColumn(disabled=True),
                    **{f"I{i}": st.column_config.NumberColumn(f"I{i}", min_value=1, max_value=7, step=1)
                       for i in range(1, 27)}
                }
            )

        if st.button("Simpan Data Kuesioner", type="primary", use_container_width=True):
            edit_l[items_label].to_csv(file_ueq_light, index=False)
            edit_d[items_label].to_csv(file_ueq_dark, index=False)
            st.success("Data berhasil disimpan!")
            st.rerun()

        with st.expander("Dataset Manager — Light Mode"):
            dataset_manager(u_light, items_label, file_ueq_light, "UEQ Light Mode", f"ueq_light_{app}")
        with st.expander("Dataset Manager — Dark Mode"):
            dataset_manager(u_dark, items_label, file_ueq_dark, "UEQ Dark Mode", f"ueq_dark_{app}")

    # Hitung dari data yang sedang diedit
    df_light_clean = edit_l[items_label].apply(pd.to_numeric, errors="coerce")
    df_dark_clean  = edit_d[items_label].apply(pd.to_numeric, errors="coerce")
    dt_light       = ueq_transform(df_light_clean)
    dt_dark        = ueq_transform(df_dark_clean)
    stats_light    = ueq_scale_stats(df_light_clean)
    stats_dark     = ueq_scale_stats(df_dark_clean)
    item_light     = ueq_item_stats(df_light_clean)
    item_dark      = ueq_item_stats(df_dark_clean)

    # ------------------------------------------------------------------
    # TAB 2: DATA TRANSFORMATION (DT)
    # ------------------------------------------------------------------
    with tab_dt:
        st.markdown("### Data Transformation — Identik Sheet DT")
        st.caption("Nilai dikonversi ke rentang -3 s.d. +3 (Nilai - 4). Item negatif di-reverse (×-1).")
        st.caption(f"**Item yang di-reverse:** {sorted(REVERSE_ITEMS)}")

        col_dt1, col_dt2 = st.columns(2)
        with col_dt1:
            st.markdown("**Light Mode**")
            dt_l_disp = dt_light.copy().round(2)
            dt_l_disp.insert(0, "Responden", [f"R{i+1}" for i in range(len(dt_l_disp))])
            st.dataframe(dt_l_disp, use_container_width=True)
        with col_dt2:
            st.markdown("**Dark Mode**")
            dt_d_disp = dt_dark.copy().round(2)
            dt_d_disp.insert(0, "Responden", [f"R{i+1}" for i in range(len(dt_d_disp))])
            st.dataframe(dt_d_disp, use_container_width=True)

    # ------------------------------------------------------------------
    # TAB 3: HASIL SKALA
    # ------------------------------------------------------------------
    with tab_hasil:
        st.markdown("### Hasil Analisis Skala UEQ — Identik Sheet Results")

        tabel_gabung = pd.DataFrame({
            "Skala":      stats_light["Skala"],
            "Mean Light": stats_light["Mean"],
            "Var. Light": stats_light["Varians"],
            "Mean Dark":  stats_dark["Mean"],
            "Var. Dark":  stats_dark["Varians"],
        })
        tabel_gabung["Unggul"] = tabel_gabung.apply(
            lambda r: "Light Mode" if r["Mean Light"] > r["Mean Dark"] else
                      ("Dark Mode" if r["Mean Dark"] > r["Mean Light"] else "Seimbang"), axis=1
        )
        st.table(tabel_gabung)

        # Pragmatic & Hedonic Quality — identik Results sheet
        pq_light = stats_light[stats_light["Skala"].isin(["Kejelasan","Efisiensi","Ketepatan"])]["Mean"].mean()
        pq_dark  = stats_dark [stats_dark ["Skala"].isin(["Kejelasan","Efisiensi","Ketepatan"])]["Mean"].mean()
        hq_light = stats_light[stats_light["Skala"].isin(["Stimulasi","Kebaruan"])]["Mean"].mean()
        hq_dark  = stats_dark [stats_dark ["Skala"].isin(["Stimulasi","Kebaruan"])]["Mean"].mean()
        at_light = float(stats_light[stats_light["Skala"]=="Daya Tarik"]["Mean"].values[0])
        at_dark  = float(stats_dark [stats_dark ["Skala"]=="Daya Tarik"]["Mean"].values[0])

        st.markdown("#### Pragmatic Quality & Hedonic Quality")
        c1, c2, c3 = st.columns(3)
        for col, title, lv, dv, sub in [
            (c1, "Daya Tarik",         at_light,  at_dark,  "Attractiveness"),
            (c2, "Kualitas Pragmatis",  pq_light,  pq_dark,  "Kejelasan · Efisiensi · Ketepatan"),
            (c3, "Kualitas Hedonis",    hq_light,  hq_dark,  "Stimulasi · Kebaruan"),
        ]:
            col.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-title">{title}</div>
                <div style="font-size:20px;font-weight:700;">
                    <span class="val-light">{lv:.2f}</span>
                    <span class="vs-divider"> | </span>
                    <span class="val-dark">{dv:.2f}</span>
                </div>
                <div style="font-size:11px;color:#6b7280;margin-top:4px;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Grafik Perbandingan Mean Skala (-3 s.d. +3)")
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=stats_light["Skala"], y=stats_light["Mean"], name="Light Mode",
            marker_color="#6366f1", text=stats_light["Mean"].round(2), textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            x=stats_dark["Skala"], y=stats_dark["Mean"], name="Dark Mode",
            marker_color="#1e293b", text=stats_dark["Mean"].round(2), textposition="outside",
        ))
        fig_bar.add_hline(y=0.0, line_color="black", line_width=1)
        fig_bar.add_hline(y=0.8,  line_dash="dot", line_color="#10b981", line_width=1, annotation_text="Batas Positif (0.8)")
        fig_bar.add_hline(y=-0.8, line_dash="dot", line_color="#ef4444", line_width=1, annotation_text="Batas Negatif (-0.8)")
        fig_bar.update_layout(
            yaxis=dict(range=[-3, 3], title="Mean Score"), barmode="group", height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("#### Interpretasi Kualitas Skala")
        col_il, col_id = st.columns(2)
        with col_il:
            st.markdown("**Light Mode**")
            cat_l = stats_light.copy()
            cat_l["Kategori"] = cat_l["Mean"].apply(interpret_category)
            st.dataframe(cat_l[["Skala","Mean","Kategori"]], use_container_width=True, hide_index=True)
        with col_id:
            st.markdown("**Dark Mode**")
            cat_d = stats_dark.copy()
            cat_d["Kategori"] = cat_d["Mean"].apply(interpret_category)
            st.dataframe(cat_d[["Skala","Mean","Kategori"]], use_container_width=True, hide_index=True)

        st.markdown("#### Analisis Per Item")
        col_il2, col_id2 = st.columns(2)
        with col_il2:
            st.markdown("**Light Mode**")
            st.dataframe(item_light[["Item","Kiri","Kanan","Skala","Mean","Varians","Std. Dev.","N"]],
                         use_container_width=True, hide_index=True)
        with col_id2:
            st.markdown("**Dark Mode**")
            st.dataframe(item_dark[["Item","Kiri","Kanan","Skala","Mean","Varians","Std. Dev.","N"]],
                         use_container_width=True, hide_index=True)

        avg_l = stats_light["Mean"].mean()
        avg_d = stats_dark["Mean"].mean()
        unggul = "Light Mode" if avg_l > avg_d else "Dark Mode"
        st.success(
            f"**Kesimpulan:** {unggul} lebih unggul pada aplikasi {app} "
            f"(Light: {avg_l:.3f} | Dark: {avg_d:.3f})."
        )

    # ------------------------------------------------------------------
    # TAB 4: CONFIDENCE INTERVAL
    # ------------------------------------------------------------------
    with tab_ci:
        st.markdown("### Confidence Interval (95%) per Skala — Identik Sheet Confidence_Intervals")
        st.caption("Semakin kecil CI, semakin tinggi presisi estimasi mean skala.")

        col_ci1, col_ci2 = st.columns(2)
        with col_ci1:
            st.markdown("**Light Mode**")
            st.dataframe(stats_light[["Skala","Mean","Std. Dev.","N","Confidence (±)","CI Bawah","CI Atas"]],
                         use_container_width=True, hide_index=True)
        with col_ci2:
            st.markdown("**Dark Mode**")
            st.dataframe(stats_dark[["Skala","Mean","Std. Dev.","N","Confidence (±)","CI Bawah","CI Atas"]],
                         use_container_width=True, hide_index=True)

        st.markdown("#### CI Per Item")
        col_ci3, col_ci4 = st.columns(2)
        with col_ci3:
            st.markdown("**Light Mode**")
            st.dataframe(item_light[["Item","Kiri","Kanan","Mean","Std. Dev.","N","Confidence (±)","CI Bawah","CI Atas"]],
                         use_container_width=True, hide_index=True)
        with col_ci4:
            st.markdown("**Dark Mode**")
            st.dataframe(item_dark[["Item","Kiri","Kanan","Mean","Std. Dev.","N","Confidence (±)","CI Bawah","CI Atas"]],
                         use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 5: DISTRIBUSI JAWABAN
    # ------------------------------------------------------------------
    with tab_dist:
        st.markdown("### Distribusi Jawaban per Item — Identik Sheet Answer_Distributions")
        mode_dist = st.radio("Pilih Mode", ["Light Mode","Dark Mode"], horizontal=True, key="dist_mode")
        df_dist = df_light_clean if mode_dist == "Light Mode" else df_dark_clean

        dist_rows = []
        for i in range(1, 27):
            col = f"I{i}"
            vals = df_dist[col].dropna() if col in df_dist.columns else pd.Series(dtype=float)
            counts = {v: 0 for v in range(1, 8)}
            for v in vals:
                if int(v) in counts:
                    counts[int(v)] += 1
            dist_rows.append({
                "Item": i,
                "Label": f"{LABEL_KIRI[i-1]} / {LABEL_KANAN[i-1]}",
                "Skala": next((s for s, it in SKALA_MAP.items() if i in it), "-"),
                **{str(k): counts[k] for k in range(1, 8)},
            })
        st.dataframe(pd.DataFrame(dist_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 6: BENCHMARK
    # ------------------------------------------------------------------
    with tab_bench:
        st.markdown("### Benchmark UEQ — Identik Sheet Benchmark")
        st.caption("Benchmark: 468 studi, 21.175 responden.")

        bench_rows = []
        for _, row_l in stats_light.iterrows():
            sk    = row_l["Skala"]
            row_d = stats_dark[stats_dark["Skala"]==sk].iloc[0]
            b     = BENCHMARK[sk]
            bench_rows.append({
                "Skala":            sk,
                "Mean Light":       row_l["Mean"],
                "Kategori Light":   benchmark_kategori(row_l["Mean"], sk),
                "Mean Dark":        row_d["Mean"],
                "Kategori Dark":    benchmark_kategori(row_d["Mean"], sk),
                "Bad (<p25)":       f"< {b['p25']}",
                "Below Avg":        f"{b['p25']} – {b['p50']}",
                "Above Avg":        f"{b['p50']} – {b['p75']}",
                "Good":             f"{b['p75']} – {b['p90']}",
                "Excellent (≥p90)": f"≥ {b['p90']}",
            })
        st.dataframe(pd.DataFrame(bench_rows), use_container_width=True, hide_index=True)

        mode_bench = st.radio("Grafik Benchmark —", ["Light Mode","Dark Mode"], horizontal=True, key="bench_mode")
        stats_bench = stats_light if mode_bench == "Light Mode" else stats_dark

        COLOR_BENCH = {
            "Excellent":"#27500A","Good":"#185FA5",
            "Above Average":"#534AB7","Below Average":"#854F0B","Bad":"#A32D2D",
        }
        fig_bench = go.Figure()
        for _, row in stats_bench.iterrows():
            kat = benchmark_kategori(row["Mean"], row["Skala"])
            fig_bench.add_trace(go.Bar(
                x=[row["Skala"]], y=[row["Mean"]],
                name=f"{row['Skala']} ({kat})", marker_color=COLOR_BENCH.get(kat, "#888"),
                text=f"{row['Mean']:.2f}<br>{kat}", textposition="outside",
            ))
        for lbl, key in [("p25","p25"),("p50","p50"),("p75","p75"),("p90","p90")]:
            y_vals = [BENCHMARK[s][key] for s in stats_bench["Skala"].tolist()]
            fig_bench.add_trace(go.Scatter(
                x=stats_bench["Skala"].tolist(), y=y_vals, mode="lines",
                name=lbl, line=dict(dash="dot", width=1), showlegend=True,
            ))
        fig_bench.update_layout(
            yaxis=dict(range=[-1, 3], title="Mean Score"),
            barmode="overlay", height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_bench, use_container_width=True)

        interp_rows = []
        for _, row in stats_bench.iterrows():
            kat = benchmark_kategori(row["Mean"], row["Skala"])
            interp_rows.append({
                "Skala": row["Skala"], "Mean": row["Mean"],
                "Benchmark": kat, "Interpretasi": benchmark_interpretasi(kat),
            })
        st.dataframe(pd.DataFrame(interp_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------------
    # TAB 7: DETEKSI INKONSISTENSI
    # ------------------------------------------------------------------
    with tab_inkonsisten:
        st.markdown("### Deteksi Jawaban Tidak Konsisten — Identik Sheet Inconsistencies")
        mode_ink = st.radio("Pilih Mode", ["Light Mode","Dark Mode"], horizontal=True, key="inkons_mode")
        df_ink = df_light_clean if mode_ink == "Light Mode" else df_dark_clean
        df_check = inconsistency_check(df_ink)

        def highlight_inkons(row):
            if row["Perlu Dihapus?"].startswith("Ya") or row["Critical Length"].startswith("Ya"):
                return ["background-color:#FEF3C7"] * len(row)
            return [""] * len(row)

        st.dataframe(df_check.style.apply(highlight_inkons, axis=1),
                     use_container_width=True, hide_index=True)

        n_hapus   = (df_check["Perlu Dihapus?"].str.startswith("Ya")).sum()
        n_crit    = (df_check["Critical Length"].str.startswith("Ya")).sum()
        if n_hapus > 0 or n_crit > 0:
            st.warning(f"Ditemukan {n_hapus} responden dengan ≥2 skala kritis dan {n_crit} dengan Critical Length.")
        else:
            st.success("Tidak ditemukan jawaban yang mencurigakan.")

    st.markdown("---")
    st.caption("UEQ Analysis · UEQ Data Analysis Tool Version 13 · Benchmark: 468 studi, 21.175 responden")
            

# ==============================================================================
# MENU: PREFERENSI RESPONDEN (TANPA IKON - DENGAN DATASET MANAGER)
# ==============================================================================

if menu == "Preferensi Responden":

    st.markdown(f"""
    <div style="font-size:28px;font-weight:700;color:#1e293b;margin-bottom:10px;">
    Preferensi Responden - {app}
    </div>
    """, unsafe_allow_html=True)

    st.info("""
    Metode Analisis: 
    Menggunakan Mean Preference Analysis (Skala Likert 1–7). 
    Sistem secara otomatis melakukan Reverse Scoring (8 - Nilai Asli) pada pernyataan negatif 
    sebelum digabungkan menjadi Grand Mean.
    """)
    
    # Definisi instrumen kolom sesuai metodologi penelitian[cite: 1]
    columns_pref = [
        "Responden",
        "R1","R2","R3","R4",      # Keterbacaan (Readability)
        "ES1","ES2","ES3","ES4",  # Kelelahan Mata (Eye Strain)
        "U1","U2","U3","U4",      # Usability
        "B1","B2","B3","B4",      # Konsumsi Baterai (Battery)
        "E1","E2","E3","E4",      # Efisien Kinerja (Efficiency)
        "ED1","ED2","ED3","ED4"   # Estetika & Daya Tarik (Aesthetic)
    ]

    # File data per aplikasi
    file_pos = os.path.join(BASE_DIR, f"preferensi_positif_{app}.csv")
    file_neg = os.path.join(BASE_DIR, f"preferensi_negatif_{app}.csv")

    # --- TAB SISTEM ---
    tab_input, tab_analisis = st.tabs(["Input Data Kuesioner", "Hasil Analisis dan Narasi"])

    with tab_input:
        # DATASET MANAGER UNTUK PREFERENSI
        with st.expander("Dataset Manager - Preferensi Positif", expanded=False):
            if os.path.exists(file_pos):
                df_manager_pos = pd.read_csv(file_pos)
            else:
                df_manager_pos = pd.DataFrame(columns=columns_pref)
            
            dataset_manager(
                df_manager_pos,
                columns_pref,
                file_pos,
                "Dataset Preferensi Positif",
                f"preferensi_positif_{app}"
            )

        with st.expander("Dataset Manager - Preferensi Negatif", expanded=False):
            if os.path.exists(file_neg):
                df_manager_neg = pd.read_csv(file_neg)
            else:
                df_manager_neg = pd.DataFrame(columns=columns_pref)
            
            dataset_manager(
                df_manager_neg,
                columns_pref,
                file_neg,
                "Dataset Preferensi Negatif",
                f"preferensi_negatif_{app}"
            )

        st.markdown("---")
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("### 1. Pernyataan Positif")
            st.caption("Skala 1 (Light) ke 7 (Dark)")
            if os.path.exists(file_pos):
                df_pos = pd.read_csv(file_pos)
            else:
                df_pos = pd.DataFrame(columns=columns_pref)
            
            df_pos = adjust_dataframe(df_pos, n)
            df_pos_edit = st.data_editor(df_pos, key="pos_editor_final", use_container_width=True)
            
            if st.button("Simpan Data Positif", use_container_width=True):
                df_pos_edit.to_csv(file_pos, index=False)
                st.success("Data Positif Berhasil Disimpan")

        with col_neg:
            st.markdown("### 2. Pernyataan Negatif")
            st.caption("Skala 1 (Light) ke 7 (Dark)")
            if os.path.exists(file_neg):
                df_neg = pd.read_csv(file_neg)
            else:
                df_neg = pd.DataFrame(columns=columns_pref)
            
            df_neg = adjust_dataframe(df_neg, n)
            df_neg_edit = st.data_editor(df_neg, key="neg_editor_final", use_container_width=True)
            
            if st.button("Simpan Data Negatif", use_container_width=True):
                df_neg_edit.to_csv(file_neg, index=False)
                st.success("Data Negatif Berhasil Disimpan")

    # --- LOGIKA ANALISIS ---
    with tab_analisis:
        if st.button("Refresh Analisis", use_container_width=True):
            st.rerun()
        
        aspek_map = {
            "Keterbacaan (Readability)": ["R1","R2","R3","R4"],
            "Kelelahan Mata (Eye Strain)": ["ES1","ES2","ES3","ES4"],
            "Usability": ["U1","U2","U3","U4"],
            "Konsumsi Baterai": ["B1","B2","B3","B4"],
            "Efisien Kinerja": ["E1","E2","E3","E4"],
            "Estetika & Daya Tarik": ["ED1","ED2","ED3","ED4"]
        }

        final_data = []
        
        # Hitung rata-rata per aspek[cite: 1]
        for name, cols in aspek_map.items():
            m_pos = df_pos_edit[cols].apply(pd.to_numeric, errors='coerce').mean().mean()
            m_neg_raw = df_neg_edit[cols].apply(pd.to_numeric, errors='coerce').mean().mean()
            
            # Reverse Scoring (8 - n) untuk data negatif sesuai Bab III[cite: 1]
            m_neg_rev = 8 - m_neg_raw
            
            # Grand Mean gabungan
            grand_mean = (m_pos + m_neg_rev) / 2
            
            # Penentuan kecenderungan (Ambang batas netral 4.0)[cite: 1]
            if pd.isna(grand_mean):
                kecenderungan = "Data Kosong"
                color_code = "#94a3b8"
            elif grand_mean < 4:
                kecenderungan = "Light Mode"
                color_code = "#6366f1"
            elif grand_mean > 4:
                kecenderungan = "Dark Mode"
                color_code = "#1e293b"
            else:
                kecenderungan = "Netral"
                color_code = "#10b981"

            final_data.append({
                "Aspek Pengalaman": name,
                "Mean Positif": round(m_pos, 3),
                "Mean Negatif (Raw)": round(m_neg_raw, 3),
                "Grand Mean": round(grand_mean, 3),
                "Preferensi": kecenderungan,
                "Color": color_code
            })

        res_df = pd.DataFrame(final_data)

        if not res_df.empty and res_df["Grand Mean"].notna().any():
            
            # 1. Tabel Rekapitulasi
            st.markdown("### Tabel Rekapitulasi Preferensi")
            st.table(res_df[["Aspek Pengalaman", "Mean Positif", "Mean Negatif (Raw)", "Grand Mean", "Preferensi"]])

            # 2. Grafik Batang
            st.markdown("### Grafik Kecenderungan Per Aspek")
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=res_df["Aspek Pengalaman"],
                y=res_df["Grand Mean"],
                marker_color=res_df["Color"],
                text=res_df["Grand Mean"],
                textposition='auto',
            ))
            fig.add_hline(y=4, line_dash="dash", line_color="red", annotation_text="Titik Netral (4.0)")
            fig.update_layout(yaxis=dict(range=[1, 7], title="Skor Preferensi"), height=450)
            st.plotly_chart(fig, use_container_width=True)

            # 3. Narasi Hasil Detail
            st.markdown("### Analisis Detail Hasil")
            l_aspek = res_df[res_df["Preferensi"] == "Light Mode"]["Aspek Pengalaman"].tolist()
            d_aspek = res_df[res_df["Preferensi"] == "Dark Mode"]["Aspek Pengalaman"].tolist()
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"""
                <div style="background-color:rgba(99, 102, 241, 0.1); padding:15px; border-radius:10px; border-left:5px solid #6366f1;">
                    <b style="color:#6366f1;">Unggul Light Mode</b><br>
                    <p style="font-size:13px; margin-top:8px;">{", ".join(l_aspek) if l_aspek else "Tidak ada"}</p>
                </div>
                """, unsafe_allow_html=True)
            with c2:
                st.markdown(f"""
                <div style="background-color:rgba(30, 41, 59, 0.1); padding:15px; border-radius:10px; border-left:5px solid #1e293b;">
                    <b style="color:#1e293b;">Unggul Dark Mode</b><br>
                    <p style="font-size:13px; margin-top:8px;">{", ".join(d_aspek) if d_aspek else "Tidak ada"}</p>
                </div>
                """, unsafe_allow_html=True)

            # Highlight Aspek Terkuat
            aspek_max = res_df.loc[res_df["Grand Mean"].idxmax()]
            aspek_min = res_df.loc[res_df["Grand Mean"].idxmin()]
            
            st.success(f"""
            Kesimpulan Akhir Preferensi:
            - Preferensi Dark Mode terkuat ada pada aspek {aspek_max['Aspek Pengalaman']} (Skor: {aspek_max['Grand Mean']}).
            - Preferensi Light Mode terkuat ada pada aspek {aspek_min['Aspek Pengalaman']} (Skor: {aspek_min['Grand Mean']}).
            - Secara keseluruhan, aplikasi {app} lebih cenderung optimal menggunakan {'Light Mode' if len(l_aspek) > len(d_aspek) else 'Dark Mode'} berdasarkan dominasi jumlah aspek.
            """)
        else:
            st.warning("Data kuesioner belum diisi. Silakan masukkan data pada tab Input Data Kuesioner terlebih dahulu.")

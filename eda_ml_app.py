"""
ML-Ready EDA Explorer — ICU Mortality Predictor
================================================
Tailored for the WIDS Datathon / Kaggle ICU dataset (training_v2.csv)

Requirements:
    pip install streamlit pandas numpy matplotlib seaborn scipy
                scikit-learn pyarrow openpyxl xlrd

Run:
    streamlit run eda_ml_app.py
"""

import io
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU EDA Explorer",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colors ─────────────────────────────────────────────────────────────────────
ACCENT  = "#0EA5E9"
ACCENT2 = "#F43F5E"
SUCCESS = "#22C55E"
WARN    = "#F59E0B"
DANGER  = "#EF4444"
BG_CARD = "#F8FAFC"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{ font-family: 'DM Sans', sans-serif; }}
.main-title {{
    font-size:2.5rem; font-weight:700; letter-spacing:-1px;
    background: linear-gradient(135deg,{ACCENT} 0%,{ACCENT2} 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.subtitle {{ color:#64748B; font-size:1rem; margin-bottom:1.5rem; }}
.section-title {{
    font-size:1.3rem; font-weight:700; color:#1E293B;
    border-left:4px solid {ACCENT}; padding-left:.7rem;
    margin-top:2rem; margin-bottom:1rem;
}}
.tag {{ display:inline-block; border-radius:99px; padding:2px 10px;
        font-size:.78rem; font-weight:600; margin:2px; }}
.tag-num  {{ background:#E0F2FE; color:#0369A1; }}
.tag-cat  {{ background:#FCE7F3; color:#BE185D; }}
.tag-bin  {{ background:#D1FAE5; color:#065F46; }}
.tag-id   {{ background:#FEF3C7; color:#92400E; }}
.recommend-box {{
    background:linear-gradient(135deg,#E0F2FE 0%,#FCE7F3 100%);
    border:1.5px solid {ACCENT}; border-radius:12px; padding:1.2rem 1.5rem; margin-top:1rem;
}}
</style>
""", unsafe_allow_html=True)

# ── Known column groups for this dataset ──────────────────────────────────────
ID_COLS        = ["encounter_id", "patient_id", "hospital_id", "icu_id"]
TARGET_COL     = "hospital_death"
DEMO_COLS      = ["age", "gender", "ethnicity", "bmi", "height", "weight"]
BINARY_FLAGS   = ["elective_surgery","apache_post_operative","arf_apache",
                  "gcs_unable_apache","intubated_apache","ventilated_apache",
                  "aids","cirrhosis","diabetes_mellitus","hepatic_failure",
                  "immunosuppression","leukemia","lymphoma",
                  "solid_tumor_with_metastasis","readmission_status"]
CAT_COLS_KNOWN = ["ethnicity","gender","hospital_admit_source","icu_admit_source",
                  "icu_stay_type","icu_type","apache_3j_bodysystem","apache_2_bodysystem"]

# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<p class="main-title">🏥 ICU Mortality — EDA Explorer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Exploratory Data Analysis for ICU mortality prediction · WIDS Datathon dataset</p>',
            unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# FILE UPLOAD
# ═══════════════════════════════════════════════════════════════════════════════
uploaded = st.file_uploader(
    "Upload dataset  (CSV · Excel · JSON · Parquet)",
    type=["csv","xlsx","xls","json","parquet"],
)

@st.cache_data(show_spinner=False)
def load_file(f):
    name = f.name.lower()
    if name.endswith(".csv"):            return pd.read_csv(f)
    elif name.endswith((".xlsx","xls")): return pd.read_excel(f)
    elif name.endswith(".json"):         return pd.read_json(f)
    elif name.endswith(".parquet"):      return pd.read_parquet(f)

if uploaded is None:
    st.info("👆 Upload `training_v2.csv` (or any compatible dataset) to begin.")
    st.stop()

with st.spinner("Loading data…"):
    df_raw = load_file(uploaded)

# ── Derive column groups ───────────────────────────────────────────────────────
all_cols    = df_raw.columns.tolist()
num_cols    = [c for c in df_raw.select_dtypes(include=np.number).columns
               if c not in ID_COLS and c != TARGET_COL and c not in BINARY_FLAGS]
cat_cols    = [c for c in CAT_COLS_KNOWN if c in all_cols]
binary_cols = [c for c in BINARY_FLAGS if c in all_cols]
apache_cols = [c for c in all_cols if "apache" in c.lower()
               and c not in BINARY_FLAGS and c not in cat_cols and c != TARGET_COL
               and c in num_cols]
vital_d1    = [c for c in all_cols if c.startswith("d1_") and c in num_cols]
vital_h1    = [c for c in all_cols if c.startswith("h1_") and c in num_cols]
lab_cols    = [c for c in num_cols if c not in vital_d1 and c not in vital_h1
               and c not in apache_cols and c not in DEMO_COLS
               and c not in ID_COLS and c in all_cols]

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    sample_pct = st.slider("Sample % for heavy plots", 10, 100, 100, 5)
    palette    = st.selectbox("Plot palette", ["coolwarm","viridis","Blues","magma","rocket"])

    st.markdown("---")
    st.subheader("🎯 Target")
    target_options = ["— none —"] + all_cols
    default_idx    = target_options.index(TARGET_COL) if TARGET_COL in target_options else 0
    target_col     = st.selectbox("Target column", target_options, index=default_idx)
    if target_col == "— none —":
        target_col = None

    st.markdown("---")
    st.subheader("📑 Sections")
    SECTIONS = [
        "1 · Overview",
        "2 · Data Quality & Missing Values",
        "3 · Demographics & Patient Profile",
        "4 · Vital Signs Analysis",
        "5 · Lab Values Analysis",
        "6 · APACHE Scores",
        "7 · Correlation & Multicollinearity",
        "8 · Outlier Detection",
        "9 · Class Balance (Mortality Rate)",
        "10 · Feature Importance vs Mortality",
        "11 · Normality & Distributions",
        "12 · ML Readiness Report",
    ]
    active = st.multiselect("Show sections", SECTIONS, default=SECTIONS)

df = df_raw.sample(frac=sample_pct/100, random_state=42) if sample_pct < 100 else df_raw.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "1 · Overview" in active:
    st.markdown('<p class="section-title">📋 1 · Dataset Overview</p>', unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Patients",        f"{df_raw.shape[0]:,}")
    c2.metric("Features",        df_raw.shape[1])
    c3.metric("Vital sign cols", len(vital_d1)+len(vital_h1))
    c4.metric("Lab value cols",  len(lab_cols))
    c5.metric("Memory (MB)",     f"{df_raw.memory_usage(deep=True).sum()/1e6:.1f}")

    with st.expander("Column catalogue by group"):
        def tags(cols, css):
            return " ".join(f'<span class="tag {css}">{c}</span>' for c in cols if c in all_cols)
        st.markdown("**🪪 ID columns** (exclude from model)")
        st.markdown(tags(ID_COLS,"tag-id"), unsafe_allow_html=True)
        st.markdown("**👤 Demographics**")
        st.markdown(tags(DEMO_COLS,"tag-num"), unsafe_allow_html=True)
        st.markdown("**📊 Categorical**")
        st.markdown(tags(cat_cols,"tag-cat"), unsafe_allow_html=True)
        st.markdown("**🔘 Binary flags**")
        st.markdown(tags(binary_cols,"tag-bin"), unsafe_allow_html=True)
        st.markdown("**🩺 APACHE scores**")
        st.markdown(tags(apache_cols[:30],"tag-num"), unsafe_allow_html=True)
        st.markdown("**💊 Day-1 vitals (d1_)**")
        st.markdown(tags(vital_d1[:40],"tag-num"), unsafe_allow_html=True)
        st.markdown("**⏱️ Hour-1 vitals (h1_)**")
        st.markdown(tags(vital_h1[:40],"tag-num"), unsafe_allow_html=True)

    with st.expander("Raw data preview (first 100 rows)"):
        st.dataframe(df_raw.head(100), use_container_width=True)

    with st.expander("Column-level info"):
        info = []
        for col in all_cols:
            info.append({
                "Column":   col,
                "Dtype":    str(df_raw[col].dtype),
                "Non-null": int(df_raw[col].notna().sum()),
                "Null %":   f"{100*df_raw[col].isna().mean():.1f}%",
                "Unique":   df_raw[col].nunique(),
                "Example":  str(df_raw[col].dropna().iloc[0]) if df_raw[col].notna().any() else "—",
            })
        st.dataframe(pd.DataFrame(info).set_index("Column"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · DATA QUALITY & MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
if "2 · Data Quality & Missing Values" in active:
    st.markdown('<p class="section-title">🩺 2 · Data Quality & Missing Values</p>', unsafe_allow_html=True)

    miss     = df_raw.isnull().sum()
    miss_pct = (miss / len(df_raw) * 100).round(2)
    miss_df  = pd.DataFrame({"Missing N": miss, "Missing %": miss_pct})\
                 .query("`Missing N` > 0").sort_values("Missing %", ascending=False)

    col1, col2, col3 = st.columns(3)
    col1.metric("Columns with missing", len(miss_df))
    col2.metric("Cols >50% missing",    int((miss_pct > 50).sum()))
    col3.metric("Duplicate rows",       int(df_raw.duplicated().sum()))

    tab_heat, tab_group, tab_advice = st.tabs(["Missing Heatmap", "By Feature Group", "Recommendations"])

    with tab_heat:
        fig, ax = plt.subplots(figsize=(8, max(4, len(miss_df)*0.22)))
        colors = [DANGER if p>50 else WARN if p>20 else ACCENT for p in miss_df["Missing %"]]
        ax.barh(miss_df.index[::-1], miss_df["Missing %"][::-1], color=colors[::-1])
        ax.set_xlabel("% missing")
        ax.set_title("Missing values per column")
        for v in [20, 50, 80]:
            ax.axvline(v, color="gray", ls="--", lw=0.7, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with tab_group:
        groups = {
            "Hour-1 vitals (h1_)": [c for c in miss_df.index if c.startswith("h1_")],
            "Day-1 vitals (d1_)":  [c for c in miss_df.index if c.startswith("d1_")],
            "APACHE scores":       [c for c in miss_df.index if "apache" in c.lower()],
            "Demographics":        [c for c in miss_df.index if c in DEMO_COLS],
            "Other":               [c for c in miss_df.index
                                    if not c.startswith("h1_") and not c.startswith("d1_")
                                    and "apache" not in c.lower() and c not in DEMO_COLS],
        }
        summary = []
        for grp, cols_g in groups.items():
            if cols_g:
                avg_miss = miss_df.loc[[c for c in cols_g if c in miss_df.index], "Missing %"].mean()
                summary.append({"Group": grp, "# Cols with missing": len(cols_g),
                                 "Avg missing %": f"{avg_miss:.1f}%"})
        st.dataframe(pd.DataFrame(summary).set_index("Group"), use_container_width=True)
        st.info("ℹ️ Hour-1 (h1_) features are much more sparse than Day-1 (d1_) features — "
                "this is clinically expected as not all tests are done in the first hour of admission.")

    with tab_advice:
        st.markdown("""
**Recommended strategy for this dataset:**
- 🔴 **>80% missing** (h1_ lab values): Consider **dropping** — too sparse to impute reliably
- 🟡 **20–80% missing** (d1_ lab values): **Median imputation** + add a **missingness indicator** binary column
- 🟢 **<20% missing** (demographics, APACHE): Safe for **simple mean/median imputation**
- ⭐ **Best practice**: **Multiple imputation** (IterativeImputer / MICE) for clinically important labs
- 🔘 **Binary flag columns** (aids, cirrhosis, etc.): Treat NaN as 0 — absence of diagnosis record implies not diagnosed
        """)


# ══════════════════════════════════════════════════════════════════════════════
# 3 · DEMOGRAPHICS & PATIENT PROFILE
# ══════════════════════════════════════════════════════════════════════════════
if "3 · Demographics & Patient Profile" in active:
    st.markdown('<p class="section-title">👤 3 · Demographics & Patient Profile</p>', unsafe_allow_html=True)

    present_demo = [c for c in DEMO_COLS if c in all_cols]
    present_cat  = [c for c in cat_cols if c in all_cols]

    tab_num_d, tab_cat_d, tab_vs_t = st.tabs(["Numeric Demographics", "Categorical Demographics", "vs Mortality"])

    with tab_num_d:
        num_demo = [c for c in present_demo if c in num_cols or c in ["age","bmi","height","weight"]]
        num_demo = [c for c in num_demo if c in all_cols]
        if num_demo:
            ncols_g = 3
            nrows_g = max(1,(len(num_demo)+ncols_g-1)//ncols_g)
            fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15, 4*nrows_g), squeeze=False)
            flat = axes.flatten()
            for i, col in enumerate(num_demo):
                ax = flat[i]
                data = df[col].dropna()
                if len(data) < 2:
                    ax.set_visible(False); continue
                ax.hist(data, bins=30, color=ACCENT, alpha=0.65, density=True)
                if len(data) > 5:
                    from scipy.stats import gaussian_kde
                    xs = np.linspace(data.min(), data.max(), 200)
                    ax.plot(xs, gaussian_kde(data)(xs), color=ACCENT2, lw=2)
                ax.set_title(f"{col}  (skew={data.skew():.2f})", fontsize=9)
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for j in range(len(num_demo), len(flat)): flat[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_cat_d:
        if present_cat:
            ncols_g = 2
            nrows_g = max(1,(len(present_cat)+1)//ncols_g)
            fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(14, 4.5*nrows_g), squeeze=False)
            flat = axes.flatten()
            for i, col in enumerate(present_cat):
                ax = flat[i]
                vc = df[col].value_counts()
                colors = plt.cm.get_cmap("cool")(np.linspace(0.2, 0.8, len(vc)))
                ax.barh(vc.index.astype(str)[::-1], vc.values[::-1], color=colors[::-1])
                ax.set_title(col, fontsize=9)
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            for j in range(len(present_cat), len(flat)): flat[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_vs_t:
        tgt = target_col if target_col else TARGET_COL
        if tgt in all_cols:
            compare_cols = [c for c in ["age","bmi","gender","ethnicity","icu_type",
                                        "hospital_admit_source","elective_surgery"] if c in all_cols]
            chosen = st.multiselect("Select columns to compare vs mortality",
                                    compare_cols, default=compare_cols[:4], key="demo_tgt")
            for col in chosen:
                fig, ax = plt.subplots(figsize=(8, 3.5))
                if col in num_cols or col in ["age","bmi"]:
                    alive = df[df[tgt]==0][col].dropna()
                    dead  = df[df[tgt]==1][col].dropna()
                    ax.hist(alive, bins=30, alpha=0.55, color=SUCCESS, label="Survived", density=True)
                    ax.hist(dead,  bins=30, alpha=0.55, color=DANGER,  label="Died",     density=True)
                    ax.legend()
                    ax.set_title(f"{col} by Mortality Outcome")
                else:
                    ct = df.groupby(col)[tgt].mean().sort_values(ascending=False)
                    overall_rate = df[tgt].mean() * 100
                    ax.barh(ct.index.astype(str)[::-1], ct.values[::-1]*100,
                            color=[DANGER if v*100 > overall_rate else ACCENT for v in ct.values[::-1]])
                    ax.set_xlabel("Mortality rate (%)")
                    ax.set_title(f"Mortality rate by {col}")
                    ax.axvline(overall_rate, color="gray", ls="--", lw=1.5, label="Overall avg")
                    ax.legend(fontsize=8)
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4 · VITAL SIGNS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if "4 · Vital Signs Analysis" in active:
    st.markdown('<p class="section-title">💓 4 · Vital Signs Analysis</p>', unsafe_allow_html=True)

    tab_d1, tab_h1, tab_compare = st.tabs(["Day-1 Vitals", "Hour-1 Vitals", "D1 vs H1 Comparison"])

    def plot_vitals(cols, key_prefix, color):
        if not cols:
            st.info("No columns available."); return
        chosen = st.multiselect("Select columns", cols,
                                default=cols[:min(6,len(cols))], key=f"vit_{key_prefix}")
        if not chosen: return
        ncols_g = 3
        nrows_g = max(1,(len(chosen)+ncols_g-1)//ncols_g)
        fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15, 4*nrows_g), squeeze=False)
        flat = axes.flatten()
        for i, col in enumerate(chosen):
            ax = flat[i]
            data = df[col].dropna()
            if len(data) < 2: ax.set_visible(False); continue
            ax.hist(data, bins=30, color=color, alpha=0.65, density=True)
            miss_pct_col = 100*df_raw[col].isna().mean()
            ax.set_title(f"{col}\nmissing={miss_pct_col:.0f}%", fontsize=8)
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for j in range(len(chosen), len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_d1: plot_vitals(vital_d1, "d1", ACCENT)
    with tab_h1: plot_vitals(vital_h1, "h1", "#8B5CF6")

    with tab_compare:
        st.markdown("Compare a vital sign measured over Day-1 vs Hour-1:")
        d1_bases = set(c.replace("d1_","").replace("_max","").replace("_min","") for c in vital_d1)
        h1_bases = set(c.replace("h1_","").replace("_max","").replace("_min","") for c in vital_h1)
        common   = sorted(d1_bases & h1_bases)
        if not common:
            st.info("No matching D1/H1 pairs found.")
        else:
            base = st.selectbox("Base vital", common, key="vh_base")
            d1_match = [c for c in vital_d1 if base in c]
            h1_match = [c for c in vital_h1 if base in c]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, match_cols, label in zip(axes, [d1_match, h1_match], ["Day 1","Hour 1"]):
                for col in match_cols[:2]:
                    data = df[col].dropna()
                    if len(data) > 0:
                        ax.hist(data, bins=30, alpha=0.5, label=col, density=True)
                ax.set_title(f"{label} — {base}"); ax.legend(fontsize=8)
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5 · LAB VALUES ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if "5 · Lab Values Analysis" in active:
    st.markdown('<p class="section-title">🧪 5 · Lab Values Analysis</p>', unsafe_allow_html=True)

    all_lab_present = [c for c in lab_cols if c in all_cols]
    chosen_labs = st.multiselect("Select lab columns", all_lab_present,
                                 default=all_lab_present[:min(6,len(all_lab_present))], key="lab_sel")

    if chosen_labs:
        ncols_g = 3
        nrows_g = max(1,(len(chosen_labs)+ncols_g-1)//ncols_g)
        fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15, 4*nrows_g), squeeze=False)
        flat = axes.flatten()
        for i, col in enumerate(chosen_labs):
            ax = flat[i]
            data = df[col].dropna()
            if len(data) < 2: ax.set_visible(False); continue
            ax.hist(data, bins=30, color=WARN, alpha=0.65, density=True)
            miss_pct_col = 100*df_raw[col].isna().mean()
            ax.set_title(f"{col}\nmissing={miss_pct_col:.0f}% | skew={data.skew():.2f}", fontsize=8)
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        for j in range(len(chosen_labs), len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        tgt = target_col if target_col else TARGET_COL
        if tgt in all_cols:
            with st.expander("Lab values vs mortality outcome"):
                col_compare = st.selectbox("Select lab column", chosen_labs, key="lab_tgt")
                fig, ax = plt.subplots(figsize=(7, 4))
                alive = df[df[tgt]==0][col_compare].dropna()
                dead  = df[df[tgt]==1][col_compare].dropna()
                ax.hist(alive, bins=30, alpha=0.55, color=SUCCESS, label=f"Survived (n={len(alive):,})", density=True)
                ax.hist(dead,  bins=30, alpha=0.55, color=DANGER,  label=f"Died (n={len(dead):,})",     density=True)
                ax.legend(); ax.set_title(f"{col_compare} by Outcome")
                if len(alive)>2 and len(dead)>2:
                    _, p = stats.mannwhitneyu(alive, dead, alternative="two-sided")
                    ax.set_xlabel(f"Mann-Whitney U  p={p:.3e}")
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 6 · APACHE SCORES
# ══════════════════════════════════════════════════════════════════════════════
if "6 · APACHE Scores" in active:
    st.markdown('<p class="section-title">📈 6 · APACHE Scores</p>', unsafe_allow_html=True)

    apache_probs = [c for c in ["apache_4a_hospital_death_prob","apache_4a_icu_death_prob"] if c in all_cols]
    tgt = target_col if target_col else TARGET_COL

    if apache_probs and tgt in all_cols:
        st.markdown("#### APACHE-predicted vs actual mortality")
        fig, axes = plt.subplots(1, len(apache_probs), figsize=(6*len(apache_probs), 4))
        if len(apache_probs) == 1: axes = [axes]
        for ax, prob_col in zip(axes, apache_probs):
            alive = df[df[tgt]==0][prob_col].dropna()
            dead  = df[df[tgt]==1][prob_col].dropna()
            ax.hist(alive, bins=30, alpha=0.55, color=SUCCESS, label="Survived", density=True)
            ax.hist(dead,  bins=30, alpha=0.55, color=DANGER,  label="Died",     density=True)
            ax.axvline(0.5, color="black", ls="--", lw=1.5)
            ax.set_title(prob_col.replace("_"," ")); ax.legend(fontsize=8)
            for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        try:
            from sklearn.metrics import roc_auc_score, roc_curve
            for prob_col in apache_probs:
                valid = df[[prob_col, tgt]].dropna()
                if len(valid) > 100:
                    auc = roc_auc_score(valid[tgt], valid[prob_col])
                    fpr, tpr, _ = roc_curve(valid[tgt], valid[prob_col])
                    fig, ax = plt.subplots(figsize=(5,4))
                    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"APACHE AUC={auc:.3f}")
                    ax.plot([0,1],[0,1],"k--",lw=1)
                    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
                    ax.set_title(f"ROC Curve — {prob_col}"); ax.legend()
                    plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
                    st.info(f"ℹ️ **{prob_col}** AUROC = **{auc:.3f}** — this is your **baseline to beat** with your ML model!")
        except ImportError:
            st.warning("Install scikit-learn for ROC curves.")

    with st.expander("All APACHE numeric columns — summary stats"):
        apache_present = [c for c in apache_cols if c in all_cols]
        if apache_present:
            chosen_ap = st.multiselect("Select", apache_present,
                                       default=apache_present[:min(6,len(apache_present))], key="ap_sel")
            if chosen_ap:
                st.dataframe(df[chosen_ap].describe().T.style.format("{:.3f}"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · CORRELATION & MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════════════════════
if "7 · Correlation & Multicollinearity" in active:
    st.markdown('<p class="section-title">🕸️ 7 · Correlation & Multicollinearity</p>', unsafe_allow_html=True)

    corr_options = {
        "Demographics":             [c for c in DEMO_COLS if c in num_cols],
        "APACHE scores":            [c for c in apache_cols if c in all_cols][:20],
        "Day-1 vitals":             vital_d1[:20],
        "Hour-1 vitals":            vital_h1[:20],
        "All numeric (sample 30)":  num_cols[:30],
    }
    group_choice  = st.selectbox("Column group", list(corr_options.keys()), key="corr_grp")
    cols_for_corr = [c for c in corr_options[group_choice] if c in all_cols]

    if len(cols_for_corr) >= 2:
        method = st.radio("Method", ["pearson","spearman"], horizontal=True, key="corr_meth")
        corr   = df[cols_for_corr].corr(method=method)

        fig, ax = plt.subplots(figsize=(max(7,len(cols_for_corr)*0.75),
                                        max(6,len(cols_for_corr)*0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(cols_for_corr)<=15, fmt=".2f",
                    cmap=palette, center=0, ax=ax, linewidths=0.3,
                    square=True, annot_kws={"size":7}, vmin=-1, vmax=1)
        ax.set_title(f"{method.capitalize()} correlation — {group_choice}")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        high = (corr.where(~mask).abs().stack()
                    .reset_index().rename(columns={"level_0":"A","level_1":"B",0:"r"})
                    .query("r > 0.85").sort_values("r", ascending=False))
        if not high.empty:
            st.warning(f"⚠️ {len(high)} highly correlated pairs (|r|>0.85) — multicollinearity risk.")
            st.dataframe(high.style.format({"r":"{:.3f}"}), use_container_width=True)
            st.markdown("> **Tip:** For logistic regression, drop one from each pair. "
                        "XGBoost/Random Forest handle this automatically.")
    else:
        st.info("Not enough columns in this group for correlation.")


# ══════════════════════════════════════════════════════════════════════════════
# 8 · OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════
if "8 · Outlier Detection" in active:
    st.markdown('<p class="section-title">🚨 8 · Outlier Detection</p>', unsafe_allow_html=True)

    out_group = st.selectbox("Column group", ["Demographics","Day-1 vitals","Hour-1 vitals","APACHE scores"], key="out_grp")
    group_map = {
        "Demographics":  [c for c in DEMO_COLS if c in num_cols],
        "Day-1 vitals":  vital_d1[:12],
        "Hour-1 vitals": vital_h1[:12],
        "APACHE scores": [c for c in apache_cols if c in all_cols][:12],
    }
    o_cols   = [c for c in group_map[out_group] if c in all_cols]
    o_method = st.radio("Method", ["IQR (1.5×)","Z-score (|z|>3)"], horizontal=True, key="out_m")

    if o_cols:
        rows = []
        for col in o_cols:
            s = df_raw[col].dropna()
            if len(s) < 5: continue
            if o_method == "IQR (1.5×)":
                q1,q3 = s.quantile(.25), s.quantile(.75); iqr = q3-q1
                mask_o = (s < q1-1.5*iqr) | (s > q3+1.5*iqr)
            else:
                mask_o = ((s-s.mean())/s.std()).abs() > 3
            rows.append({
                "Column":    col,
                "Outliers":  int(mask_o.sum()),
                "% outlier": f"{100*mask_o.sum()/len(s):.2f}%",
                "Min":       f"{s.min():.2f}",
                "Max":       f"{s.max():.2f}",
                "Note":      "⚠️ Review" if mask_o.sum()/len(s) > 0.05 else "✅ OK",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows).set_index("Column"), use_container_width=True)

        ncols_g = 3
        nrows_g = max(1,(len(o_cols)+ncols_g-1)//ncols_g)
        fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15, 4*nrows_g), squeeze=False)
        flat = axes.flatten()
        for i, col in enumerate(o_cols):
            ax = flat[i]
            data = df[col].dropna()
            if len(data) < 2: ax.set_visible(False); continue
            ax.boxplot(data, vert=True, patch_artist=True,
                       boxprops=dict(facecolor=ACCENT, alpha=0.5),
                       flierprops=dict(marker="o", color=ACCENT2, markersize=3, alpha=0.4))
            ax.set_title(col, fontsize=8)
        for j in range(len(o_cols), len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 9 · CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════════
if "9 · Class Balance (Mortality Rate)" in active:
    st.markdown('<p class="section-title">⚖️ 9 · Class Balance — Mortality Rate</p>', unsafe_allow_html=True)

    tgt = target_col if target_col else TARGET_COL
    if tgt not in all_cols:
        st.info("Target column not found in dataset.")
    else:
        vc    = df_raw[tgt].value_counts().sort_index()
        ratio = vc.min() / vc.max()

        c1,c2,c3 = st.columns(3)
        c1.metric("Survived (0)",   f"{vc.get(0,0):,}")
        c2.metric("Died (1)",       f"{vc.get(1,0):,}")
        c3.metric("Mortality rate", f"{100*vc.get(1,0)/len(df_raw):.1f}%")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].bar(["Survived","Died"], [vc.get(0,0),vc.get(1,0)],
                    color=[SUCCESS,DANGER], alpha=0.8)
        axes[0].set_title("Class Distribution"); axes[0].set_ylabel("Count")
        for spine in ["top","right"]: axes[0].spines[spine].set_visible(False)

        axes[1].pie([vc.get(0,0),vc.get(1,0)], labels=["Survived","Died"],
                    colors=[SUCCESS,DANGER], autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor":"white","linewidth":2})
        axes[1].set_title("Mortality Proportion")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.error(f"🔴 **Severe class imbalance** (ratio={ratio:.3f}) — only ~8.6% mortality rate.")
        st.markdown("""
**This is critical for your model.** Without handling imbalance:
- The model will predict "survived" for almost everyone and achieve ~91% accuracy — **misleading!**
- Use **AUROC, F1, Precision-Recall AUC** as metrics — NOT raw accuracy

**Recommended strategies:**
- `class_weight='balanced'` in Logistic Regression / Random Forest
- `scale_pos_weight` in XGBoost
- **SMOTE** oversampling on the training set only (after train/test split)
- **Precision-Recall curves** are more informative than ROC for imbalanced data
        """)

        st.markdown("#### Mortality rate by subgroup")
        subgroup_col = st.selectbox("Subgroup by",
                                    [c for c in ["gender","ethnicity","icu_type",
                                                 "hospital_admit_source","elective_surgery"] if c in all_cols],
                                    key="sg_col")
        sg = df_raw.groupby(subgroup_col)[tgt].agg(["mean","sum","count"]).reset_index()
        sg.columns = [subgroup_col,"Mortality Rate","Deaths","Total"]
        sg["Mortality Rate %"] = (sg["Mortality Rate"]*100).round(2)
        overall = df_raw[tgt].mean()*100

        fig, ax = plt.subplots(figsize=(8, max(3, len(sg)*0.5)))
        colors = [DANGER if v > overall else ACCENT for v in sg["Mortality Rate %"]]
        ax.barh(sg[subgroup_col].astype(str), sg["Mortality Rate %"], color=colors)
        ax.axvline(overall, color="black", ls="--", lw=1.5, label=f"Overall ({overall:.1f}%)")
        ax.set_xlabel("Mortality rate %"); ax.set_title(f"Mortality by {subgroup_col}")
        ax.legend(fontsize=8)
        for spine in ["top","right"]: ax.spines[spine].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.dataframe(sg.set_index(subgroup_col).style.format({"Mortality Rate %":"{:.2f}%"}),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 10 · FEATURE IMPORTANCE VS MORTALITY
# ══════════════════════════════════════════════════════════════════════════════
if "10 · Feature Importance vs Mortality" in active:
    st.markdown('<p class="section-title">🎯 10 · Feature Importance vs Mortality</p>', unsafe_allow_html=True)

    tgt = target_col if target_col else TARGET_COL
    if tgt not in all_cols:
        st.info("Target column not found.")
    else:
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline

            fi_cols = [c for c in num_cols if c in all_cols and c not in ID_COLS][:60]
            fi_cols += [c for c in binary_cols if c in all_cols]

            y = df_raw[tgt].dropna()
            X = df_raw.loc[y.index, fi_cols]

            n_samples = min(20000, len(X))
            idx = X.sample(n_samples, random_state=42).index
            X_s, y_s = X.loc[idx], y.loc[idx]

            model = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf",  RandomForestClassifier(n_estimators=100, random_state=42,
                                               class_weight="balanced", n_jobs=-1))
            ])
            with st.spinner("Fitting Random Forest for feature importance (~30s)…"):
                model.fit(X_s, y_s)

            fi = pd.DataFrame({"Feature": fi_cols,
                                "Importance": model.named_steps["rf"].feature_importances_})\
                   .sort_values("Importance", ascending=False)

            c1, c2 = st.columns([2,1])
            with c1:
                top_n = st.slider("Top N features", 10, min(50,len(fi)), 20, key="fi_topn")
                top   = fi.head(top_n)
                fig, ax = plt.subplots(figsize=(8, top_n*0.38))
                colors = plt.cm.get_cmap("viridis")(np.linspace(0.1,0.9,len(top)))
                ax.barh(top["Feature"][::-1], top["Importance"][::-1], color=colors[::-1])
                ax.set_title("Random Forest Feature Importances (class_weight=balanced)")
                ax.set_xlabel("Gini importance")
                for spine in ["top","right"]: ax.spines[spine].set_visible(False)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            with c2:
                st.dataframe(fi.head(top_n).style.format({"Importance":"{:.4f}"}), use_container_width=True)

            top3 = fi.head(3)["Feature"].tolist()
            st.success(f"✅ Top 3 predictors of mortality: **{', '.join(top3)}**")
            st.info("💡 Compare these with APACHE score components — do they align with clinical knowledge?")

        except ImportError:
            st.warning("Install scikit-learn: `pip install scikit-learn`")


# ══════════════════════════════════════════════════════════════════════════════
# 11 · NORMALITY & DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
if "11 · Normality & Distributions" in active:
    st.markdown('<p class="section-title">🔔 11 · Normality & Distributions</p>', unsafe_allow_html=True)

    norm_group = st.selectbox("Group", ["Demographics","Day-1 vitals","APACHE scores"], key="norm_grp")
    group_map2 = {
        "Demographics":  [c for c in DEMO_COLS if c in num_cols],
        "Day-1 vitals":  vital_d1[:9],
        "APACHE scores": [c for c in apache_cols if c in all_cols][:9],
    }
    norm_cols = [c for c in group_map2[norm_group] if c in all_cols]

    if norm_cols:
        results = []
        for col in norm_cols:
            data = df_raw[col].dropna()
            sample = data.sample(min(5000, len(data)), random_state=42)
            if len(sample) < 8: continue
            _, p = stats.normaltest(sample)
            results.append({"Column":col, "N":len(data),
                             "Skewness":f"{data.skew():.3f}", "Kurtosis":f"{data.kurt():.3f}",
                             "Normal?":"✅" if p>0.05 else "❌", "p-value":f"{p:.3e}"})
        if results:
            st.dataframe(pd.DataFrame(results).set_index("Column"), use_container_width=True)

        ncols_g = 3
        nrows_g = max(1,(len(norm_cols)+ncols_g-1)//ncols_g)
        fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15,4*nrows_g), squeeze=False)
        flat = axes.flatten()
        for i, col in enumerate(norm_cols):
            ax = flat[i]
            data = df_raw[col].dropna()
            sample = data.sample(min(2000, len(data)), random_state=42)
            if len(sample) < 8: ax.set_visible(False); continue
            stats.probplot(sample, dist="norm", plot=ax)
            ax.set_title(f"Q-Q: {col}", fontsize=9)
            ax.get_lines()[0].set(markersize=2, alpha=0.3, color=ACCENT)
            ax.get_lines()[1].set(color=ACCENT2, lw=1.5)
        for j in range(len(norm_cols), len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown("> Most clinical lab values and vitals are **right-skewed**. "
                    "Consider **log transform** before logistic regression. "
                    "XGBoost and Random Forest are **invariant** to distribution shape.")


# ══════════════════════════════════════════════════════════════════════════════
# 12 · ML READINESS REPORT
# ══════════════════════════════════════════════════════════════════════════════
if "12 · ML Readiness Report" in active:
    st.markdown('<p class="section-title">🤖 12 · ML Readiness Report</p>', unsafe_allow_html=True)

    tgt        = target_col if target_col else TARGET_COL
    n_miss_cols = (df_raw.isnull().mean() > 0).sum()
    n_dups     = df_raw.duplicated().sum()
    n_features = len(num_cols) + len(cat_cols) + len(binary_cols)

    st.markdown("#### ✅ Pre-Modeling Checklist")
    checks = [
        ("ID columns identified",             True,  f"Exclude: {', '.join(ID_COLS)}"),
        ("Target column set",                 tgt in all_cols, "Set hospital_death as target"),
        ("Missing values — action required",  n_miss_cols>0,   f"{n_miss_cols} cols have missing values"),
        ("No duplicate rows",                 n_dups==0,       f"{n_dups} duplicates"),
        ("Class imbalance — must address",    False,           "~8.6% mortality — use class_weight or SMOTE"),
        ("Categorical cols need encoding",    True,            f"{len(cat_cols)} cols need OHE/target encoding"),
        ("Scale features for linear models",  True,            "StandardScaler before LR/SVM"),
        ("Feature selection recommended",     True,            "Consider dropping h1_ cols with >80% missing"),
        ("Correct evaluation metrics",        True,            "Use AUROC & F1 — NOT raw accuracy"),
    ]
    for label, is_warning, note in checks:
        icon = "⚠️" if is_warning else "✅"
        st.markdown(f"{icon} **{label}** — *{note}*")

    st.markdown("#### 🤖 Recommended Models")
    html = '<div class="recommend-box">'
    html += '<div style="font-size:1.05rem;font-weight:700;color:#0369A1;margin-bottom:.6rem;">🎯 For ICU Mortality (Imbalanced Binary Classification)</div>'
    html += '<ol style="margin:0;padding-left:1.3rem;">'
    recs = [
        ("Logistic Regression + L2 + class_weight='balanced'",
         "Interpretable baseline. Doctors can understand odds ratios. Required for Meeting 2."),
        ("Random Forest + class_weight='balanced'",
         "Handles high dimensionality, missing indicators, non-linearity. Strong ensemble model."),
        ("XGBoost / LightGBM + scale_pos_weight",
         "State-of-the-art on tabular clinical data. Handles missingness natively. Best expected AUROC."),
    ]
    for name, reason in recs:
        html += f"<li style='margin-bottom:.5rem'><strong>{name}</strong><br>"
        html += f"<span style='color:#475569;font-size:.9rem'>{reason}</span></li>"
    html += "</ol></div>"
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("#### 🛠️ Preprocessing Roadmap")
    steps = [
        ("1",  "Drop ID columns",             "`encounter_id`, `patient_id`, `hospital_id`, `icu_id`"),
        ("2",  "Drop readmission_status",      "Zero variance — only 1 unique value in this dataset"),
        ("3",  "Handle h1_ missing cols",      "Drop cols >80% missing OR keep + add `_was_missing` flag"),
        ("4",  "Impute d1_ lab values",        "Median imputation + binary missingness indicator columns"),
        ("5",  "Binary flag NaN → 0",          "`aids`, `cirrhosis`, etc. — NaN likely means not diagnosed"),
        ("6",  "Encode categoricals",          "OHE for gender/ethnicity; Target Encoding for hospital_id/icu_type"),
        ("7",  "Scale numeric features",       "StandardScaler — required for Logistic Regression"),
        ("8",  "Handle class imbalance",       "SMOTE on train set only, or class_weight='balanced'"),
        ("9",  "Train/val/test split",         "70/15/15 stratified on `hospital_death`"),
        ("10", "Evaluation metrics",           "AUROC, F1-score, Precision-Recall AUC — NOT accuracy"),
        ("11", "Baseline to beat",             "APACHE-4a AUROC ≈ 0.85 — aim to match or exceed this"),
    ]
    for num, title, detail in steps:
        st.markdown(f"**{num}.** **{title}** — {detail}")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("ICU Mortality EDA Explorer · Healthcare Predictive Modeling project · "
           "Streamlit · Pandas · Seaborn · Scikit-learn · SciPy")

"""
ICU Mortality EDA Explorer
==========================
Tailored for the WIDS Datathon ICU dataset (training_v2.csv)

Requirements:
    pip install streamlit pandas numpy matplotlib seaborn scipy scikit-learn pyarrow openpyxl xlrd

Run:
    streamlit run eda_ml_app.py
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

st.set_page_config(page_title="ICU EDA Explorer", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

# ── Medical color palette ──────────────────────────────────────────────────────
C_BLUE  = "#1A5276"
C_TEAL  = "#148F77"
C_RED   = "#C0392B"
C_GREEN = "#1E8449"
C_AMBER = "#D68910"
C_GRID  = "#D5D8DC"

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": C_GRID, "axes.grid": True, "grid.color": C_GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
})

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&display=swap');
html, body, [class*="css"] {{ font-family: 'Source Sans 3', sans-serif; }}
.main-title {{ font-size:2.2rem; font-weight:700; color:{C_BLUE};
    border-bottom:3px solid {C_TEAL}; padding-bottom:.4rem; margin-bottom:.2rem; }}
.subtitle {{ color:#5D6D7E; font-size:.95rem; margin-bottom:1.5rem; }}
.section-title {{ font-size:1.2rem; font-weight:700; color:{C_BLUE};
    border-left:4px solid {C_TEAL}; padding:.4rem .7rem;
    background:#EBF5FB; border-radius:0 6px 6px 0; margin-top:2rem; margin-bottom:1rem; }}
.kpi-box {{ background:white; border:1px solid {C_GRID}; border-top:3px solid {C_TEAL};
    border-radius:6px; padding:.9rem 1.1rem; text-align:center; }}
.kpi-val {{ font-size:1.8rem; font-weight:700; color:{C_BLUE}; }}
.kpi-lbl {{ font-size:.78rem; color:#7F8C8D; text-transform:uppercase; letter-spacing:.05em; }}
.alert-red   {{ background:#FDEDEC; border:1px solid {C_RED};   border-radius:6px; padding:.7rem 1rem; color:{C_RED};   margin:.5rem 0; }}
.alert-green {{ background:#EAFAF1; border:1px solid {C_GREEN}; border-radius:6px; padding:.7rem 1rem; color:{C_GREEN}; margin:.5rem 0; }}
.alert-amber {{ background:#FEF9E7; border:1px solid {C_AMBER}; border-radius:6px; padding:.7rem 1rem; color:#7D6608;   margin:.5rem 0; }}
.tag {{ display:inline-block; border-radius:4px; padding:2px 8px; font-size:.75rem; font-weight:600; margin:2px; }}
.tag-num {{ background:#D6EAF8; color:{C_BLUE}; }}
.tag-cat {{ background:#D5F5E3; color:#1E8449; }}
.tag-bin {{ background:#FDEBD0; color:#A04000; }}
.tag-id  {{ background:#F2F3F4; color:#7F8C8D; }}
</style>
""", unsafe_allow_html=True)

# ── Column groups ──────────────────────────────────────────────────────────────
ID_COLS        = ["encounter_id","patient_id","hospital_id","icu_id"]
TARGET_COL     = "hospital_death"
DEMO_COLS      = ["age","gender","ethnicity","bmi","height","weight"]
BINARY_FLAGS   = ["elective_surgery","apache_post_operative","arf_apache","gcs_unable_apache",
                  "intubated_apache","ventilated_apache","aids","cirrhosis","diabetes_mellitus",
                  "hepatic_failure","immunosuppression","leukemia","lymphoma",
                  "solid_tumor_with_metastasis","readmission_status"]
CAT_COLS_KNOWN = ["ethnicity","gender","hospital_admit_source","icu_admit_source",
                  "icu_stay_type","icu_type","apache_3j_bodysystem","apache_2_bodysystem"]

st.markdown('<p class="main-title">🏥 ICU Mortality — Exploratory Data Analysis</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">WIDS Datathon dataset · Healthcare Predictive Modeling Project</p>', unsafe_allow_html=True)

uploaded = st.file_uploader("Upload dataset (CSV · Excel · JSON · Parquet)",
                            type=["csv","xlsx","xls","json","parquet"])

@st.cache_data(show_spinner=False)
def load_file(f):
    n = f.name.lower()
    if n.endswith(".csv"):             return pd.read_csv(f)
    elif n.endswith((".xlsx",".xls")): return pd.read_excel(f)
    elif n.endswith(".json"):          return pd.read_json(f)
    elif n.endswith(".parquet"):       return pd.read_parquet(f)

if uploaded is None:
    st.info("👆 Upload `training_v2.csv` to begin.")
    st.stop()

with st.spinner("Loading data…"):
    df_raw = load_file(uploaded)

all_cols    = df_raw.columns.tolist()
num_cols    = [c for c in df_raw.select_dtypes(include=np.number).columns
               if c not in ID_COLS and c != TARGET_COL and c not in BINARY_FLAGS]
cat_cols    = [c for c in CAT_COLS_KNOWN if c in all_cols]
binary_cols = [c for c in BINARY_FLAGS if c in all_cols]
apache_num  = [c for c in all_cols if "apache" in c.lower() and c not in BINARY_FLAGS
               and c not in cat_cols and c != TARGET_COL and c in num_cols]
vital_d1    = [c for c in all_cols if c.startswith("d1_") and c in num_cols]
vital_h1    = [c for c in all_cols if c.startswith("h1_") and c in num_cols]
apache_prob = [c for c in ["apache_4a_hospital_death_prob","apache_4a_icu_death_prob"] if c in all_cols]

with st.sidebar:
    st.markdown(f"<div style='color:{C_BLUE};font-weight:700;font-size:1.05rem;'>⚙️ Settings</div>",
                unsafe_allow_html=True)
    sample_pct = st.slider("Sample % for heavy plots", 10, 100, 100, 5)
    palette    = st.selectbox("Heatmap palette", ["RdBu_r","coolwarm","Blues","YlOrRd","PuOr"])
    st.markdown("---")
    st.markdown(f"<div style='color:{C_BLUE};font-weight:700;'>🎯 Target column</div>",
                unsafe_allow_html=True)
    t_opts     = ["— none —"] + all_cols
    t_def      = t_opts.index(TARGET_COL) if TARGET_COL in t_opts else 0
    target_col = st.selectbox("", t_opts, index=t_def, label_visibility="collapsed")
    if target_col == "— none —": target_col = None
    st.markdown("---")
    SECTIONS = [
        "1 · Overview",
        "2 · Missing Values",
        "3 · Class Balance & Mortality Rate",
        "4 · Demographics & Segmentation",
        "5 · Mortality by ICU Unit",
        "6 · Surgery vs Emergency Patients",
        "7 · Vital Signs Analysis",
        "8 · Age vs APACHE Score",
        "9 · Vital Signs vs Mortality (Heatmap)",
        "10 · Correlation Analysis",
        "11 · APACHE Scores",
        "12 · Outlier Detection",
        "13 · Normality & Distributions",
    ]
    active = st.multiselect("📑 Sections", SECTIONS, default=SECTIONS)

tgt = target_col if target_col else TARGET_COL
df  = df_raw.sample(frac=sample_pct/100, random_state=42) if sample_pct < 100 else df_raw.copy()


# ══════════════════════════════════════════════════════════════════════════════
# 1 · OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "1 · Overview" in active:
    st.markdown('<p class="section-title">📋 1 · Dataset Overview</p>', unsafe_allow_html=True)
    mort_rate = df_raw[TARGET_COL].mean()*100 if TARGET_COL in all_cols else 0
    miss_pct  = df_raw.isnull().mean().mean()*100
    cols = st.columns(5)
    for col, (lbl,val) in zip(cols, [
        ("Patients", f"{df_raw.shape[0]:,}"), ("Features", f"{df_raw.shape[1]}"),
        ("Mortality Rate", f"{mort_rate:.1f}%"), ("Avg Missing", f"{miss_pct:.1f}%"),
        ("ICU Types", str(df_raw["icu_type"].nunique()) if "icu_type" in all_cols else "—"),
    ]):
        col.markdown(f'<div class="kpi-box"><div class="kpi-val">{val}</div>'
                     f'<div class="kpi-lbl">{lbl}</div></div>', unsafe_allow_html=True)

    with st.expander("Column catalogue"):
        def tags(lst, css):
            return " ".join(f'<span class="tag {css}">{c}</span>' for c in lst if c in all_cols)
        for lbl, lst, css in [
            ("🪪 ID (exclude from model)", ID_COLS, "tag-id"),
            ("👤 Demographics", DEMO_COLS, "tag-num"),
            ("📊 Categorical", cat_cols, "tag-cat"),
            ("🔘 Binary flags", binary_cols, "tag-bin"),
            ("🩺 APACHE scores", apache_num[:20], "tag-num"),
            ("💊 Day-1 vitals", vital_d1[:30], "tag-num"),
            ("⏱️ Hour-1 vitals", vital_h1[:20], "tag-num"),
        ]:
            st.markdown(f"**{lbl}**")
            st.markdown(tags(lst, css), unsafe_allow_html=True)

    with st.expander("Data preview"):
        st.dataframe(df_raw.head(100), use_container_width=True)

    with st.expander("Column-level info"):
        info = [{"Column":c,"Dtype":str(df_raw[c].dtype),"Non-null":int(df_raw[c].notna().sum()),
                 "Null %":f"{100*df_raw[c].isna().mean():.1f}%","Unique":df_raw[c].nunique(),
                 "Example":str(df_raw[c].dropna().iloc[0]) if df_raw[c].notna().any() else "—"}
                for c in all_cols]
        st.dataframe(pd.DataFrame(info).set_index("Column"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 2 · MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════════
if "2 · Missing Values" in active:
    st.markdown('<p class="section-title">🕳️ 2 · Missing Values Analysis</p>', unsafe_allow_html=True)
    miss    = df_raw.isnull().sum()
    miss_pct_s = miss / len(df_raw) * 100
    miss_df = pd.DataFrame({"Missing N":miss,"Missing %":miss_pct_s})\
                .query("`Missing N` > 0").sort_values("Missing %", ascending=False)

    c1,c2,c3 = st.columns(3)
    c1.metric("Columns with missing", len(miss_df))
    c2.metric("Cols >50% missing", int((miss_pct_s>50).sum()))
    c3.metric("Complete rows", f"{int((~df_raw.isnull().any(axis=1)).sum()):,}")

    tab_bar, tab_corr, tab_group = st.tabs(["By Column","🔥 Missingness Correlation","By Group"])

    with tab_bar:
        fig, ax = plt.subplots(figsize=(8, max(4, len(miss_df)*0.22)))
        colors = [C_RED if p>50 else C_AMBER if p>20 else C_TEAL for p in miss_df["Missing %"]]
        ax.barh(miss_df.index[::-1], miss_df["Missing %"][::-1], color=colors[::-1], height=0.7)
        ax.set_xlabel("% Missing"); ax.set_title("Missing Values per Column")
        for v in [20,50,80]: ax.axvline(v, color=C_GRID, ls="--", lw=1)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_corr:
        st.markdown("**Do variables tend to be missing together?** High correlation = likely Missing Not At Random (MNAR).")
        miss_for_corr = miss_df[(miss_df["Missing %"]>5)&(miss_df["Missing %"]<95)].index.tolist()[:35]
        if len(miss_for_corr) >= 2:
            miss_ind  = df_raw[miss_for_corr].isnull().astype(int)
            corr_miss = miss_ind.corr()
            fig, ax = plt.subplots(figsize=(max(8,len(miss_for_corr)*0.55),
                                            max(7,len(miss_for_corr)*0.5)))
            mask = np.triu(np.ones_like(corr_miss, dtype=bool))
            sns.heatmap(corr_miss, mask=mask, cmap="RdBu_r", center=0, ax=ax,
                        linewidths=0.2, square=True, vmin=-1, vmax=1,
                        cbar_kws={"shrink":0.6,"label":"Correlation"})
            ax.set_title("Missingness Correlation\n(red = tend to be missing together — suggests MNAR)")
            plt.xticks(rotation=45, ha="right", fontsize=7); plt.yticks(fontsize=7)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('<div class="alert-amber">⚠️ Correlated missingness suggests tests were ordered (or not) '
                        'for clinical reasons — this is medically meaningful information, not random noise.</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Not enough partially-missing columns for this analysis.")

    with tab_group:
        groups = {
            "Hour-1 vitals (h1_)": [c for c in miss_df.index if c.startswith("h1_")],
            "Day-1 vitals (d1_)":  [c for c in miss_df.index if c.startswith("d1_")],
            "APACHE scores":       [c for c in miss_df.index if "apache" in c.lower()],
            "Demographics":        [c for c in miss_df.index if c in DEMO_COLS],
            "Other":               [c for c in miss_df.index if not c.startswith("h1_")
                                    and not c.startswith("d1_") and "apache" not in c.lower()
                                    and c not in DEMO_COLS],
        }
        rows = []
        for grp,gc in groups.items():
            gc_in = [c for c in gc if c in miss_df.index]
            if gc_in:
                avg = miss_df.loc[gc_in,"Missing %"].mean()
                rows.append({"Group":grp,"Cols with missing":len(gc_in),"Avg missing %":f"{avg:.1f}%",
                             "Recommendation":"Consider dropping" if avg>60
                             else "Impute + missingness flag" if avg>20 else "Simple imputation"})
        st.dataframe(pd.DataFrame(rows).set_index("Group"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 3 · CLASS BALANCE
# ══════════════════════════════════════════════════════════════════════════════
if "3 · Class Balance & Mortality Rate" in active:
    st.markdown('<p class="section-title">⚖️ 3 · Class Balance & Mortality Rate</p>', unsafe_allow_html=True)
    if tgt not in all_cols:
        st.info("Target column not found.")
    else:
        vc   = df_raw[tgt].value_counts().sort_index()
        mort = vc.get(1,0)/len(df_raw)*100
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Survived", f"{vc.get(0,0):,}")
        c2.metric("Died",     f"{vc.get(1,0):,}")
        c3.metric("Mortality rate", f"{mort:.1f}%")
        c4.metric("Imbalance ratio", f"1 : {int(vc.get(0,0)/max(vc.get(1,1),1))}")

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        bars = axes[0].bar(["Survived","Died"],[vc.get(0,0),vc.get(1,0)],
                           color=[C_GREEN,C_RED], alpha=0.85, width=0.5, edgecolor="white")
        for bar in bars:
            axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+300,
                         f"{bar.get_height():,}", ha="center", va="bottom", fontsize=9, fontweight="bold")
        axes[0].set_title("Class Distribution"); axes[0].set_ylabel("Patient Count")

        axes[1].pie([vc.get(0,0),vc.get(1,0)], labels=["Survived","Died"],
                    colors=[C_GREEN,C_RED], autopct="%1.1f%%", startangle=90,
                    wedgeprops={"edgecolor":"white","linewidth":2}, textprops={"fontsize":9})
        axes[1].set_title("Mortality Proportion")

        if "icu_type" in all_cols:
            icu_mort = df_raw.groupby("icu_type")[tgt].mean().sort_values()*100
            colors_i = [C_RED if v>mort else C_TEAL for v in icu_mort.values]
            axes[2].barh(icu_mort.index, icu_mort.values, color=colors_i, alpha=0.85)
            axes[2].axvline(mort, color="black", ls="--", lw=1.5, label=f"Overall {mort:.1f}%")
            axes[2].set_xlabel("Mortality %"); axes[2].set_title("Mortality by ICU Type")
            axes[2].legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        st.markdown('<div class="alert-red">🔴 <strong>Severe class imbalance (~8.6%)</strong>. '
                    'Raw accuracy is misleading — use <strong>AUROC, F1, Precision-Recall AUC</strong>. '
                    'Apply <code>class_weight="balanced"</code> or SMOTE on the training set only.</div>',
                    unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 4 · DEMOGRAPHICS & SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
if "4 · Demographics & Segmentation" in active:
    st.markdown('<p class="section-title">👤 4 · Demographics & Segmentation by Categorical Variables</p>',
                unsafe_allow_html=True)

    seg_tabs = st.tabs(["Gender","Ethnicity","ICU Type","Admit Source","Comorbidities"])

    def seg_plot(group_col, num_col):
        if group_col not in all_cols or num_col not in all_cols: return
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for v in df_raw[group_col].dropna().unique():
            data = df_raw[df_raw[group_col]==v][num_col].dropna()
            if len(data)>5: axes[0].hist(data, bins=30, alpha=0.5, density=True, label=str(v))
        axes[0].set_title(f"{num_col} by {group_col}"); axes[0].legend(fontsize=8)
        if tgt in all_cols:
            mort_by = df_raw.groupby(group_col)[tgt].mean().sort_values()*100
            overall = df_raw[tgt].mean()*100
            axes[1].barh(mort_by.index.astype(str)[::-1], mort_by.values[::-1],
                         color=[C_RED if v>overall else C_TEAL for v in mort_by.values[::-1]], alpha=0.85)
            axes[1].axvline(overall, color="black", ls="--", lw=1.5, label=f"Overall {overall:.1f}%")
            axes[1].set_xlabel("Mortality rate (%)"); axes[1].set_title(f"Mortality by {group_col}")
            axes[1].legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with seg_tabs[0]:
        seg_plot("gender","age")
        if tgt in all_cols and "gender" in all_cols:
            s = df_raw.groupby("gender").agg(
                N=("age","count"), Avg_age=("age","mean"), Mortality_pct=(tgt,"mean")).round(3)
            s["Mortality_pct"] = (s["Mortality_pct"]*100).round(2)
            st.dataframe(s, use_container_width=True)

    with seg_tabs[1]: seg_plot("ethnicity","age")
    with seg_tabs[2]:
        seg_plot("icu_type","age")
        if "icu_type" in all_cols:
            fig, ax = plt.subplots(figsize=(10,4))
            icu_vals = df_raw["icu_type"].dropna().unique()
            grps = [df_raw[df_raw["icu_type"]==v]["age"].dropna().values for v in icu_vals]
            ax.boxplot(grps, labels=icu_vals, patch_artist=True,
                       boxprops=dict(facecolor=C_TEAL, alpha=0.4),
                       flierprops=dict(marker=".", color=C_RED, markersize=3, alpha=0.3),
                       medianprops=dict(color=C_BLUE, lw=2))
            ax.set_title("Age Distribution by ICU Type"); ax.set_ylabel("Age")
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with seg_tabs[3]: seg_plot("hospital_admit_source","age")

    with seg_tabs[4]:
        comorbidities = [c for c in ["aids","cirrhosis","diabetes_mellitus","hepatic_failure",
                                     "immunosuppression","leukemia","lymphoma",
                                     "solid_tumor_with_metastasis"] if c in all_cols]
        if comorbidities and tgt in all_cols:
            overall_mort = df_raw[tgt].mean()*100
            rates = {}
            for c in comorbidities:
                has   = df_raw[df_raw[c]==1][tgt].mean()*100 if df_raw[c].eq(1).any() else 0
                hasnt = df_raw[df_raw[c]==0][tgt].mean()*100 if df_raw[c].eq(0).any() else 0
                rates[c.replace("_"," ").title()] = {"With condition":has,"Without condition":hasnt}
            rate_df = pd.DataFrame(rates).T
            fig, ax = plt.subplots(figsize=(10,5))
            x = np.arange(len(rate_df)); w=0.35
            ax.bar(x-w/2, rate_df["With condition"],    w, label="Has condition", color=C_RED,  alpha=0.8)
            ax.bar(x+w/2, rate_df["Without condition"], w, label="No condition",  color=C_TEAL, alpha=0.8)
            ax.axhline(overall_mort, color="black", ls="--", lw=1.2, label=f"Overall ({overall_mort:.1f}%)")
            ax.set_xticks(x); ax.set_xticklabels(rate_df.index, rotation=35, ha="right")
            ax.set_ylabel("Mortality rate (%)"); ax.set_title("Mortality Rate by Comorbidity")
            ax.legend(fontsize=9)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 5 · MORTALITY BY ICU UNIT
# ══════════════════════════════════════════════════════════════════════════════
if "5 · Mortality by ICU Unit" in active:
    st.markdown('<p class="section-title">🏨 5 · Mortality Rate by ICU Unit</p>', unsafe_allow_html=True)
    if tgt not in all_cols:
        st.info("Target column not found.")
    else:
        unit_col = st.selectbox("Group by",
                                [c for c in ["icu_type","icu_admit_source","apache_2_bodysystem",
                                             "apache_3j_bodysystem"] if c in all_cols], key="unit_sel")
        agg = df_raw.groupby(unit_col).agg(Patients=(tgt,"count"), Deaths=(tgt,"sum")).reset_index()
        agg["Mortality %"] = (agg["Deaths"]/agg["Patients"]*100).round(2)
        agg["Survival %"]  = (100-agg["Mortality %"]).round(2)
        agg = agg.sort_values("Mortality %", ascending=False)
        overall = df_raw[tgt].mean()*100

        fig, axes = plt.subplots(1, 2, figsize=(14, max(4,len(agg)*0.55)))
        bars = axes[0].barh(agg[unit_col].astype(str)[::-1], agg["Mortality %"][::-1],
                            color=[C_RED if v>overall else C_TEAL for v in agg["Mortality %"][::-1]], alpha=0.85)
        axes[0].axvline(overall, color="black", ls="--", lw=1.5, label=f"Overall ({overall:.1f}%)")
        axes[0].set_xlabel("Mortality (%)"); axes[0].set_title(f"Mortality % by {unit_col}")
        axes[0].legend(fontsize=8)
        for bar in bars:
            axes[0].text(bar.get_width()+0.2, bar.get_y()+bar.get_height()/2,
                         f"{bar.get_width():.1f}%", va="center", fontsize=8)
        axes[1].barh(agg[unit_col].astype(str)[::-1],
                     (agg["Patients"]-agg["Deaths"])[::-1], label="Survived", color=C_GREEN, alpha=0.75)
        axes[1].barh(agg[unit_col].astype(str)[::-1], agg["Deaths"][::-1],
                     left=(agg["Patients"]-agg["Deaths"])[::-1], label="Died", color=C_RED, alpha=0.75)
        axes[1].set_xlabel("Patient count"); axes[1].set_title(f"Absolute Counts by {unit_col}")
        axes[1].legend(fontsize=9)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
        st.dataframe(agg.set_index(unit_col).style.format({"Mortality %":"{:.2f}%","Survival %":"{:.2f}%"}),
                     use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 6 · SURGERY VS EMERGENCY PATIENTS
# ══════════════════════════════════════════════════════════════════════════════
if "6 · Surgery vs Emergency Patients" in active:
    st.markdown('<p class="section-title">🔪 6 · Surgery vs Emergency Patients</p>', unsafe_allow_html=True)

    def classify_admit(row):
        src    = str(row.get("hospital_admit_source","")).lower()
        elec   = row.get("elective_surgery", 0)
        post_op= row.get("apache_post_operative", 0)
        if elec==1 or post_op==1 or "operating room" in src or "recovery" in src or "pacu" in src:
            return "Post-Surgery"
        elif "emergency" in src or "accident" in src:
            return "Emergency"
        elif "floor" in src or "step-down" in src or "acute" in src:
            return "Medical Ward Transfer"
        else:
            return "Other"

    df_raw["patient_type"] = df_raw.apply(classify_admit, axis=1)
    pt_counts = df_raw["patient_type"].value_counts()
    overall_mort = df_raw[tgt].mean()*100 if tgt in all_cols else 0

    tab_ov, tab_cmp, tab_mort = st.tabs(["Overview","Clinical Comparison","Mortality Stats"])

    with tab_ov:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        colors_pt = [C_TEAL, C_RED, C_AMBER, C_BLUE]
        axes[0].bar(pt_counts.index, pt_counts.values, color=colors_pt[:len(pt_counts)], alpha=0.85)
        axes[0].set_title("Volume by Admission Type"); axes[0].set_ylabel("Count")
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=20, ha="right")
        if tgt in all_cols:
            mort_pt = df_raw.groupby("patient_type")[tgt].mean()*100
            axes[1].bar(mort_pt.index, mort_pt.values,
                        color=[C_RED if v>overall_mort else C_TEAL for v in mort_pt.values], alpha=0.85)
            axes[1].axhline(overall_mort, color="black", ls="--", lw=1.5, label=f"Overall ({overall_mort:.1f}%)")
            axes[1].set_title("Mortality Rate by Admission Type"); axes[1].set_ylabel("Mortality rate (%)")
            axes[1].legend(fontsize=8)
            plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=20, ha="right")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_cmp:
        compare_cols = [c for c in ["age","bmi","pre_icu_los_days"]+apache_prob+
                        ["d1_heartrate_max","d1_sysbp_min","d1_resprate_max"] if c in all_cols]
        chosen_c = st.multiselect("Variables to compare", compare_cols,
                                  default=compare_cols[:4], key="surg_cmp")
        if chosen_c:
            pt_types = [p for p in ["Post-Surgery","Emergency","Medical Ward Transfer"]
                        if p in df_raw["patient_type"].values]
            ncols_g=2; nrows_g=max(1,(len(chosen_c)+1)//ncols_g)
            fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(13, 4.5*nrows_g), squeeze=False)
            flat = axes.flatten()
            for i, col in enumerate(chosen_c):
                ax = flat[i]
                for pt, color in zip(pt_types, [C_TEAL, C_RED, C_AMBER]):
                    data = df_raw[df_raw["patient_type"]==pt][col].dropna()
                    if len(data)>5: ax.hist(data, bins=25, alpha=0.5, density=True, label=pt, color=color)
                ax.set_title(col); ax.legend(fontsize=7)
            for j in range(len(chosen_c),len(flat)): flat[j].set_visible(False)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_mort:
        if tgt in all_cols:
            summ = df_raw.groupby("patient_type").agg(
                N=(tgt,"count"), Deaths=(tgt,"sum"),
                Mortality_pct=(tgt,"mean"), Avg_age=("age","mean")).round(3)
            summ["Mortality_pct"] = (summ["Mortality_pct"]*100).round(2)
            st.dataframe(summ.style.format({"Mortality_pct":"{:.2f}%","Avg_age":"{:.1f}"}),
                         use_container_width=True)
            surg  = df_raw[df_raw["patient_type"]=="Post-Surgery"][tgt].dropna()
            emerg = df_raw[df_raw["patient_type"]=="Emergency"][tgt].dropna()
            if len(surg)>10 and len(emerg)>10:
                ct = pd.crosstab(
                    df_raw[df_raw["patient_type"].isin(["Post-Surgery","Emergency"])]["patient_type"],
                    df_raw[df_raw["patient_type"].isin(["Post-Surgery","Emergency"])][tgt])
                _, p = stats.chi2_contingency(ct)[:2]
                msg = f"Chi-square (Surgery vs Emergency): p = {p:.3e}"
                css = "alert-red" if p<0.05 else "alert-green"
                note = "statistically significant difference" if p<0.05 else "no significant difference"
                st.markdown(f'<div class="{css}">📊 {msg} — <strong>{note}</strong></div>',
                            unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 7 · VITAL SIGNS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if "7 · Vital Signs Analysis" in active:
    st.markdown('<p class="section-title">💓 7 · Vital Signs Analysis</p>', unsafe_allow_html=True)
    tab_d1, tab_h1, tab_vs_m = st.tabs(["Day-1 Vitals","Hour-1 Vitals","Vitals vs Mortality"])

    def plot_vital_grid(cols_list, color, key_sfx):
        if not cols_list: st.info("No columns."); return
        chosen = st.multiselect("Select columns", cols_list,
                                default=cols_list[:min(6,len(cols_list))], key=f"vit_{key_sfx}")
        if not chosen: return
        ncols_g=3; nrows_g=max(1,(len(chosen)+2)//ncols_g)
        fig, axes = plt.subplots(nrows_g, ncols_g, figsize=(15,4*nrows_g), squeeze=False)
        flat = axes.flatten()
        for i,col in enumerate(chosen):
            ax=flat[i]; data=df[col].dropna()
            if len(data)<2: ax.set_visible(False); continue
            ax.hist(data, bins=30, color=color, alpha=0.7, density=True, edgecolor="white")
            if len(data)>5:
                from scipy.stats import gaussian_kde
                xs=np.linspace(data.min(),data.max(),200)
                ax.plot(xs, gaussian_kde(data)(xs), color=C_RED, lw=1.5)
            ax.set_title(f"{col}\nmissing={100*df_raw[col].isna().mean():.0f}%", fontsize=8)
        for j in range(len(chosen),len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

    with tab_d1: plot_vital_grid(vital_d1, C_TEAL, "d1")
    with tab_h1: plot_vital_grid(vital_h1, C_BLUE, "h1")
    with tab_vs_m:
        if tgt in all_cols and vital_d1:
            col_v = st.selectbox("Vital sign", vital_d1, key="vit_mort_sel")
            fig, axes = plt.subplots(1, 2, figsize=(13,4))
            alive = df_raw[df_raw[tgt]==0][col_v].dropna()
            dead  = df_raw[df_raw[tgt]==1][col_v].dropna()
            axes[0].hist(alive, bins=30, alpha=0.6, color=C_GREEN, density=True, label=f"Survived (n={len(alive):,})")
            axes[0].hist(dead,  bins=30, alpha=0.6, color=C_RED,   density=True, label=f"Died (n={len(dead):,})")
            axes[0].legend(); axes[0].set_title(f"{col_v} by Outcome")
            if len(alive)>2 and len(dead)>2:
                _, p = stats.mannwhitneyu(alive, dead, alternative="two-sided")
                axes[0].set_xlabel(f"Mann-Whitney U  p={p:.3e}")
            bp = axes[1].boxplot([alive,dead], labels=["Survived","Died"], patch_artist=True,
                                 medianprops=dict(color=C_BLUE,lw=2),
                                 flierprops=dict(marker=".",markersize=2,alpha=0.2))
            bp["boxes"][0].set_facecolor(C_GREEN); bp["boxes"][0].set_alpha(0.4)
            bp["boxes"][1].set_facecolor(C_RED);   bp["boxes"][1].set_alpha(0.4)
            axes[1].set_title(f"{col_v} — Box Plot by Outcome")
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 8 · AGE VS APACHE SCORE
# ══════════════════════════════════════════════════════════════════════════════
if "8 · Age vs APACHE Score" in active:
    st.markdown('<p class="section-title">📍 8 · Age vs APACHE Score — Colored by Outcome</p>', unsafe_allow_html=True)
    ap_cols = [c for c in apache_prob+["apache_2_diagnosis","apache_3j_diagnosis"] if c in all_cols]
    if "age" not in all_cols or not ap_cols:
        st.info("Age or APACHE columns not found.")
    else:
        y_col    = st.selectbox("APACHE metric (Y axis)", ap_cols, key="ap_scatter")
        plot_df  = df_raw[["age",y_col,tgt]].dropna().sample(min(5000,len(df_raw)), random_state=42)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if tgt in all_cols:
            survived = plot_df[plot_df[tgt]==0]
            died     = plot_df[plot_df[tgt]==1]
            axes[0].scatter(survived["age"], survived[y_col], c=C_GREEN, alpha=0.2, s=12,
                            label=f"Survived (n={len(survived):,})", rasterized=True)
            axes[0].scatter(died["age"],     died[y_col],     c=C_RED,   alpha=0.45, s=16,
                            label=f"Died (n={len(died):,})",     rasterized=True)
            for grp, color in [(survived,C_GREEN),(died,C_RED)]:
                clean = grp[["age",y_col]].dropna()
                if len(clean)>10:
                    m,b = np.polyfit(clean["age"],clean[y_col],1)
                    xs  = np.linspace(clean["age"].min(),clean["age"].max(),100)
                    axes[0].plot(xs, m*xs+b, color=color, lw=2, ls="--", alpha=0.8)
            axes[0].legend(fontsize=9, markerscale=2)
        axes[0].set_xlabel("Age (years)"); axes[0].set_ylabel(y_col.replace("_"," ").title())
        axes[0].set_title("Age vs APACHE Score\n(colored by outcome, dashed = trend lines)")

        axes[1].hexbin(plot_df["age"], plot_df[y_col], gridsize=30, cmap="YlOrRd", mincnt=1, linewidths=0.1)
        plt.colorbar(axes[1].collections[0], ax=axes[1], label="Patient count")
        axes[1].set_xlabel("Age (years)"); axes[1].set_ylabel(y_col.replace("_"," ").title())
        axes[1].set_title("Age vs APACHE Score\n(density hexbin)")
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        r, p = stats.pearsonr(plot_df["age"], plot_df[y_col])
        css  = "alert-red" if abs(r)>0.3 else "alert-amber" if abs(r)>0.1 else "alert-green"
        st.markdown(f'<div class="{css}">📊 Pearson r = {r:.3f}, p = {p:.3e}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# 9 · VITAL SIGNS VS MORTALITY HEATMAP
# ══════════════════════════════════════════════════════════════════════════════
if "9 · Vital Signs vs Mortality (Heatmap)" in active:
    st.markdown('<p class="section-title">🌡️ 9 · Vital Signs vs Mortality — Correlation Heatmap</p>',
                unsafe_allow_html=True)
    if tgt not in all_cols or not vital_d1:
        st.info("Target or vital sign columns not found.")
    else:
        corrs = []
        for col in vital_d1:
            data = df_raw[[col,tgt]].dropna()
            if len(data)<50: continue
            r, p = stats.pointbiserialr(data[tgt], data[col])
            corrs.append({"Feature":col,"r":r,"p":p,"Significant":"✅" if p<0.05 else "❌"})

        if corrs:
            corr_df = pd.DataFrame(corrs).sort_values("r", key=abs, ascending=False)
            tab_hm, tab_bar, tab_tbl = st.tabs(["Heatmap","Bar Chart","Table"])

            with tab_hm:
                top_vitals = corr_df.head(30)["Feature"].tolist()
                vital_mort = corr_df.set_index("Feature")["r"].loc[top_vitals].to_frame().T
                vital_mort.index = ["Mortality correlation"]
                fig, ax = plt.subplots(figsize=(max(10,len(top_vitals)*0.55), 2.5))
                sns.heatmap(vital_mort, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
                            linewidths=0.3, vmin=-0.5, vmax=0.5, annot_kws={"size":7},
                            cbar_kws={"shrink":0.5,"label":"Correlation"})
                ax.set_title("Day-1 Vital Signs — Correlation with Hospital Mortality\n"
                             "(red = higher value → more deaths | blue = higher value → survival)")
                ax.set_yticklabels(["Mortality correlation"], rotation=0)
                plt.xticks(rotation=45, ha="right", fontsize=7)
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            with tab_bar:
                top_n = st.slider("Top N", 10, min(40,len(corr_df)), 20, key="vmort_n")
                top   = corr_df.head(top_n)
                fig, ax = plt.subplots(figsize=(8, top_n*0.4))
                ax.barh(top["Feature"][::-1], top["r"][::-1],
                        color=[C_RED if v>0 else C_TEAL for v in top["r"][::-1]], alpha=0.85)
                ax.axvline(0, color="black", lw=1)
                ax.set_xlabel("Point-biserial correlation with mortality")
                ax.set_title("Vital Signs Correlation with Mortality")
                plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

            with tab_tbl:
                st.dataframe(corr_df.rename(columns={"r":"Correlation","p":"p-value"})
                             .style.format({"Correlation":"{:.4f}","p-value":"{:.3e}"}),
                             use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# 10 · CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
if "10 · Correlation Analysis" in active:
    st.markdown('<p class="section-title">🕸️ 10 · Correlation Analysis</p>', unsafe_allow_html=True)
    corr_opts = {
        "Demographics":         [c for c in DEMO_COLS if c in num_cols],
        "APACHE scores":        [c for c in apache_num if c in all_cols][:20],
        "Day-1 vitals":         vital_d1[:20],
        "Hour-1 vitals":        vital_h1[:20],
        "All numeric (top 30)": num_cols[:30],
    }
    grp  = st.selectbox("Column group", list(corr_opts.keys()), key="corr_grp")
    c4c  = [c for c in corr_opts[grp] if c in all_cols]
    meth = st.radio("Method", ["pearson","spearman"], horizontal=True, key="corr_meth")

    if len(c4c) >= 2:
        corr = df[c4c].corr(method=meth)
        fig, ax = plt.subplots(figsize=(max(7,len(c4c)*0.75), max(6,len(c4c)*0.7)))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=len(c4c)<=18, fmt=".2f", cmap=palette,
                    center=0, ax=ax, linewidths=0.3, square=True, annot_kws={"size":7},
                    vmin=-1, vmax=1, cbar_kws={"shrink":0.6,"label":"Correlation"})
        ax.set_title(f"{meth.capitalize()} Correlation — {grp}")
        plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        high = (corr.where(~mask).abs().stack()
                .reset_index().rename(columns={"level_0":"A","level_1":"B",0:"r"})
                .query("r > 0.85").sort_values("r", ascending=False))
        if not high.empty:
            st.markdown(f'<div class="alert-amber">⚠️ {len(high)} pairs with |r|>0.85 — '
                        f'multicollinearity risk. XGBoost/Random Forest handle this automatically.</div>',
                        unsafe_allow_html=True)
            with st.expander("View pairs"):
                st.dataframe(high.style.format({"r":"{:.3f}"}), use_container_width=True)

        with st.expander("Scatter plot — explore a pair"):
            ca = st.selectbox("X", c4c, key="pair_x")
            cb = st.selectbox("Y", c4c, index=min(1,len(c4c)-1), key="pair_y")
            hp = st.selectbox("Color by", ["None"]+cat_cols+binary_cols, key="pair_hue")
            fig, ax = plt.subplots(figsize=(6,4))
            if hp != "None":
                for v in df[hp].dropna().unique():
                    g = df[df[hp]==v]
                    ax.scatter(g[ca], g[cb], alpha=0.3, s=12, label=str(v))
                ax.legend(title=hp, fontsize=7, markerscale=2)
            else:
                ax.scatter(df[ca], df[cb], alpha=0.3, s=12, color=C_TEAL)
            clean = df[[ca,cb]].dropna()
            if len(clean)>5:
                m,b = np.polyfit(clean[ca],clean[cb],1)
                xs  = np.linspace(clean[ca].min(),clean[ca].max(),100)
                ax.plot(xs, m*xs+b, color=C_RED, lw=1.5, ls="--")
                r,p = stats.pearsonr(clean[ca],clean[cb])
                ax.set_title(f"{ca} vs {cb}  |  r={r:.3f}, p={p:.3e}")
            ax.set_xlabel(ca); ax.set_ylabel(cb)
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 11 · APACHE SCORES
# ══════════════════════════════════════════════════════════════════════════════
if "11 · APACHE Scores" in active:
    st.markdown('<p class="section-title">📈 11 · APACHE Scores</p>', unsafe_allow_html=True)
    if apache_prob and tgt in all_cols:
        fig, axes = plt.subplots(1, len(apache_prob), figsize=(6*len(apache_prob), 4))
        if len(apache_prob)==1: axes=[axes]
        for ax, pc in zip(axes, apache_prob):
            alive=df[df[tgt]==0][pc].dropna(); dead=df[df[tgt]==1][pc].dropna()
            ax.hist(alive,bins=30,alpha=0.6,color=C_GREEN,density=True,label="Survived")
            ax.hist(dead, bins=30,alpha=0.6,color=C_RED,  density=True,label="Died")
            ax.axvline(0.5,color="black",ls="--",lw=1.5,label="Threshold 0.5")
            ax.set_title(pc.replace("_"," ").title()); ax.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        try:
            from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for pc in apache_prob:
                valid=df[[pc,tgt]].dropna()
                if len(valid)<100: continue
                auc=roc_auc_score(valid[tgt],valid[pc])
                fpr,tpr,_=roc_curve(valid[tgt],valid[pc])
                axes[0].plot(fpr,tpr,lw=2,label=f"{pc.replace('apache_4a_','').replace('_prob','')} AUC={auc:.3f}")
                pr,rc,_=precision_recall_curve(valid[tgt],valid[pc])
                ap=average_precision_score(valid[tgt],valid[pc])
                axes[1].plot(rc,pr,lw=2,label=f"AP={ap:.3f}")
            axes[0].plot([0,1],[0,1],"k--",lw=1); axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
            axes[0].set_title("ROC Curve — APACHE Baseline"); axes[0].legend()
            axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall — APACHE Baseline"); axes[1].legend()
            plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()
            st.markdown('<div class="alert-amber">💡 <strong>Baseline to beat:</strong> APACHE-4a AUROC ≈ 0.85 — '
                        'your ML model should aim to match or exceed this.</div>', unsafe_allow_html=True)
        except ImportError:
            st.warning("Install scikit-learn for ROC curves.")


# ══════════════════════════════════════════════════════════════════════════════
# 12 · OUTLIER DETECTION
# ══════════════════════════════════════════════════════════════════════════════
if "12 · Outlier Detection" in active:
    st.markdown('<p class="section-title">🚨 12 · Outlier Detection</p>', unsafe_allow_html=True)
    out_grp = st.selectbox("Group", ["Demographics","Day-1 vitals","APACHE scores"], key="out_grp")
    grp_map = {"Demographics": [c for c in DEMO_COLS if c in num_cols],
               "Day-1 vitals": vital_d1[:12],
               "APACHE scores":[c for c in apache_num if c in all_cols][:12]}
    o_cols  = [c for c in grp_map[out_grp] if c in all_cols]
    o_meth  = st.radio("Method", ["IQR (1.5×)","Z-score (|z|>3)"], horizontal=True, key="out_m")
    if o_cols:
        rows=[]
        for col in o_cols:
            s=df_raw[col].dropna()
            if len(s)<5: continue
            mask_o = ((s<s.quantile(.25)-1.5*(s.quantile(.75)-s.quantile(.25)))|(s>s.quantile(.75)+1.5*(s.quantile(.75)-s.quantile(.25)))) \
                     if o_meth=="IQR (1.5×)" else ((s-s.mean())/s.std()).abs()>3
            rows.append({"Column":col,"Outliers":int(mask_o.sum()),"% outlier":f"{100*mask_o.sum()/len(s):.2f}%",
                         "Min":f"{s.min():.2f}","Max":f"{s.max():.2f}",
                         "Action":"⚠️ Review" if mask_o.sum()/len(s)>0.05 else "✅ OK"})
        if rows: st.dataframe(pd.DataFrame(rows).set_index("Column"), use_container_width=True)
        ncols_g=3; nrows_g=max(1,(len(o_cols)+2)//ncols_g)
        fig,axes=plt.subplots(nrows_g,ncols_g,figsize=(15,4*nrows_g),squeeze=False)
        flat=axes.flatten()
        for i,col in enumerate(o_cols):
            ax=flat[i]; data=df[col].dropna()
            if len(data)<2: ax.set_visible(False); continue
            bp=ax.boxplot(data,patch_artist=True,boxprops=dict(facecolor=C_TEAL,alpha=0.4),
                          flierprops=dict(marker="o",color=C_RED,markersize=3,alpha=0.35),
                          medianprops=dict(color=C_BLUE,lw=2))
            ax.set_title(col,fontsize=8)
        for j in range(len(o_cols),len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 13 · NORMALITY & DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
if "13 · Normality & Distributions" in active:
    st.markdown('<p class="section-title">🔔 13 · Normality & Distributions</p>', unsafe_allow_html=True)
    norm_grp = st.selectbox("Group", ["Demographics","Day-1 vitals","APACHE scores"], key="norm_grp")
    grp_map2 = {"Demographics": [c for c in DEMO_COLS if c in num_cols],
                "Day-1 vitals": vital_d1[:9],
                "APACHE scores":[c for c in apache_num if c in all_cols][:9]}
    norm_cols = [c for c in grp_map2[norm_grp] if c in all_cols]
    if norm_cols:
        results=[]
        for col in norm_cols:
            data=df_raw[col].dropna(); s=data.sample(min(5000,len(data)),random_state=42)
            if len(s)<8: continue
            _,p=stats.normaltest(s)
            results.append({"Column":col,"N":len(data),"Skewness":f"{data.skew():.3f}",
                            "Kurtosis":f"{data.kurt():.3f}","Normal?":"✅" if p>0.05 else "❌","p-value":f"{p:.3e}"})
        if results: st.dataframe(pd.DataFrame(results).set_index("Column"), use_container_width=True)
        ncols_g=3; nrows_g=max(1,(len(norm_cols)+2)//ncols_g)
        fig,axes=plt.subplots(nrows_g,ncols_g,figsize=(15,4*nrows_g),squeeze=False)
        flat=axes.flatten()
        for i,col in enumerate(norm_cols):
            ax=flat[i]; data=df_raw[col].dropna(); s=data.sample(min(2000,len(data)),random_state=42)
            if len(s)<8: ax.set_visible(False); continue
            stats.probplot(s,dist="norm",plot=ax)
            ax.set_title(f"Q-Q: {col}",fontsize=9)
            ax.get_lines()[0].set(markersize=2,alpha=0.3,color=C_TEAL)
            ax.get_lines()[1].set(color=C_RED,lw=1.5)
        for j in range(len(norm_cols),len(flat)): flat[j].set_visible(False)
        plt.tight_layout(); st.pyplot(fig,use_container_width=True); plt.close()
        st.markdown('<div class="alert-amber">💡 Most clinical variables are <strong>right-skewed</strong>. '
                    'Apply <strong>log transform</strong> before Logistic Regression. '
                    'XGBoost and Random Forest are invariant to distribution shape.</div>',
                    unsafe_allow_html=True)

st.markdown("---")
st.markdown(f"<div style='color:#7F8C8D;font-size:.8rem;text-align:center;'>"
            f"ICU Mortality EDA Explorer · Healthcare Predictive Modeling Project · "
            f"Streamlit · Pandas · Seaborn · Scikit-learn · SciPy</div>", unsafe_allow_html=True)

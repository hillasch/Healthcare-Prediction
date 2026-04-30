import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency, gaussian_kde
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ICU EDA", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family:'Fraunces',Georgia,serif; background:#f7f4ef; color:#1a1a1a; }
[data-testid="stSidebar"] { background:#1c1c1c; }
[data-testid="stSidebar"] * { color:#e8e0d0 !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e0d8cc; border-radius:6px; padding:.8rem; }
.sec-label { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.15em; color:#8a7a6a; text-transform:uppercase; margin-bottom:.2rem; }
.sec-title { font-family:'Fraunces',serif; font-weight:900; font-size:1.9rem; color:#1a1a1a; margin:0; line-height:1.1; }
.sec-sub { color:#6a5a4a; font-size:.9rem; margin-bottom:1.2rem; font-style:italic; }
.insight { background:#fff8f0; border-left:4px solid #c84b2f; border-radius:0 6px 6px 0; padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; color:#2a1a0a; }
.warn { background:#fffff0; border-left:4px solid #c8a02f; border-radius:0 6px 6px 0; padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; }
.concl { background:#1c1c1c; color:#e8e0d0; border-radius:8px; padding:1rem 1.3rem; margin-top:1rem; font-size:.86rem; line-height:1.7; }
.concl h4 { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; color:#e07a5f; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

TARGET = "hospital_death"
C1, C2 = "#2f6bc8", "#c84b2f"
PT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(247,244,239,.5)", font_family="Fraunces", font_color="#1a1a1a")

# ── helpers ────────────────────────────────────────────────────────────────────
def sh(label, title, sub=""):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)
    if sub: st.markdown(f'<p class="sec-sub">{sub}</p>', unsafe_allow_html=True)

def ins(t): st.markdown(f'<div class="insight">💡 {t}</div>', unsafe_allow_html=True)
def wrn(t): st.markdown(f'<div class="warn">⚠️ {t}</div>', unsafe_allow_html=True)
def divider(): st.markdown("<hr style='border:none;border-top:1px solid #e0d8cc;margin:1.5rem 0'>", unsafe_allow_html=True)

def concl(title, bullets):
    li = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f'<div class="concl"><h4>CONCLUSION — {title.upper()}</h4><ul>{li}</ul></div>', unsafe_allow_html=True)

def safe_float(x):
    try: return float(x)
    except: return 0.0

def is_binary(s):
    v = s.dropna().unique()
    return set(v).issubset({0, 1}) and len(v) == 2

def kde_curve(data, n=200):
    data = np.array(data, dtype=float)
    data = data[np.isfinite(data)]
    if len(data) < 5:
        return [], []
    try:
        kde = gaussian_kde(data)
        x = np.linspace(data.min(), data.max(), n)
        return x.tolist(), kde(x).tolist()
    except:
        return [], []

# ── cached computations ────────────────────────────────────────────────────────
@st.cache_data
def split_cols(df):
    num = df.select_dtypes(include=np.number).columns.drop(TARGET, errors='ignore').tolist()
    cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    bn = [c for c in num if is_binary(df[c])]
    tn = [c for c in num if c not in bn]
    return num, cat, bn, tn

@st.cache_data
def get_miss(df):
    m = pd.DataFrame({'missing': df.isnull().sum(), 'pct': (df.isnull().mean() * 100).round(2)})
    return m[m['missing'] > 0].sort_values('pct', ascending=False)

@st.cache_data
def get_desc(df, cols):
    d = df[cols].describe().T.copy()
    d['skew'] = df[cols].skew()
    d['kurtosis'] = df[cols].kurtosis()
    d['missing%'] = (df[cols].isnull().mean() * 100).round(2)
    return d

@st.cache_data
def get_outlier_counts(df, cols):
    res = {}
    for c in cols:
        s = df[c].dropna()
        if len(s) == 0: res[c] = 0; continue
        Q1, Q3 = s.quantile(.25), s.quantile(.75); IQR = Q3 - Q1
        res[c] = int(((s < Q1 - 1.5*IQR) | (s > Q3 + 1.5*IQR)).sum())
    return pd.Series(res).sort_values(ascending=False)

@st.cache_data
def get_group_means(df, cols):
    valid = [c for c in cols if c in df.columns]
    gm = df[valid + [TARGET]].groupby(TARGET).mean().T
    gm.columns = ['Survived (0)', 'Died (1)']
    gm['abs_diff'] = (gm['Died (1)'] - gm['Survived (0)']).abs()
    gm['rel_diff%'] = ((gm['Died (1)'] - gm['Survived (0)']) / gm['Survived (0)'].replace(0, np.nan) * 100).round(1)
    return gm.sort_values('abs_diff', ascending=False)

@st.cache_data
def get_corr_target(df, tn, bn, cat):
    cols = [c for c in tn + bn if c in df.columns]
    pearson = (df[cols].corrwith(df[TARGET]).abs()
               .reset_index().rename(columns={'index': 'feature', 0: 'abs_corr'}))
    pearson['type'] = pearson['feature'].apply(lambda x: 'binary' if x in bn else 'numeric')
    rows = []
    for col in cat:
        try:
            clean = df[[col, TARGET]].dropna()
            ct = pd.crosstab(clean[col], clean[TARGET])
            chi2 = chi2_contingency(ct)[0]; n = len(clean)
            v = float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))
            rows.append({'feature': col, 'abs_corr': round(v, 4), 'type': 'categorical'})
        except: pass
    result = pd.concat([pearson, pd.DataFrame(rows)], ignore_index=True)
    result['abs_corr'] = pd.to_numeric(result['abs_corr'], errors='coerce')
    return result.sort_values('abs_corr', ascending=False).reset_index(drop=True)

# ── sidebar ────────────────────────────────────────────────────────────────────
PAGES = ["① Upload & Overview","② Missing Values","③ Numeric Distributions",
         "④ Outliers","⑤ Categorical Columns","⑥ Feature vs Target","⑦ Correlations"]

with st.sidebar:
    st.markdown("## 🏥 ICU EDA")
    st.markdown("*Hospital Death Prediction*")
    st.divider()
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Upload your CSV on step ① to unlock all pages.")

# ══════════════════════════════════════════════════════════════════════════════
# ① UPLOAD & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    sh("STEP 01","Upload & Overview","Load the dataset — shape, types, duplicates, target balance.")

    f = st.file_uploader("", type=["csv","xlsx"])
    if f:
        try:
            df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
            df = df.drop(columns=['patient_id','encounter_id'], errors='ignore')
            st.session_state['df'] = df
            st.success(f"✅ Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns**")
        except Exception as e:
            st.error(f"Error loading file: {e}"); st.stop()
    elif 'df' not in st.session_state:
        st.info("Upload your CSV to begin."); st.stop()

    df = st.session_state['df']
    num_cols, cat_cols, binary_cols, true_num_cols = split_cols(df)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Rows", f"{df.shape[0]:,}")
    c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(true_num_cols))
    c4.metric("Binary (0/1)", len(binary_cols))
    c5.metric("Categorical", len(cat_cols))

    divider()

    dups = int(df.duplicated().sum())
    ins("No duplicate rows.") if dups == 0 else wrn(f"{dups:,} duplicate rows found.")

    # ── target balance ────────────────────────────────────────────────────────
    st.subheader("Target balance — `hospital_death`")

    if TARGET not in df.columns:
        wrn(f"Column `{TARGET}` not found in dataset.")
    else:
        vc = df[TARGET].value_counts().sort_index()
        pct = (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2)

        survived_pct = safe_float(pct.get(0, 0))
        died_pct = safe_float(pct.get(1, 0))

        col_a, col_b = st.columns([1, 2])
        with col_a:
            st.dataframe(
                pd.DataFrame({'count': vc, 'percent %': pct}),
                use_container_width=True
            )
        with col_b:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=['Survived (0)', 'Died (1)'],
                y=[survived_pct, died_pct],
                marker_color=[C1, C2],
                text=[f"{survived_pct:.1f}%", f"{died_pct:.1f}%"],
                textposition='outside',
                width=0.5
            ))
            fig.update_layout(**PT, height=280, showlegend=False,
                              yaxis=dict(title='% of patients', range=[0, max(survived_pct, died_pct)*1.2]),
                              xaxis=dict(title=''))
            st.plotly_chart(fig, use_container_width=True)

        wrn(f"Target is imbalanced — only {died_pct:.1f}% deaths. Consider class weights or SMOTE when modelling.")

    divider()

    # ── column info ───────────────────────────────────────────────────────────
    st.subheader("Column info")
    info_rows = []
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        null_pct = round(df[col].isnull().mean() * 100, 2)
        info_rows.append({
            'column': col,
            'dtype': str(df[col].dtype),
            'nulls': null_count,
            'null %': null_pct,
            'unique': int(df[col].nunique()),
        })
    info_df = pd.DataFrame(info_rows)
    st.dataframe(info_df, use_container_width=True, height=400)

    dtype_counts = info_df['dtype'].value_counts().reset_index()
    dtype_counts.columns = ['dtype', 'count']
    fig2 = go.Figure(go.Pie(
        labels=dtype_counts['dtype'].tolist(),
        values=dtype_counts['count'].tolist(),
        hole=0.55,
        marker_colors=[C1, C2, '#5a9ad4', '#c8a02f', '#888'],
        textinfo='label+value'
    ))
    fig2.update_layout(**PT, height=260, showlegend=True,
                       title=dict(text='Column dtype breakdown', font_size=13))
    st.plotly_chart(fig2, use_container_width=True)

    concl("Data Overview",[
        "The dataset includes 91,713 ICU patient records and 184 columns after removing ID fields.",
        "The prediction target is hospital_death, representing in-hospital mortality.",
        "The target is highly imbalanced: 8.6% deaths vs. 91.4% survivors.",
        "Most input features are numeric, with additional binary and categorical variables.",
        "No duplicate patient records were found.",
        "Modeling implication: evaluation should focus on clinically relevant metrics such as recall, precision, ROC-AUC, and threshold-based decision support rather than accuracy alone.",
    ])

# ── guard ──────────────────────────────────────────────────────────────────────
elif 'df' not in st.session_state:
    st.warning("👈 Go to **① Upload & Overview** first and upload your CSV."); st.stop()
else:
    df = st.session_state['df']
    num_cols, cat_cols, binary_cols, true_num_cols = split_cols(df)
    miss = get_miss(df)

# ══════════════════════════════════════════════════════════════════════════
# ② MISSING VALUES
# ══════════════════════════════════════════════════════════════════════════
    if page == PAGES[1]:
        sh("STEP 02","Missing Values","Assessing where data is missing, whether missingness is clustered, and how it may affect mortality prediction.")

        sparse_pct = float((df.isnull().sum(axis=1) > df.shape[1]*.5).mean()*100)
        c1,c2,c3 = st.columns(3)
        c1.metric("Columns with missing", len(miss))
        c2.metric("Columns >50% missing", int((miss['pct']>50).sum()))
        c3.metric("Rows >50% missing", f"{sparse_pct:.1f}%")

        divider()

        bins_=[0,5,20,50,80,100]; labels_=['<5%','5–20%','20–50%','50–80%','>80%']
        miss2 = miss.copy()
        miss2['band'] = pd.cut(miss2['pct'], bins=bins_, labels=labels_)
        bc = miss2['band'].value_counts().sort_index().reset_index()
        bc.columns = ['band','count']
        fig = go.Figure(go.Bar(
            x=bc['band'].astype(str).tolist(),
            y=[int(v) for v in bc['count'].tolist()],
            marker_color=[C1,'#5a9ad4','#c8a02f','#d4752f',C2],
            text=[str(int(v)) for v in bc['count'].tolist()],
            textposition='outside'
        ))
        fig.update_layout(**PT, height=280, showlegend=False,
                          xaxis_title='Missing % band', yaxis_title='Number of columns',
                          yaxis=dict(range=[0, int(bc['count'].max())*1.25]))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
<div style="background:#f0f7ff; border-left:4px solid #2f6bc8; border-radius:0 6px 6px 0; padding:.8rem 1rem; margin:.8rem 0; font-size:.87rem; color:#1a1a1a;">
<strong>💬 Why this matters:</strong><br>
Missing values are common in real-world EHR data. They may reflect clinical workflow, such as whether a specific lab test, blood gas measurement, or monitoring procedure was performed. Therefore, missingness should be treated as part of the modeling problem rather than ignored.
</div>
""", unsafe_allow_html=True)

        divider()

        st.subheader("Top 20 columns by missing %")
        st.caption("Dashed reference lines mark 20% and 50% missingness thresholds.")
        top40 = miss.head(20).copy()
        pct_vals = [float(v) for v in top40['pct'].tolist()]
        col_names = top40.index.tolist()
        colors = [C2 if p > 50 else C1 for p in pct_vals]
        fig2 = go.Figure(go.Bar(
            x=pct_vals[::-1], y=col_names[::-1],
            orientation='h', marker_color=colors[::-1]
        ))
        fig2.add_vline(x=50, line_dash='dash', line_color='red', annotation_text='50%')
        fig2.add_vline(x=20, line_dash='dash', line_color='orange', annotation_text='20%')
        fig2.update_layout(**PT, height=max(400, len(top40)*22),
                           xaxis_title='Missing %', yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

        divider()

        st.subheader("Rows with >50% Missing Values")
        st.markdown(
            "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
            "A <em>highly missing row</em> is a patient record where more than 50% of the feature values are missing. "
            "This analysis checks whether these rows have a different mortality rate compared with the rest of the dataset."
            "</p>", unsafe_allow_html=True
        )
        sparse_mask = df.isnull().sum(axis=1) > df.shape[1]*.5
        dr_sp = float(df[sparse_mask][TARGET].mean()*100)
        dr_nm = float(df[~sparse_mask][TARGET].mean()*100)
        c1,c2,c3 = st.columns(3)
        c1.metric("High-missingness rows", f"{int(sparse_mask.sum()):,}")
        c2.metric("Death rate — high missing", f"{dr_sp:.1f}%")
        c3.metric("Death rate — other rows", f"{dr_nm:.1f}%")

        if abs(dr_sp - dr_nm) < 3:
            ins("Rows with high missingness have a similar mortality rate to other rows, suggesting that row-level missingness is not strongly associated with the target. However, missing values still require preprocessing before modeling.")
        else:
            wrn(f"Rows with high missingness die at {dr_sp:.1f}% vs {dr_nm:.1f}% for other rows — consider adding a missingness indicator as a feature.")

        divider()

        st.subheader("Co-missing column pairs (missingness correlation)")
        high_miss_cols = miss[miss['pct']>50].index.tolist()
        if high_miss_cols:
            miss_corr = df[high_miss_cols].isnull().corr()
            upper = miss_corr.where(np.triu(np.ones(miss_corr.shape), k=1).astype(bool))
            pairs = (upper.stack().reset_index()
                     .rename(columns={'level_0':'col_A','level_1':'col_B',0:'correlation'})
                     .sort_values('correlation', ascending=False))
            pairs['correlation'] = pairs['correlation'].round(4)

            thr_min, thr_max = st.slider(
                "Show pairs with correlation between",
                min_value=0.0, max_value=1.0,
                value=(0.7, 1.0), step=0.05
            )
            strong = pairs[
                (pairs['correlation'] >= thr_min) & (pairs['correlation'] <= thr_max)
            ].reset_index(drop=True)
            st.caption(f"Showing **{len(strong)}** pairs with correlation in [{thr_min:.2f}, {thr_max:.2f}]")
            st.dataframe(strong.head(60), use_container_width=True)
            ins("Many variables tend to be missing together. This suggests that missingness is clustered by clinical measurement groups, such as blood gas tests, blood panels, or invasive monitoring.")

        concl("Missing Values",[
            f"Missing values are widespread: {len(miss)} of {df.shape[1]} columns contain at least one missing value.",
            f"{int((miss['pct']>50).sum())} columns have more than 50% missingness, many related to first-hour clinical measurements.",
            "High-missingness columns should be reviewed carefully before modeling rather than removed automatically.",
            "Missingness appears clustered: related clinical measurements often go missing together.",
            f"Rows with >50% missing values represent {int(sparse_mask.sum()):,} patients ({sparse_pct:.1f}%) and have a similar mortality rate to other rows.",
            "Modeling implication: missing data will require a planned preprocessing strategy, such as imputation, missingness indicators, or model-specific handling.",
        ])

# ══════════════════════════════════════════════════════════════════════════
# ③ NUMERIC DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[2]:
        sh("STEP 03","Numeric & Binary Feature Distributions","Assessing numeric feature skewness, missingness, and binary feature prevalence before modeling.")

        tab_ov, tab_drill = st.tabs(["📊 All columns overview","🔍 Single Feature Drill-Down"])

        with tab_ov:
            desc = get_desc(df, true_num_cols)
            high_skew = desc[desc['skew'].abs()>2].sort_values('skew', key=abs, ascending=False)
            c1,c2 = st.columns(2)
            c1.metric("Highly skewed (|skew|>2)", len(high_skew))
            c2.metric("Total numeric columns", len(true_num_cols))

            st.subheader("Numeric Features: Skewness vs. Missingness")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "Each point represents a numeric feature. The x-axis shows skewness and the y-axis shows missingness. "
                "Features with high skewness may require transformation or robust preprocessing before modeling."
                "</p>", unsafe_allow_html=True
            )
            skew_ser = desc['skew'].dropna()
            miss_ser = desc['missing%'].reindex(skew_ser.index).fillna(0)
            skew_cols = skew_ser.index.tolist()
            skew_vals = [float(v) for v in skew_ser.values.tolist()]
            miss_vals = [float(v) for v in miss_ser.values.tolist()]

            # top 10 most skewed for labeling only
            skewed_indices = sorted(
                [i for i in range(len(skew_vals)) if abs(skew_vals[i]) > 2],
                key=lambda i: abs(skew_vals[i]), reverse=True
            )
            top10_label_idx = set(skewed_indices[:10])

            fig = go.Figure()
            normal_x = [skew_vals[i] for i in range(len(skew_vals)) if abs(skew_vals[i]) <= 2]
            normal_y = [miss_vals[i] for i in range(len(miss_vals)) if abs(skew_vals[i]) <= 2]
            normal_t = [skew_cols[i] for i in range(len(skew_cols)) if abs(skew_vals[i]) <= 2]
            fig.add_trace(go.Scatter(
                x=normal_x, y=normal_y, mode='markers', name='|skew| ≤ 2',
                marker=dict(color=C1, size=7, opacity=0.7),
                text=normal_t, hovertemplate='<b>%{text}</b><br>Skew: %{x:.2f}<br>Missing: %{y:.1f}%<extra></extra>'
            ))
            # skewed — split into labelled top 10 and unlabelled rest
            skewed_all_x = [skew_vals[i] for i in range(len(skew_vals)) if abs(skew_vals[i]) > 2]
            skewed_all_y = [miss_vals[i] for i in range(len(miss_vals)) if abs(skew_vals[i]) > 2]
            skewed_all_t = [skew_cols[i] for i in range(len(skew_cols)) if abs(skew_vals[i]) > 2]
            skewed_all_idx = [i for i in range(len(skew_vals)) if abs(skew_vals[i]) > 2]

            lab_x = [skew_vals[i] for i in skewed_all_idx if i in top10_label_idx]
            lab_y = [miss_vals[i] for i in skewed_all_idx if i in top10_label_idx]
            lab_t = [skew_cols[i] for i in skewed_all_idx if i in top10_label_idx]
            nolab_x = [skew_vals[i] for i in skewed_all_idx if i not in top10_label_idx]
            nolab_y = [miss_vals[i] for i in skewed_all_idx if i not in top10_label_idx]
            nolab_t = [skew_cols[i] for i in skewed_all_idx if i not in top10_label_idx]

            fig.add_trace(go.Scatter(
                x=nolab_x, y=nolab_y, mode='markers', name='|skew| > 2',
                marker=dict(color=C2, size=9, opacity=0.7),
                text=nolab_t, hovertemplate='<b>%{text}</b><br>Skew: %{x:.2f}<br>Missing: %{y:.1f}%<extra></extra>',
                showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=lab_x, y=lab_y, mode='markers+text', name='|skew| > 2 (top 10)',
                marker=dict(color=C2, size=11, opacity=0.9),
                text=lab_t, textposition='top center',
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>Skew: %{x:.2f}<br>Missing: %{y:.1f}%<extra></extra>'
            ))
            fig.add_vline(x=2, line_dash='dash', line_color='red', opacity=0.5)
            fig.add_vline(x=-2, line_dash='dash', line_color='red', opacity=0.5)
            fig.add_vline(x=0, line_dash='dot', line_color='gray', opacity=0.4)
            fig.update_layout(**PT, height=420,
                              xaxis_title='Skewness', yaxis_title='Missing %',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig, use_container_width=True)
            ins("Red dots indicate highly skewed features (|skewness| > 2). Features with both high skewness and high missingness should be reviewed carefully during preprocessing.")

            st.subheader("Top Skewed Numeric Features — Distribution Summary")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This table summarizes the numeric features with the strongest distribution skew. "
                "Detailed checks for suspicious or clinically implausible values are handled in the Outliers section."
                "</p>", unsafe_allow_html=True
            )
            display_cols = ['mean','std','min','25%','50%','75%','max','skew','missing%']
            display_cols = [c for c in display_cols if c in high_skew.columns]
            skew_display = high_skew[display_cols].head(20).copy()
            for c in skew_display.columns:
                skew_display[c] = skew_display[c].apply(lambda x: round(float(x), 3) if pd.notnull(x) else None)
            st.dataframe(skew_display, use_container_width=True)

            divider()
            st.subheader("Binary Feature Prevalence")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This chart shows the percentage of patients with value = 1 for each binary feature. "
                "It helps distinguish common clinical indicators from rare comorbidities."
                "</p>", unsafe_allow_html=True
            )
            bin_pct = [float(df[c].mean()*100) for c in binary_cols]
            bin_order = sorted(range(len(binary_cols)), key=lambda i: bin_pct[i], reverse=True)
            fig_b = go.Figure(go.Bar(
                x=[binary_cols[i] for i in bin_order],
                y=[bin_pct[i] for i in bin_order],
                marker_color=C2,
                text=[f"{bin_pct[i]:.1f}%" for i in bin_order],
                textposition='outside'
            ))
            fig_b.update_layout(**PT, height=320, xaxis_tickangle=-40,
                                xaxis_title='', yaxis_title='Patients with value = 1 (%)',
                                yaxis=dict(range=[0, max(bin_pct)*1.25]))
            st.plotly_chart(fig_b, use_container_width=True)

        with tab_drill:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.4rem;'>"
                "Select a numeric clinical feature to inspect its distribution, missingness, and separation between survived and died patients."
                "</p>", unsafe_allow_html=True
            )

            # exclude identifier columns
            ID_COLS = {'patient_id','encounter_id','hospital_id','icu_id'}
            drill_cols = [c for c in true_num_cols if c not in ID_COLS]

            # pick a sensible default
            preferred = ['age','bmi','apache_4a_hospital_death_prob','d1_spo2_min','d1_heartrate_max','creatinine_apache','glucose_apache']
            default_col = next((c for c in preferred if c in drill_cols), drill_cols[0])
            default_idx = drill_cols.index(default_col)

            st.caption("Identifier columns are excluded because their numeric values do not represent clinical measurements.")
            col_sel = st.selectbox("Select feature", drill_cols, index=default_idx, key='dist_col')
            raw = df[col_sel].dropna()
            series = [float(v) for v in raw.tolist()]
            if len(series) == 0:
                st.warning("No data for this column."); st.stop()

            bins_n = st.slider("Histogram bins", 10, 150, 40, key='dist_bins')

            # summary metrics — 5 cards, no kurtosis
            sk = float(raw.skew())
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Mean", f"{float(raw.mean()):.4g}")
            c2.metric("Median", f"{float(raw.median()):.4g}")
            c3.metric("Std", f"{float(raw.std()):.4g}")
            c4.metric("Skewness", f"{sk:.3f}")
            c5.metric("Missing", f"{df[col_sel].isnull().mean()*100:.1f}%")

            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin:.8rem 0 .4rem 0;'>"
                "A feature may be informative if its distribution differs between survived and died patients. "
                "Similar distributions suggest limited univariate signal, although the feature may still contribute when combined with other variables."
                "</p>", unsafe_allow_html=True
            )

            survived_data = [float(v) for v in df[df[TARGET]==0][col_sel].dropna().tolist()]
            died_data = [float(v) for v in df[df[TARGET]==1][col_sel].dropna().tolist()]
            q_lo = float(raw.quantile(.01))
            q_hi = float(raw.quantile(.99))
            surv_clip = [max(q_lo, min(q_hi, v)) for v in survived_data]
            died_clip = [max(q_lo, min(q_hi, v)) for v in died_data]

            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=["Overall Distribution", "Distribution by Outcome"],
                                column_widths=[0.55, 0.45])
            fig.add_trace(go.Histogram(x=series, nbinsx=bins_n, name='All patients',
                                       marker_color=C1, opacity=0.7, histnorm=''), row=1, col=1)
            kx, ky = kde_curve(series)
            if kx:
                hist_vals, bin_edges = np.histogram(series, bins=bins_n)
                bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges)>1 else 1.0
                scale = len(series) * bin_width
                fig.add_trace(go.Scatter(x=kx, y=[v*scale for v in ky],
                                         mode='lines', name='KDE',
                                         line=dict(color='#e07a5f', width=2.5)), row=1, col=1)
            fig.add_trace(go.Box(y=surv_clip, name='Survived', marker_color=C1,
                                 boxmean=True, showlegend=False), row=1, col=2)
            fig.add_trace(go.Box(y=died_clip, name='Died', marker_color=C2,
                                 boxmean=True, showlegend=False), row=1, col=2)
            fig.update_layout(**PT, height=400, title=dict(text=col_sel, font_size=14))
            fig.update_xaxes(title_text=col_sel, row=1, col=1)
            fig.update_yaxes(title_text='Count', row=1, col=1)
            fig.update_xaxes(title_text='', row=1, col=2)
            fig.update_yaxes(title_text=col_sel, row=1, col=2)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Left: histogram and KDE for all patients with non-missing values. Right: boxplot comparing the feature between survived and died patients.")

            divider()
            st.subheader(f"📋 Feature Summary — `{col_sel}`")
            surv_m = float(raw[df[TARGET]==0].mean()) if len(survived_data)>0 else 0
            died_m = float(raw[df[TARGET]==1].mean()) if len(died_data)>0 else 0
            diff_direction = "higher" if died_m > surv_m else "lower"
            diff_pct_val = abs(died_m - surv_m) / abs(surv_m) * 100 if surv_m != 0 else 0
            miss_pct_col = float(df[col_sel].isnull().mean() * 100)
            neg_count = int((df[col_sel] < 0).sum())
            bullets = []
            if abs(sk) <= 1:
                bullets.append(f"Approximately symmetric distribution based on skewness (skew={sk:.2f}).")
            elif sk > 1:
                bullets.append(f"Right-skewed (skewness={sk:.2f}) — most patients have low values, with a long tail of high values. Consider log1p transform.")
            else:
                bullets.append(f"Left-skewed (skewness={sk:.2f}) — most patients cluster at high values (e.g. SpO2 near 100% is clinically expected).")
            if diff_pct_val > 20:
                bullets.append(f"Strong outcome separation: died patients have {diff_direction} average values ({died_m:.3g} vs {surv_m:.3g}, {diff_pct_val:.1f}% difference). Likely a useful feature.")
            elif diff_pct_val > 5:
                bullets.append(f"Moderate outcome separation: died patients have {diff_direction} average values ({died_m:.3g} vs {surv_m:.3g}, {diff_pct_val:.1f}% difference).")
            else:
                bullets.append(f"Limited univariate separation: average values are similar between survived and died patients ({surv_m:.3g} vs {died_m:.3g}). The feature may still contribute when combined with other variables.")
            if miss_pct_col > 50:
                bullets.append(f"High missingness ({miss_pct_col:.1f}%) — this column requires careful review before modeling.")
            elif miss_pct_col > 5:
                bullets.append(f"Moderate missingness ({miss_pct_col:.1f}%) — impute with median or use a missingness indicator before modeling.")
            else:
                bullets.append(f"Low missingness ({miss_pct_col:.1f}%) — no major missing-data concern for this feature.")
            if neg_count > 0:
                bullets.append(f"⚠️ {neg_count} negative values found — investigate as a potential data quality issue.")
            concl(col_sel, bullets)

        concl("Numeric & Binary Distributions",[
            f"The dataset contains {len(true_num_cols)} numeric features, of which {len(high_skew)} are highly skewed (|skewness| > 2).",
            "Highly skewed variables may require transformations, robust scaling, or model choices that are less sensitive to distribution shape.",
            "Some skewed variables also have substantial missingness, so distribution shape and missingness should be considered together during preprocessing.",
            "SpO2 variables are negatively skewed because most values are close to 100%, which is clinically plausible.",
            "Binary features vary substantially in prevalence: ventilation, diabetes, and elective surgery are relatively common, while several comorbidities are rare.",
            "Some variables show unusual ranges that will be reviewed in the Outliers section.",
            "Modeling implication: preprocessing should account for skewed numeric variables and rare binary indicators, which may affect model calibration.",
        ])

# ══════════════════════════════════════════════════════════════════════════
# ④ OUTLIERS
# ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[3]:
        sh("STEP 04","Outliers","IQR-based statistical outlier detection across numeric clinical features.")
        st.markdown(
            "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
            "Outliers are detected using the IQR rule. In ICU data, statistical outliers often represent clinically severe patients "
            "rather than data errors, so they should be reviewed carefully instead of removed automatically."
            "</p>", unsafe_allow_html=True
        )

        out_series = get_outlier_counts(df, true_num_cols)
        out_pct = (out_series / len(df) * 100).round(2)
        out_df = pd.DataFrame({
            'Feature': out_series.index.tolist(),
            'IQR-flagged values': [int(v) for v in out_series.values.tolist()],
            'Percent flagged': [float(v) for v in out_pct.values.tolist()]
        })

        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single Feature Outlier Inspection"])

        with tab_ov:
            top20 = out_df.head(20).copy()
            st.caption("Top features by percentage of values flagged as statistical outliers using the IQR rule.")
            fig = go.Figure(go.Bar(
                x=top20['Feature'].tolist(),
                y=top20['Percent flagged'].tolist(),
                marker_color=[C2 if v>10 else C1 for v in top20['Percent flagged'].tolist()],
                text=[f"{v:.1f}%" for v in top20['Percent flagged'].tolist()],
                textposition='outside'
            ))
            fig.update_layout(**PT, height=360, xaxis_tickangle=-50,
                              xaxis_title='', yaxis_title='% of rows flagged by IQR rule',
                              yaxis=dict(range=[0, float(top20['Percent flagged'].max())*1.3]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out_df.head(20), use_container_width=True)
            ins("In ICU data, many statistical outliers may be clinically meaningful because critically ill patients often have extreme physiological values.")
            wrn("Some high outlier rates occur in bounded or ordinal clinical scores, such as GCS. For these variables, IQR flags may reflect the discrete scale structure rather than true abnormal values.")

            concl("Outliers",[
                "Outliers were detected using the IQR rule, so they represent statistical extremes rather than confirmed data errors.",
                "In ICU data, many extreme values may be clinically meaningful because severely ill patients often have abnormal physiological measurements.",
                "GCS variables show high IQR-flag rates, likely due to their bounded ordinal scale and concentration at clinically meaningful extremes.",
                "pre_icu_los_days contains negative values, which should be reviewed as a potential timestamp or data quality issue before modeling.",
                "Extreme values in creatinine, BUN, and bilirubin may reflect severe organ dysfunction and should not be removed automatically.",
                "Modeling implication: outlier handling should be feature-specific and clinically informed rather than based on automatic removal.",
            ])

        with tab_col:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.4rem;'>"
                "Select a numeric clinical feature to inspect IQR-based outliers and compare its distribution between survived and died patients."
                "</p>", unsafe_allow_html=True
            )

            ID_COLS = {'patient_id','encounter_id','hospital_id','icu_id'}
            drill_cols_out = [c for c in true_num_cols if c not in ID_COLS]

            preferred_out = ['pre_icu_los_days','creatinine_apache','bun_apache','bilirubin_apache',
                             'd1_creatinine_max','d1_spo2_min','temp_apache','gcs_motor_apache']
            default_out = next((c for c in preferred_out if c in drill_cols_out), drill_cols_out[0])
            default_out_idx = drill_cols_out.index(default_out)

            st.caption("Identifier columns are excluded because their numeric values do not represent clinical measurements and should not be inspected as continuous outlier-prone features.")
            col_sel = st.selectbox("Select feature", drill_cols_out, index=default_out_idx, key='out_col')
            s = df[col_sel].dropna()

            if len(s) > 0:
                Q1,Q3 = float(s.quantile(.25)), float(s.quantile(.75))
                IQR = Q3 - Q1
                lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                outlier_mask = df[col_sel].notna() & ((df[col_sel]<lo)|(df[col_sel]>hi))
                n_flagged = int(outlier_mask.sum())
                pct_flagged = float(outlier_mask.mean()*100)

                c1,c2,c3,c4 = st.columns(4)
                c1.metric("IQR-flagged values", f"{n_flagged:,}")
                c2.metric("Percent flagged", f"{pct_flagged:.1f}%")
                c3.metric("Lower fence", f"{lo:.4g}")
                c4.metric("Upper fence", f"{hi:.4g}")
                st.caption("IQR fences define statistical outlier thresholds, not automatic clinical error thresholds.")

                st.markdown(
                    "<p style='color:#6a5a4a; font-size:.87rem; margin:.8rem 0 .4rem 0;'>"
                    "Values outside the IQR fences are statistical outliers. In ICU data, these values may reflect severe illness, "
                    "measurement processes, or potential data quality issues depending on the feature."
                    "</p>", unsafe_allow_html=True
                )

                fig = go.Figure()
                fig.add_trace(go.Box(
                    y=[float(v) for v in df[df[TARGET]==0][col_sel].dropna().tolist()],
                    name='Survived', marker_color=C1, boxmean=True, boxpoints='outliers',
                    marker=dict(size=3, opacity=0.4)
                ))
                fig.add_trace(go.Box(
                    y=[float(v) for v in df[df[TARGET]==1][col_sel].dropna().tolist()],
                    name='Died', marker_color=C2, boxmean=True, boxpoints='outliers',
                    marker=dict(size=3, opacity=0.4)
                ))
                fig.update_layout(**PT, height=400, yaxis_title=col_sel,
                                  title=f"Distribution by Outcome — {col_sel}")
                st.plotly_chart(fig, use_container_width=True)

                if n_flagged > 0:
                    st.subheader(f"Example IQR-flagged values — `{col_sel}`")
                    sample_cols = [col_sel, TARGET]
                    for extra in ['age','gender','ethnicity']:
                        if extra in df.columns:
                            sample_cols.append(extra)
                    flagged_sample = df[outlier_mask][sample_cols].head(10)
                    st.dataframe(flagged_sample, use_container_width=True)
                else:
                    ins(f"No values were flagged by the IQR rule for this feature.")

                divider()

                # feature-specific summary
                GCS_COLS = {'gcs_motor_apache','gcs_eyes_apache','gcs_verbal_apache','gcs_unable_apache'}
                LAB_COLS = {'creatinine_apache','bun_apache','bilirubin_apache','d1_creatinine_max',
                            'd1_bilirubin_max','d1_bun_max','h1_creatinine_max'}
                neg_count = int((df[col_sel] < 0).sum())

                bullets = [
                    f"IQR-flagged values: {n_flagged:,} ({pct_flagged:.1f}% of rows).",
                    f"Lower fence: {lo:.4g}; upper fence: {hi:.4g}.",
                    "These values should be interpreted as statistical extremes, not automatic data errors.",
                    "Handling decision should depend on the clinical meaning of this feature.",
                ]
                if col_sel == 'pre_icu_los_days' or neg_count > 0:
                    bullets.append(f"⚠️ Negative values detected ({neg_count:,} rows). These should be reviewed as a potential timestamp or data quality issue before modeling.")
                elif col_sel in LAB_COLS:
                    bullets.append("Extreme high values in lab variables may reflect severe organ dysfunction and should not be removed automatically.")
                elif col_sel in GCS_COLS:
                    bullets.append("IQR flags in GCS variables may reflect the bounded ordinal scale rather than true abnormal values.")

                concl(f"Feature Summary — {col_sel}", bullets)

# ══════════════════════════════════════════════════════════════════════════
# ⑤ CATEGORICAL COLUMNS
# ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[4]:
        sh("STEP 05","Categorical Columns","Summarizing categorical feature cardinality, missingness, dominant categories, and outcome differences.")

        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single Categorical Feature Drill-Down"])

        with tab_ov:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This table summarizes each categorical feature by number of unique categories, missingness, most frequent category, and dominance of the top category."
                "</p>", unsafe_allow_html=True
            )
            rows = []
            for c in cat_cols:
                vc = df[c].dropna()
                top_val = str(vc.value_counts().index[0]) if len(vc)>0 else 'N/A'
                top_freq = round(vc.value_counts(normalize=True).iloc[0]*100, 1) if len(vc)>0 else 0
                rows.append({
                    'Feature': c,
                    'Unique categories': int(df[c].nunique()),
                    'Missing %': round(df[c].isnull().mean()*100, 2),
                    'Most frequent category': top_val,
                    'Top category %': top_freq
                })
            cat_summary = pd.DataFrame(rows).sort_values('Unique categories', ascending=False)
            st.dataframe(cat_summary, use_container_width=True)
            ins("A very dominant category (>90%) indicates low variability. Such features may provide limited predictive signal unless the minority categories have clinically meaningful risk differences.")

            divider()

            # ── Strongest Categorical Outcome Differences ─────────────────────
            st.subheader("Strongest Categorical Outcome Differences")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "For each categorical feature, mortality rates were compared across categories to identify potential univariate signal."
                "</p>", unsafe_allow_html=True
            )
            signal_rows = []
            for c in cat_cols:
                try:
                    sub = df[[c, TARGET]].dropna()
                    ct = sub.groupby(c)[TARGET].mean() * 100
                    if len(ct) < 2: continue
                    lo_cat = ct.idxmin(); hi_cat = ct.idxmax()
                    lo_pct = round(float(ct.min()), 1); hi_pct = round(float(ct.max()), 1)
                    signal_rows.append({
                        'Feature': c,
                        'Lowest-risk category': str(lo_cat),
                        'Lowest mortality %': lo_pct,
                        'Highest-risk category': str(hi_cat),
                        'Highest mortality %': hi_pct,
                        'Risk difference (pp)': round(hi_pct - lo_pct, 1)
                    })
                except: pass
            if signal_rows:
                signal_df = pd.DataFrame(signal_rows).sort_values('Risk difference (pp)', ascending=False)
                st.dataframe(signal_df, use_container_width=True)

                # bar chart of risk differences
                fig_sig = go.Figure(go.Bar(
                    x=signal_df['Feature'].tolist(),
                    y=signal_df['Risk difference (pp)'].tolist(),
                    marker_color=[C2 if v > 5 else C1 for v in signal_df['Risk difference (pp)'].tolist()],
                    text=[f"{v:.1f}pp" for v in signal_df['Risk difference (pp)'].tolist()],
                    textposition='outside'
                ))
                fig_sig.update_layout(**PT, height=320, xaxis_tickangle=-30,
                                      xaxis_title='', yaxis_title='Mortality rate difference (pp)',
                                      yaxis=dict(range=[0, signal_df['Risk difference (pp)'].max()*1.3]))
                st.plotly_chart(fig_sig, use_container_width=True)

            concl("Categorical Features",[
                f"The dataset contains {len(cat_cols)} categorical features with low-to-moderate cardinality, making them suitable for encoding before modeling.",
                "Some features show meaningful outcome differences across categories, especially ICU admission source, APACHE body system, and hospital admission source.",
                "icu_admit_source shows large mortality-rate differences across categories, suggesting potential predictive signal.",
                "apache_3j_bodysystem shows the largest univariate categorical difference observed, with substantial variation across clinical body systems.",
                "Gender shows very similar mortality rates across groups, suggesting limited univariate signal.",
                "Some categorical values require cleaning before encoding, such as inconsistent casing in APACHE body system categories.",
                "Modeling implication: categorical features should be cleaned, encoded, and checked for rare categories before model training.",
            ])

        with tab_col:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.4rem;'>"
                "Select a categorical feature to inspect category frequencies and compare mortality rates across categories."
                "</p>", unsafe_allow_html=True
            )
            col_sel = st.selectbox("Select column", cat_cols, key='cat_col')
            top_n = st.slider("Number of categories to display", 3, 30, 15, key='cat_n')

            # clean up None/NaN display label
            vc_raw = df[col_sel].value_counts(dropna=False).head(top_n).reset_index()
            vc_raw.columns = ['Category', 'Patient count']
            vc_raw['Category'] = vc_raw['Category'].astype(str).replace({'nan': 'Missing / Unknown', 'None': 'Missing / Unknown'})
            vc_vals = [int(v) for v in vc_raw['Patient count'].tolist()]
            vc_labels = vc_raw['Category'].tolist()

            st.subheader(f"Category Frequency — {col_sel}")
            st.caption("This chart shows how many patients belong to each category of the selected feature.")
            c_left, c_right = st.columns([2,1])
            with c_left:
                fig = go.Figure(go.Bar(
                    x=vc_vals[::-1], y=vc_labels[::-1],
                    orientation='h', marker_color=C1,
                    text=[str(v) for v in vc_vals[::-1]], textposition='outside'
                ))
                fig.update_layout(**PT, height=max(300,top_n*28),
                                  xaxis_title='Patient count', yaxis_title='',
                                  xaxis=dict(range=[0, max(vc_vals)*1.2]))
                st.plotly_chart(fig, use_container_width=True)
            with c_right:
                st.dataframe(vc_raw, use_container_width=True)

            divider()
            st.subheader(f"Mortality Rate by Category — {col_sel}")
            top_cats_raw = df[col_sel].value_counts().head(top_n).index
            sub = df[df[col_sel].isin(top_cats_raw)].copy()
            sub[col_sel] = sub[col_sel].astype(str).replace({'nan': 'Missing / Unknown', 'None': 'Missing / Unknown'})

            ct = pd.crosstab(sub[col_sel], sub[TARGET], normalize='index') * 100
            if 1 in ct.columns:
                ct = ct.rename(columns={0:'Survival rate (%)', 1:'Mortality rate (%)'})
                ct['Patient count'] = df[col_sel].value_counts().reindex(
                    df[col_sel].astype(str).replace({'nan':'Missing / Unknown','None':'Missing / Unknown'}).value_counts().index
                ).fillna(0).astype(int)
                ct = ct.sort_values('Mortality rate (%)', ascending=False).reset_index()
                ct[col_sel] = ct[col_sel].astype(str)

                mort_vals = [float(v) for v in ct['Mortality rate (%)'].tolist()]
                cat_labels = ct[col_sel].tolist()

                # simple mortality-rate bar chart (much easier to read than 100% stacked)
                fig_mort = go.Figure(go.Bar(
                    x=mort_vals, y=cat_labels,
                    orientation='h',
                    marker_color=[C2 if v > df[TARGET].mean()*100 else C1 for v in mort_vals],
                    text=[f"{v:.1f}%" for v in mort_vals], textposition='outside'
                ))
                overall_dr = float(df[TARGET].mean() * 100)
                fig_mort.add_vline(x=overall_dr, line_dash='dash', line_color='gray',
                                   annotation_text=f"Overall: {overall_dr:.1f}%",
                                   annotation_position='top right')
                fig_mort.update_layout(**PT,
                                       height=max(320, len(ct)*32),
                                       xaxis_title='Mortality rate (%)', yaxis_title='',
                                       yaxis_autorange='reversed',
                                       xaxis=dict(range=[0, max(mort_vals)*1.25]))
                st.plotly_chart(fig_mort, use_container_width=True)
                st.caption("Dashed line shows the overall dataset mortality rate. Red bars indicate categories above average mortality.")

                st.dataframe(ct[[col_sel, 'Mortality rate (%)', 'Survival rate (%)', 'Patient count']], use_container_width=True)

                divider()

                # feature-specific conclusion
                n_cats = int(df[col_sel].nunique(dropna=False))
                top_cat = str(df[col_sel].value_counts().index[0]) if len(df[col_sel].dropna()) > 0 else 'N/A'
                top_freq_pct = round(float(df[col_sel].value_counts(normalize=True).iloc[0] * 100), 1)
                min_mort = round(min(mort_vals), 1)
                max_mort = round(max(mort_vals), 1)
                min_cat = cat_labels[mort_vals.index(min(mort_vals))]
                max_cat = cat_labels[mort_vals.index(max(mort_vals))]

                concl(f"Feature Summary — {col_sel}", [
                    f"This feature has {n_cats} categories (showing top {min(top_n, n_cats)}).",
                    f"The most common category is '{top_cat}', representing {top_freq_pct}% of patients.",
                    f"Mortality rates range from {min_mort}% ({min_cat}) to {max_mort}% ({max_cat}) across displayed categories.",
                    "Differences in small categories should be interpreted carefully due to limited sample size.",
                    "Modeling implication: this feature should be cleaned, rare categories should be reviewed, and appropriate encoding applied before model training.",
                ])

# ══════════════════════════════════════════════════════════════════════════
# ⑥ FEATURE VS TARGET
# ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[5]:
        sh("STEP 06","Feature vs Target","Which features show the strongest univariate association with hospital mortality?")

        tab_num, tab_bin, tab_cat, tab_corr = st.tabs([
            "📈 Numeric — standardized difference","🔢 Binary — death rates",
            "🏷️ Categorical — mortality rates","📐 Correlation with target"])

        with tab_num:
            ID_COLS = {'patient_id','encounter_id','hospital_id','icu_id'}
            num_drill_cols = [c for c in true_num_cols if c not in ID_COLS]

            # compute standardized mean difference for all numeric features
            std_rows = []
            for col in num_drill_cols:
                s0 = df[df[TARGET]==0][col].dropna()
                s1 = df[df[TARGET]==1][col].dropna()
                if len(s0) < 2 or len(s1) < 2: continue
                m0, m1 = float(s0.mean()), float(s1.mean())
                pooled_std = float(np.sqrt((s0.var() + s1.var()) / 2))
                smd = abs(m1 - m0) / pooled_std if pooled_std > 0 else 0.0
                std_rows.append({
                    'feature': col,
                    'Mean — Survived': round(m0, 4),
                    'Mean — Died': round(m1, 4),
                    'Standardized difference': round(smd, 4),
                    'Missing %': round(float(df[col].isnull().mean()*100), 2),
                    'Higher in': 'Died' if m1 > m0 else 'Survived'
                })
            smd_df = pd.DataFrame(std_rows).sort_values('Standardized difference', ascending=False).reset_index(drop=True)

            st.subheader("Top 20 Numeric Features by Standardized Mean Difference")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "Raw mean differences are affected by feature scale. Standardized mean difference compares the survived and died groups "
                "after accounting for each feature's variability, making numeric features more comparable. "
                "Formula: |mean_died − mean_survived| / pooled standard deviation."
                "</p>", unsafe_allow_html=True
            )
            top20_smd = smd_df.head(20)
            smd_vals = top20_smd['Standardized difference'].tolist()
            feat_names_smd = top20_smd['feature'].tolist()
            bar_colors = [C2 if r == 'Died' else C1 for r in top20_smd['Higher in'].tolist()]

            fig_smd = go.Figure(go.Bar(
                x=smd_vals[::-1], y=feat_names_smd[::-1], orientation='h',
                marker_color=bar_colors[::-1],
                text=[f"{v:.3f}" for v in smd_vals[::-1]], textposition='outside'
            ))
            fig_smd.update_layout(**PT, height=max(420, len(top20_smd)*24),
                                  xaxis_title='Standardized mean difference', yaxis_title='',
                                  xaxis=dict(range=[0, max(smd_vals)*1.25]))
            st.plotly_chart(fig_smd, use_container_width=True)
            st.caption("Red bars = higher average in died patients. Blue bars = higher average in survived patients.")

            # optional top 12 box plots behind checkbox
            divider()
            if st.checkbox("Show box plots for top numeric features", value=False):
                st.subheader("Top 12 Features — Distribution by Outcome (clipped to 1st–99th percentile)")
                top12 = smd_df.head(12)['feature'].tolist()
                for row_i in range(0, len(top12), 3):
                    row_cols = st.columns(3)
                    for ci, col in enumerate(top12[row_i:row_i+3]):
                        if ci >= len(top12[row_i:]): break
                        with row_cols[ci]:
                            q_lo = float(df[col].quantile(.01))
                            q_hi = float(df[col].quantile(.99))
                            pf = df[[col,TARGET]].copy()
                            pf[col] = pf[col].clip(q_lo, q_hi)
                            surv = [float(v) for v in pf[pf[TARGET]==0][col].dropna().tolist()]
                            died = [float(v) for v in pf[pf[TARGET]==1][col].dropna().tolist()]
                            fig = go.Figure()
                            fig.add_trace(go.Box(y=surv, name='Survived', marker_color=C1,
                                                 showlegend=False, boxmean=True))
                            fig.add_trace(go.Box(y=died, name='Died', marker_color=C2,
                                                 showlegend=False, boxmean=True))
                            fig.update_layout(**PT, height=280,
                                              title=dict(text=col, font_size=10),
                                              margin=dict(t=35,b=10,l=10,r=10))
                            st.plotly_chart(fig, use_container_width=True)

            divider()
            st.subheader("Selected Feature Distribution by Outcome")
            st.caption("Identifier columns are excluded because their numeric values do not represent clinical measurements.")
            preferred_fvt = ['age','d1_lactate_min','gcs_motor_apache','creatinine_apache','apache_4a_hospital_death_prob']
            default_fvt = next((c for c in preferred_fvt if c in num_drill_cols), num_drill_cols[0])
            col_sel = st.selectbox("Select feature", num_drill_cols, index=num_drill_cols.index(default_fvt), key='fvt_num')

            row_match = smd_df[smd_df['feature']==col_sel]
            surv_mean = float(row_match['Mean — Survived'].iloc[0]) if len(row_match) > 0 else float(df[df[TARGET]==0][col_sel].mean())
            died_mean = float(row_match['Mean — Died'].iloc[0]) if len(row_match) > 0 else float(df[df[TARGET]==1][col_sel].mean())
            smd_val = float(row_match['Standardized difference'].iloc[0]) if len(row_match) > 0 else 0.0
            miss_pct = float(df[col_sel].isnull().mean()*100)

            c1,c2,c3,c4 = st.columns(4)
            c1.metric("Mean — Survived", f"{surv_mean:.4g}")
            c2.metric("Mean — Died", f"{died_mean:.4g}")
            c3.metric("Standardized difference", f"{smd_val:.3f}")
            c4.metric("Missing %", f"{miss_pct:.1f}%")

            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin:.6rem 0 .4rem 0;'>"
                "These plots compare the selected feature between survived and died patients. "
                "Visible separation suggests potential univariate predictive signal, but does not imply causality."
                "</p>", unsafe_allow_html=True
            )

            q_lo = float(df[col_sel].quantile(.01))
            q_hi = float(df[col_sel].quantile(.99))
            pf = df[[col_sel,TARGET]].copy()
            pf[col_sel] = pf[col_sel].clip(q_lo, q_hi)
            surv_data = [float(v) for v in pf[pf[TARGET]==0][col_sel].dropna().tolist()]
            died_data = [float(v) for v in pf[pf[TARGET]==1][col_sel].dropna().tolist()]

            fig2 = make_subplots(rows=1, cols=2,
                                 subplot_titles=["Distribution by Outcome — KDE", "Distribution by Outcome — Box Plot"],
                                 column_widths=[0.55, 0.45])
            kx_s, ky_s = kde_curve(surv_data)
            kx_d, ky_d = kde_curve(died_data)
            if kx_s:
                fig2.add_trace(go.Scatter(x=kx_s, y=ky_s, mode='lines', name='Survived',
                                          line=dict(color=C1, width=2.5),
                                          fill='tozeroy', fillcolor='rgba(47,107,200,0.15)'), row=1, col=1)
            if kx_d:
                fig2.add_trace(go.Scatter(x=kx_d, y=ky_d, mode='lines', name='Died',
                                          line=dict(color=C2, width=2.5),
                                          fill='tozeroy', fillcolor='rgba(200,75,47,0.15)'), row=1, col=1)
            fig2.add_trace(go.Box(y=surv_data, name='Survived', marker_color=C1,
                                  showlegend=False, boxmean=True), row=1, col=2)
            fig2.add_trace(go.Box(y=died_data, name='Died', marker_color=C2,
                                  showlegend=False, boxmean=True), row=1, col=2)
            fig2.update_layout(**PT, height=380,
                               legend=dict(orientation='h', yanchor='bottom', y=1.05))
            fig2.update_xaxes(title_text=col_sel, row=1, col=1)
            fig2.update_yaxes(title_text='Density', row=1, col=1)
            fig2.update_yaxes(title_text=col_sel, row=1, col=2)
            st.plotly_chart(fig2, use_container_width=True)

        with tab_bin:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This chart compares mortality rates for each binary feature when the feature is present (value = 1) versus absent (value = 0). "
                "Positive values mean mortality is higher when the feature is present; negative values mean mortality is lower when the feature is present. "
                "These differences are associations, not causal effects."
                "</p>", unsafe_allow_html=True
            )
            rows = []
            for col in binary_cols:
                g = df.groupby(col)[TARGET].mean() * 100
                if 0.0 in g.index and 1.0 in g.index:
                    rows.append({
                        'Feature': col,
                        'Mortality rate when absent (%)': round(float(g[0.0]), 2),
                        'Mortality rate when present (%)': round(float(g[1.0]), 2),
                        'Difference (percentage points)': round(float(g[1.0]-g[0.0]), 2),
                        'Feature prevalence (%)': round(float(df[col].mean()*100), 2)
                    })
            bin_df = pd.DataFrame(rows).sort_values('Difference (percentage points)', key=abs, ascending=False)
            feat_list = bin_df['Feature'].tolist()
            diff_list = [float(v) for v in bin_df['Difference (percentage points)'].tolist()]
            dr0_list = [float(v) for v in bin_df['Mortality rate when absent (%)'].tolist()]
            dr1_list = [float(v) for v in bin_df['Mortality rate when present (%)'].tolist()]

            # grouped bar: mortality rate absent vs present
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Mortality rate when absent (value = 0)', x=feat_list, y=dr0_list,
                                 marker_color=C1, opacity=0.85))
            fig.add_trace(go.Bar(name='Mortality rate when present (value = 1)', x=feat_list, y=dr1_list,
                                 marker_color=C2, opacity=0.85))
            fig.update_layout(**PT, barmode='group', height=380, xaxis_tickangle=-40,
                              xaxis_title='', yaxis_title='Mortality rate (%)',
                              legend=dict(orientation='h', yanchor='bottom', y=1.02))
            st.plotly_chart(fig, use_container_width=True)

            # difference bar
            fig2 = go.Figure(go.Bar(
                x=feat_list, y=diff_list,
                marker_color=[C2 if v>0 else C1 for v in diff_list],
                text=[f"{v:+.1f}pp" for v in diff_list], textposition='outside'
            ))
            fig2.add_hline(y=0, line_color='black', line_width=1)
            max_d = max(abs(min(diff_list)), abs(max(diff_list)))
            fig2.update_layout(**PT, height=320, xaxis_tickangle=-40,
                               xaxis_title='', yaxis_title='Mortality rate difference (percentage points)',
                               yaxis=dict(range=[-max_d*1.4, max_d*1.4]),
                               title='Binary Features: Mortality Rate Difference When Present vs. Absent')
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("The zero line indicates no difference in mortality rate between value = 1 and value = 0.")

            st.dataframe(bin_df, use_container_width=True)

            ins("Negative differences indicate lower observed mortality when the feature is present. This should not be interpreted as a causal protective effect.")
            wrn("Features with low prevalence should be interpreted carefully because their mortality-rate differences may be less stable.")

            divider()
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem;'>"
                "Binary indicators related to ventilation, intubation, and inability to assess GCS show higher observed mortality when present. "
                "Features such as elective surgery and post-operative status show lower observed mortality, likely reflecting patient selection rather than a direct protective effect."
                "</p>", unsafe_allow_html=True
            )

        with tab_cat:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This view compares observed mortality rates across categories for the selected categorical feature. "
                "These are univariate associations and should not be interpreted as causal effects."
                "</p>", unsafe_allow_html=True
            )
            col_sel = st.selectbox("Categorical column", cat_cols, key='fvt_cat')
            top_n = st.slider("Top N categories", 3, 30, 15, key='fvt_catn')

            top_cats = df[col_sel].value_counts().head(top_n).index
            sub = df[df[col_sel].isin(top_cats)].copy()
            ct = pd.crosstab(sub[col_sel], sub[TARGET], normalize='index') * 100
            if 1 in ct.columns:
                ct = ct.rename(columns={0:'Survival rate (%)', 1:'Mortality rate (%)'})
                counts = df[col_sel].value_counts()
                ct['Patient count'] = ct.index.map(counts).fillna(0).astype(int)
                ct = ct.sort_values('Mortality rate (%)', ascending=False).reset_index()
                ct = ct.rename(columns={col_sel: 'Category'})

                mort_vals = [float(v) for v in ct['Mortality rate (%)'].tolist()]
                cat_labels = ct['Category'].astype(str).tolist()
                pat_counts = ct['Patient count'].tolist()

                # summary metric cards
                hi_mort = round(max(mort_vals), 1)
                lo_mort = round(min(mort_vals), 1)
                hi_cat = cat_labels[mort_vals.index(max(mort_vals))]
                lo_cat = cat_labels[mort_vals.index(min(mort_vals))]
                mort_range = round(hi_mort - lo_mort, 1)

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Highest mortality category", hi_cat, f"{hi_mort}%")
                c2.metric("Lowest mortality category", lo_cat, f"{lo_mort}%")
                c3.metric("Mortality range", f"{mort_range} pp")
                c4.metric("Categories shown", len(mort_vals))

                # ethnicity fairness note
                if 'ethnicity' in col_sel.lower():
                    wrn("Ethnicity should be interpreted carefully. It may be more relevant for fairness and subgroup performance analysis than as a direct clinical predictor.")

                # mortality-rate bar chart (simple, easier to read than 100% stacked)
                fig = go.Figure(go.Bar(
                    x=mort_vals, y=cat_labels,
                    orientation='h',
                    marker_color=[C2 if v > float(df[TARGET].mean()*100) else C1 for v in mort_vals],
                    text=[f"{v:.1f}% (n={n:,})" for v, n in zip(mort_vals, pat_counts)],
                    textposition='outside'
                ))
                overall_dr = float(df[TARGET].mean() * 100)
                fig.add_vline(x=overall_dr, line_dash='dash', line_color='gray',
                              annotation_text=f"Overall: {overall_dr:.1f}%",
                              annotation_position='top right')
                fig.update_layout(**PT,
                                  height=max(320, len(ct)*36),
                                  xaxis_title='Mortality rate (%)', yaxis_title='',
                                  yaxis_autorange='reversed',
                                  xaxis=dict(range=[0, max(mort_vals)*1.35]))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("Dashed line shows the overall dataset mortality rate. Patient count (n) is shown in each bar label. Red bars indicate categories above average mortality.")

                ins("Category-level mortality rates should be interpreted together with patient count, because small categories may have less stable estimates.")

                st.dataframe(ct[['Category','Mortality rate (%)','Survival rate (%)','Patient count']], use_container_width=True)

        with tab_corr:
            ID_COLS = {'patient_id','encounter_id','hospital_id','icu_id'}
            LEAKAGE_COLS = {'apache_4a_hospital_death_prob','apache_4a_icu_death_prob'}

            # compute signed Pearson for numeric/binary, keep Cramér's V for categorical
            corr_t_raw = get_corr_target(df, true_num_cols, binary_cols, cat_cols)

            # add signed correlation column
            num_bin_cols = [c for c in true_num_cols + binary_cols if c in df.columns and c not in ID_COLS]
            signed_map = {}
            for c in num_bin_cols:
                try:
                    signed_map[c] = float(df[c].dropna().corr(df[TARGET].reindex(df[c].dropna().index)))
                except: signed_map[c] = 0.0

            # build enriched table
            enriched = []
            for _, row in corr_t_raw.iterrows():
                feat = row['feature']
                if feat in ID_COLS: continue
                ftype = row['type']
                abs_c = float(row['abs_corr']) if pd.notnull(row['abs_corr']) else 0.0
                signed_c = signed_map.get(feat, None)
                direction = 'N/A (Cramér\'s V)' if ftype == 'categorical' else ('Positive' if signed_c and signed_c > 0 else 'Negative')
                enriched.append({
                    'Feature': feat,
                    'Correlation / Association': round(signed_c, 4) if signed_c is not None else round(abs_c, 4),
                    'Absolute strength': round(abs_c, 4),
                    'Feature type': ftype,
                    'Direction': direction,
                    '⚠️ Leakage risk': '⚠️ Yes' if feat in LEAKAGE_COLS else ''
                })
            enrich_df = pd.DataFrame(enriched).sort_values('Absolute strength', ascending=False).reset_index(drop=True)

            st.subheader("Feature Association with hospital_death")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "Numeric and binary features are ranked by Pearson correlation with the target. "
                "Categorical features are ranked by Cramér's V. "
                "These metrics show association strength, not causality."
                "</p>", unsafe_allow_html=True
            )

            # leakage warning box
            wrn("Potential leakage warning: apache_4a_hospital_death_prob and apache_4a_icu_death_prob are pre-calculated mortality-risk scores. "
                "They may inflate apparent predictive signal and should be excluded from fair feature interpretation and most modeling baselines.")

            top_n = st.slider("Show top N features", 10, len(enrich_df), 20, key='corr_n')
            top = enrich_df.head(top_n).copy()

            # ── A. Ranking bar chart ──────────────────────────────────────────
            color_map = {'numeric': C1, 'binary': '#5a9ad4', 'categorical': C2}
            bar_colors = [color_map.get(str(t), C1) for t in top['Feature type'].tolist()]
            abs_vals = top['Absolute strength'].tolist()
            feat_vals = top['Feature'].tolist()

            fig_bar = go.Figure(go.Bar(
                x=abs_vals[::-1], y=feat_vals[::-1], orientation='h',
                marker_color=bar_colors[::-1],
                text=[f"{v:.3f}" for v in abs_vals[::-1]], textposition='outside'
            ))
            # mark leakage features
            for i, feat in enumerate(feat_vals[::-1]):
                if feat in LEAKAGE_COLS:
                    fig_bar.add_annotation(x=abs_vals[len(abs_vals)-1-i], y=feat,
                                           text="⚠️", showarrow=False, xanchor='left', font=dict(size=12))
            fig_bar.update_layout(**PT, height=max(420, top_n*24), yaxis_autorange='reversed',
                                  xaxis_title='Association strength with hospital_death', yaxis_title='',
                                  xaxis=dict(range=[0, max(abs_vals)*1.3]))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.markdown("🔵 **Numeric** &nbsp;&nbsp; 🔷 **Binary** &nbsp;&nbsp; 🔴 **Categorical** (Cramér's V) &nbsp;&nbsp; ⚠️ Potential leakage")

            divider()

            # ── B. Single-column target heatmap (numeric + binary only) ──────
            st.subheader("Heatmap of Top Feature Correlations with hospital_death")
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "Pearson correlation is shown for numeric and binary features only. "
                "Positive values (red) indicate higher values associated with more deaths; "
                "negative values (blue) indicate the opposite. Cramér's V for categorical features is non-directional and is excluded here."
                "</p>", unsafe_allow_html=True
            )
            heatmap_feats = [r['Feature'] for _, r in top.iterrows()
                             if r['Feature type'] != 'categorical' and r['Feature'] not in ID_COLS][:20]
            if heatmap_feats:
                hm_vals = [[signed_map.get(f, 0.0)] for f in heatmap_feats]
                fig_hm = go.Figure(go.Heatmap(
                    z=hm_vals,
                    x=['hospital_death'],
                    y=heatmap_feats,
                    colorscale='RdBu_r', zmid=0, zmin=-0.5, zmax=0.5,
                    colorbar=dict(title='Pearson r'),
                    text=[[f"{v[0]:.3f}"] for v in hm_vals],
                    texttemplate="%{text}",
                    hoverongaps=False
                ))
                fig_hm.update_layout(**PT, height=max(400, len(heatmap_feats)*28),
                                     xaxis_title='', yaxis_title='',
                                     yaxis_autorange='reversed')
                st.plotly_chart(fig_hm, use_container_width=True)

            divider()

            # ── C. Split tables ───────────────────────────────────────────────
            st.subheader("A. Numeric & Binary — Pearson Correlation")
            st.caption("Pearson correlation captures signed linear association. Positive = higher feature value associated with more deaths.")
            nb_table = enrich_df[enrich_df['Feature type'] != 'categorical'][
                ['Feature','Correlation / Association','Absolute strength','Feature type','Direction','⚠️ Leakage risk']
            ].head(top_n)
            st.dataframe(nb_table, use_container_width=True)

            divider()
            st.subheader("B. Categorical — Cramér's V")
            st.caption("Cramér's V measures association strength for categorical features but has no direction. Values range from 0 (no association) to 1 (perfect association).")
            cat_table = enrich_df[enrich_df['Feature type'] == 'categorical'][
                ['Feature','Absolute strength','Feature type','Direction']
            ].head(top_n)
            st.dataframe(cat_table, use_container_width=True)

        concl("Feature vs Target",[
            "Numeric features should be compared using standardized differences rather than raw mean differences, because raw differences depend on feature scale.",
            "Several clinical measurements show visible differences between survived and died patients, suggesting potential univariate predictive signal.",
            "Largest numeric associations include first-day lactate, GCS motor score, and APACHE death probability.",
            "Binary features such as gcs_unable_apache, ventilated_apache, and intubated_apache show higher mortality rates when present, representing the largest binary mortality-rate differences observed.",
            "Features such as elective_surgery and apache_post_operative show lower observed mortality when present, suggesting a potentially protective or selection-related association.",
            "Categorical features such as apache_3j_bodysystem and icu_admit_source show the largest categorical mortality-rate variation across categories.",
            "These associations should not be interpreted as causal effects.",
            "Modeling implication: feature-vs-target analysis can guide feature inspection, but final importance should be evaluated through model performance and explainability.",
        ])

# ══════════════════════════════════════════════════════════════════════════
# ⑦ CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[6]:
        sh("STEP 07","Feature Correlations","Identifying highly correlated features that may carry overlapping information.")

        LEAKAGE_COLS = {'apache_4a_hospital_death_prob','apache_4a_icu_death_prob'}
        ID_COLS_C = {'patient_id','encounter_id','hospital_id','icu_id'}

        corr_t = get_corr_target(df, true_num_cols, binary_cols, cat_cols)
        all_num_feats = [c for c in corr_t[corr_t['type']!='categorical']['feature'].tolist()
                         if c in df.columns and c not in ID_COLS_C]

        tab_heat, tab_pairs = st.tabs(["🗺️ Heatmap","📋 Highly Correlated Pairs"])

        with tab_heat:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.4rem;'>"
                "This heatmap shows Pearson correlations between the top numeric features associated with hospital_death. "
                "Strong correlations may indicate redundant information or groups of related clinical measurements."
                "</p>", unsafe_allow_html=True
            )
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "High correlation does not mean a feature should be removed automatically. "
                "Feature handling should depend on the model type, clinical interpretation, and validation performance."
                "</p>", unsafe_allow_html=True
            )
            wrn("APACHE mortality probability variables (apache_4a_hospital_death_prob, apache_4a_icu_death_prob) are pre-calculated risk scores and may dominate correlation-based analysis. They should be reviewed separately before modeling.")

            n_feat = st.slider("Number of top target-associated numeric features to include", 10, len(all_num_feats), 20, key='heat_n')
            sel_cols = [c for c in corr_t[corr_t['type']!='categorical'].head(n_feat)['feature'].tolist()
                        if c in df.columns and c not in ID_COLS_C]
            corr_mat = df[sel_cols].corr()
            z = corr_mat.values.tolist()
            for i in range(len(z)):
                for j in range(i+1, len(z[i])):
                    z[i][j] = None
            fig = go.Figure(go.Heatmap(
                z=z,
                x=corr_mat.columns.tolist(),
                y=corr_mat.index.tolist(),
                colorscale='RdBu_r', zmid=0, zmin=-1, zmax=1,
                colorbar=dict(title='Pearson r'),
                hoverongaps=False
            ))
            fig.update_layout(**PT, height=max(500, n_feat*22), xaxis_tickangle=-50)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Only the lower triangle is shown to avoid duplicate correlations. Red indicates positive correlation, blue indicates negative correlation, and stronger color intensity indicates stronger correlation.")

        with tab_pairs:
            st.markdown(
                "<p style='color:#6a5a4a; font-size:.87rem; margin-bottom:.8rem;'>"
                "This table lists feature pairs with high absolute Pearson correlation. "
                "Highly correlated features can indicate redundancy, repeated measurements, or clinically related variables. "
                "Highly correlated pairs may provide partially overlapping information and should be reviewed before modeling, "
                "especially for linear models."
                "</p>", unsafe_allow_html=True
            )
            n_pairs = st.slider("Use top N target-associated numeric features for pair analysis", 10, len(all_num_feats), 30, key='pairs_n')
            pair_cols = [c for c in corr_t[corr_t['type']!='categorical'].head(n_pairs)['feature'].tolist()
                         if c in df.columns and c not in ID_COLS_C]

            # compute both signed and absolute correlations
            corr_signed = df[pair_cols].corr()
            corr_abs = corr_signed.abs()
            upper_mask = np.triu(np.ones(corr_abs.shape), k=1).astype(bool)
            upper_abs = corr_abs.where(upper_mask)
            pairs_list = []
            for col_a in upper_abs.columns:
                for col_b in upper_abs.index:
                    val = upper_abs.loc[col_b, col_a]
                    if pd.notnull(val):
                        signed_val = round(float(corr_signed.loc[col_b, col_a]), 4)
                        pairs_list.append({
                            'Feature A': col_a,
                            'Feature B': col_b,
                            'Pearson r': signed_val,
                            'Absolute Pearson r': round(abs(signed_val), 4),
                            '⚠️ Leakage': '⚠️' if col_a in LEAKAGE_COLS or col_b in LEAKAGE_COLS else ''
                        })
            pairs = pd.DataFrame(pairs_list).sort_values('Absolute Pearson r', ascending=False)

            thr_min, thr_max = st.slider(
                "Show feature pairs with absolute Pearson correlation between",
                min_value=0.0, max_value=1.0,
                value=(0.8, 1.0), step=0.05, key='pair_thr'
            )
            strong = pairs[
                (pairs['Absolute Pearson r'] >= thr_min) & (pairs['Absolute Pearson r'] <= thr_max)
            ].reset_index(drop=True)
            st.caption(f"{len(strong)} feature pairs have |r| between {thr_min:.2f} and {thr_max:.2f}. These pairs may contain overlapping information and should be reviewed before modeling.")
            st.dataframe(strong, use_container_width=True)
            ins("Highly correlated feature groups, such as lactate measurements or GCS components, may provide overlapping information. One representative feature or a clinically meaningful combined score can be considered during feature engineering.")

        concl("Feature Correlations",[
            "Several clinical measurement groups show strong internal correlation, suggesting overlapping information.",
            "Lactate-related variables are highly correlated; one representative feature may be sufficient depending on the modeling approach.",
            "APACHE hospital and ICU mortality probability variables are nearly identical and should be reviewed as potential leakage or derived-risk features.",
            "GCS motor, verbal, and eye scores are correlated components of the overall consciousness assessment.",
            "Invasive blood pressure measurements are correlated and may provide partially redundant information.",
            "Highly correlated features should be reviewed before modeling, especially for linear models, but they should not be removed automatically.",
        ])

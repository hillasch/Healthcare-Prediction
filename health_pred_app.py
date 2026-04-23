import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
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
.sec-sub   { color:#6a5a4a; font-size:.9rem; margin-bottom:1.2rem; font-style:italic; }
.insight   { background:#fff8f0; border-left:4px solid #c84b2f; border-radius:0 6px 6px 0; padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; color:#2a1a0a; }
.warn      { background:#fffff0; border-left:4px solid #c8a02f; border-radius:0 6px 6px 0; padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; }
.concl     { background:#1c1c1c; color:#e8e0d0; border-radius:8px; padding:1rem 1.3rem; margin-top:1rem; font-size:.86rem; line-height:1.7; }
.concl h4  { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em; color:#e07a5f; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

TARGET = "hospital_death"
C1, C2 = "#2f6bc8", "#c84b2f"
PT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(247,244,239,.4)", font_family="Fraunces", font_color="#1a1a1a")

def sh(label, title, sub=""):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)
    if sub: st.markdown(f'<p class="sec-sub">{sub}</p>', unsafe_allow_html=True)

def ins(t):    st.markdown(f'<div class="insight">💡 {t}</div>', unsafe_allow_html=True)
def wrn(t):    st.markdown(f'<div class="warn">⚠️ {t}</div>', unsafe_allow_html=True)
def divider(): st.markdown("<hr style='border:none;border-top:1px solid #e0d8cc;margin:1.5rem 0'>", unsafe_allow_html=True)
def concl(title, bullets):
    li = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f'<div class="concl"><h4>CONCLUSION — {title.upper()}</h4><ul>{li}</ul></div>', unsafe_allow_html=True)

def is_binary(s):
    v = s.dropna().unique()
    return set(v).issubset({0, 1}) and len(v) == 2

@st.cache_data
def split_cols(df):
    num = df.select_dtypes(include=np.number).columns.drop(TARGET, errors='ignore').tolist()
    cat = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    bn  = [c for c in num if is_binary(df[c])]
    tn  = [c for c in num if c not in bn]
    return num, cat, bn, tn

@st.cache_data
def get_miss(df):
    m = pd.DataFrame({'missing': df.isnull().sum(), 'pct': (df.isnull().mean() * 100).round(2)})
    return m[m['missing'] > 0].sort_values('pct', ascending=False)

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

PAGES = ["① Upload & Overview","② Missing Values","③ Numeric Distributions",
         "④ Outliers","⑤ Categorical Columns","⑥ Feature vs Target","⑦ Correlations"]

with st.sidebar:
    st.markdown("## 🏥 ICU EDA")
    st.markdown("*Hospital Death Prediction*")
    st.divider()
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Upload your CSV on step ① to unlock all pages.")

# ══ ① UPLOAD ══════════════════════════════════════════════════════════════════
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
    c1.metric("Rows", f"{df.shape[0]:,}"); c2.metric("Columns", df.shape[1])
    c3.metric("Numeric", len(true_num_cols)); c4.metric("Binary (0/1)", len(binary_cols)); c5.metric("Categorical", len(cat_cols))
    divider()
    dups = int(df.duplicated().sum())
    ins("No duplicate rows.") if dups == 0 else wrn(f"{dups:,} duplicate rows found.")
    st.subheader("Target balance — `hospital_death`")
    vc  = df[TARGET].value_counts().sort_index()
    pct = (df[TARGET].value_counts(normalize=True).sort_index() * 100).round(2)
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.dataframe(pd.DataFrame({'count': vc, 'percent': pct}), use_container_width=True)
    with col_b:
        vals = [float(pct.get(0,0)), float(pct.get(1,0))]
        fig = go.Figure(go.Bar(x=['Survived (0)','Died (1)'], y=vals, marker_color=[C1,C2],
                               text=[f"{v:.1f}%" for v in vals], textposition='outside'))
        fig.update_layout(**PT, height=270, showlegend=False, yaxis=dict(title='%', range=[0,110]))
        st.plotly_chart(fig, use_container_width=True)
    dr = float(pct.get(1, 0))
    wrn(f"Target is imbalanced — only {dr:.1f}% deaths. Consider class weights or SMOTE when modelling.")
    divider()
    st.subheader("Column info")
    info = pd.DataFrame({'dtype': df.dtypes.astype(str), 'nulls': df.isnull().sum(),
                         'null%': (df.isnull().mean()*100).round(2), 'unique': df.nunique()})
    st.dataframe(info.style.background_gradient(subset=['null%'], cmap='Oranges'), use_container_width=True)
    concl("Overview",[
        f"{df.shape[0]:,} rows, {df.shape[1]} columns after dropping IDs",
        f"Target is imbalanced: {dr:.1f}% deaths vs {100-dr:.1f}% survivors",
        f"{len(true_num_cols)} continuous, {len(binary_cols)} binary, {len(cat_cols)} categorical features",
        "No duplicate rows" if dups==0 else f"{dups} duplicate rows detected",
    ])

elif 'df' not in st.session_state:
    st.warning("👈 Go to **① Upload & Overview** first."); st.stop()

else:
    df = st.session_state['df']
    num_cols, cat_cols, binary_cols, true_num_cols = split_cols(df)
    miss = get_miss(df)

    # ══ ② MISSING VALUES ══════════════════════════════════════════════════════
    if page == PAGES[1]:
        sh("STEP 02","Missing Values","Where the gaps are, how they cluster, and what they mean.")
        sparse_pct = float((df.isnull().sum(axis=1) > df.shape[1]*.5).mean()*100)
        c1,c2,c3 = st.columns(3)
        c1.metric("Columns with missing", len(miss))
        c2.metric("Columns >50% missing", int((miss['pct']>50).sum()))
        c3.metric("Rows >50% missing", f"{sparse_pct:.1f}%")
        divider()
        bins_=[0,5,20,50,80,100]; labels_=['<5%','5–20%','20–50%','50–80%','>80%']
        miss2 = miss.copy(); miss2['band'] = pd.cut(miss2['pct'], bins=bins_, labels=labels_)
        bc = miss2['band'].value_counts().sort_index().reset_index(); bc.columns=['band','count']
        fig = go.Figure(go.Bar(x=bc['band'].astype(str).tolist(), y=bc['count'].tolist(),
                               marker_color=[C1,'#5a9ad4','#c8a02f','#d4752f',C2]))
        fig.update_layout(**PT, height=260, showlegend=False, xaxis_title='Missing % band', yaxis_title='# columns')
        st.plotly_chart(fig, use_container_width=True)
        ins(f"**{int((miss['pct']>50).sum())} columns** >50% missing → drop. **{int(((miss['pct']>5)&(miss['pct']<=20)).sum())} columns** 5–20% missing → impute.")
        divider()
        st.subheader("Top 40 columns by missing %")
        top40 = miss.head(40).copy()
        colors = [C2 if p>50 else C1 for p in top40['pct'].tolist()]
        fig2 = go.Figure(go.Bar(x=top40['pct'].tolist()[::-1], y=top40.index.tolist()[::-1],
                                orientation='h', marker_color=colors[::-1]))
        fig2.add_vline(x=50, line_dash='dash', line_color='red', annotation_text='50%')
        fig2.add_vline(x=20, line_dash='dash', line_color='orange', annotation_text='20%')
        fig2.update_layout(**PT, height=max(400,len(top40)*22), xaxis_title='Missing %', yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)
        divider()
        st.subheader("Sparse row analysis")
        sparse_mask = df.isnull().sum(axis=1) > df.shape[1]*.5
        dr_sp = float(df[sparse_mask][TARGET].mean()*100)
        dr_nm = float(df[~sparse_mask][TARGET].mean()*100)
        c1,c2,c3 = st.columns(3)
        c1.metric("Sparse rows", f"{int(sparse_mask.sum()):,}")
        c2.metric("Death rate — sparse", f"{dr_sp:.1f}%")
        c3.metric("Death rate — normal", f"{dr_nm:.1f}%")
        (ins if abs(dr_sp-dr_nm)<3 else wrn)("Sparse rows have nearly identical death rate to normal rows — missingness is random." if abs(dr_sp-dr_nm)<3 else f"Sparse rows die at {dr_sp:.1f}% vs {dr_nm:.1f}% — consider flagging them.")
        divider()
        st.subheader("Missingness correlation — co-missing column pairs")
        high_miss_cols = miss[miss['pct']>50].index.tolist()
        if high_miss_cols:
            miss_corr = df[high_miss_cols].isnull().corr()
            upper = miss_corr.where(np.triu(np.ones(miss_corr.shape),k=1).astype(bool))
            pairs = (upper.stack().reset_index()
                         .rename(columns={'level_0':'col_A','level_1':'col_B',0:'correlation'})
                         .sort_values('correlation', ascending=False))
            thr = st.slider("Show pairs with correlation ≥", 0.7, 1.0, 0.9, 0.05)
            st.dataframe(pairs[pairs['correlation']>=thr].reset_index(drop=True).head(40), use_container_width=True)
            ins(f"{len(pairs[pairs['correlation']>=thr])} pairs ≥ {thr} — columns go missing together (same clinical procedure).")
        concl("Missing Values",[
            f"{len(miss)} of {df.shape[1]} columns have missing values",
            f"{int((miss['pct']>50).sum())} columns >50% missing — mostly h1_ measurements → drop as a group",
            "Missing data follows clinical patterns: invasive BP, arterial blood gas, blood panels go missing together",
            f"Sparse rows ({int(sparse_mask.sum()):,}, {sparse_pct:.1f}%) similar death rate → no special treatment needed",
        ])

    # ══ ③ NUMERIC DISTRIBUTIONS ═══════════════════════════════════════════════
    elif page == PAGES[2]:
        sh("STEP 03","Numeric Distributions","Overview of all columns + deep-dive into any single column.")
        tab_ov, tab_drill = st.tabs(["📊 All columns overview","🔍 Single column drill-down"])

        with tab_ov:
            desc = df[true_num_cols].describe().T
            desc['skew']     = df[true_num_cols].skew()
            desc['kurtosis'] = df[true_num_cols].kurtosis()
            desc['missing%'] = (df[true_num_cols].isnull().mean()*100).round(2)
            high_skew = desc[desc['skew'].abs()>2].sort_values('skew',key=abs,ascending=False)
            c1,c2 = st.columns(2)
            c1.metric("Highly skewed (|skew|>2)", len(high_skew))
            c2.metric("Total numeric columns", len(true_num_cols))
            st.subheader("Skew — all numeric columns")
            skew_vals = desc['skew'].dropna().sort_values(key=abs, ascending=False)
            fig = go.Figure(go.Bar(x=skew_vals.index.tolist(), y=skew_vals.values.tolist(),
                                   marker_color=[C2 if abs(v)>2 else C1 for v in skew_vals.values.tolist()]))
            fig.add_hline(y=2,  line_dash='dash', line_color='red')
            fig.add_hline(y=-2, line_dash='dash', line_color='red')
            fig.update_layout(**PT, height=320, xaxis_tickangle=-45, xaxis_title='', yaxis_title='Skewness')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Top skewed columns")
            st.dataframe(high_skew[['mean','std','min','max','skew','missing%']].head(20)
                         .style.background_gradient(subset=['skew'], cmap='RdBu_r'), use_container_width=True)
            divider()
            st.subheader("Binary columns — % positive (1) values")
            bin_sum = pd.DataFrame({'pct_ones':(df[binary_cols].mean()*100).round(2),
                                    'missing%':(df[binary_cols].isnull().mean()*100).round(2)}
                                   ).sort_values('pct_ones',ascending=False).reset_index()
            bin_sum.columns = ['feature','pct_ones','missing%']
            fig_b = go.Figure(go.Bar(x=bin_sum['feature'].tolist(), y=bin_sum['pct_ones'].tolist(), marker_color=C2))
            fig_b.update_layout(**PT, height=300, xaxis_tickangle=-35, xaxis_title='', yaxis_title='% value = 1')
            st.plotly_chart(fig_b, use_container_width=True)

        with tab_drill:
            col_sel = st.selectbox("Select column", true_num_cols, key='dist_col')
            series  = df[col_sel].dropna()
            bins_n  = st.slider("Histogram bins", 10, 150, 40, key='dist_bins')
            c1,c2,c3,c4,c5 = st.columns(5)
            c1.metric("Mean",    f"{float(series.mean()):.4g}")
            c2.metric("Median",  f"{float(series.median()):.4g}")
            c3.metric("Std",     f"{float(series.std()):.4g}")
            c4.metric("Skew",    f"{float(series.skew()):.3f}")
            c5.metric("Missing", f"{df[col_sel].isnull().mean()*100:.1f}%")
            fig = go.Figure(go.Histogram(x=series.tolist(), nbinsx=bins_n, marker_color=C1, opacity=0.85))
            fig.update_layout(**PT, height=320, showlegend=False, xaxis_title=col_sel, yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
            st.subheader(f"`{col_sel}` by survival outcome")
            q_lo,q_hi = float(df[col_sel].quantile(.01)), float(df[col_sel].quantile(.99))
            pf = df[[col_sel,TARGET]].copy(); pf[col_sel] = pf[col_sel].clip(q_lo,q_hi)
            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=pf[pf[TARGET]==0][col_sel].dropna().tolist(), name='Survived', marker_color=C1, boxmean=True))
            fig2.add_trace(go.Box(y=pf[pf[TARGET]==1][col_sel].dropna().tolist(), name='Died',     marker_color=C2, boxmean=True))
            fig2.update_layout(**PT, height=320, yaxis_title=col_sel)
            st.plotly_chart(fig2, use_container_width=True)
            sk = float(series.skew())
            if sk>2:    wrn(f"Skew = {sk:.2f} — consider log1p transform.")
            elif sk<-2: wrn(f"Skew = {sk:.2f} — left-tailed distribution.")
            else:       ins(f"Skew = {sk:.2f} — approximately symmetric.")
            if col_sel=='pre_icu_los_days':
                wrn(f"pre_icu_los_days has {int((df[col_sel]<0).sum())} negative values — impossible clinically.")

        concl("Distributions",[
            f"{len(high_skew)} numeric columns have |skew|>2 — candidates for log transform",
            "pre_icu_los_days has negative minimum values — data quality issue",
            "SpO2 columns negatively skewed (most patients near 100%) — clinically expected",
            "Binary: ventilated_apache (32%), diabetes_mellitus (22%), elective_surgery (18%) most common",
        ])

    # ══ ④ OUTLIERS ════════════════════════════════════════════════════════════
    elif page == PAGES[3]:
        sh("STEP 04","Outliers","IQR-based detection — all columns overview + per-column inspection.")
        out_series = get_outlier_counts(df, true_num_cols)
        out_pct    = (out_series / len(df) * 100).round(2)
        out_df     = pd.DataFrame({'column': out_series.index, 'outliers': out_series.values,
                                   'pct_of_rows': out_pct.values})
        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single column drill-down"])

        with tab_ov:
            top30 = out_df.head(30).copy()
            fig = go.Figure(go.Bar(x=top30['column'].tolist(), y=top30['pct_of_rows'].tolist(),
                                   marker_color=C2,
                                   text=[f"{v:.1f}%" for v in top30['pct_of_rows'].tolist()],
                                   textposition='outside'))
            fig.update_layout(**PT, height=340, xaxis_tickangle=-45, xaxis_title='', yaxis_title='% rows flagged',
                              yaxis=dict(range=[0, float(top30['pct_of_rows'].max())*1.25]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(out_df.head(30), use_container_width=True)
            ins("In ICU data most outliers are real — critically ill patients have extreme values by definition.")

        with tab_col:
            col_sel = st.selectbox("Select column", true_num_cols, key='out_col')
            s = df[col_sel].dropna()
            if len(s) > 0:
                Q1,Q3 = float(s.quantile(.25)), float(s.quantile(.75)); IQR = Q3-Q1
                lo,hi = Q1-1.5*IQR, Q3+1.5*IQR
                outlier_mask = df[col_sel].notna() & ((df[col_sel]<lo)|(df[col_sel]>hi))
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Outliers (IQR)", f"{int(outlier_mask.sum()):,}")
                c2.metric("% of rows",      f"{outlier_mask.mean()*100:.1f}%")
                c3.metric("Lower fence",    f"{lo:.3g}")
                c4.metric("Upper fence",    f"{hi:.3g}")
                fig = go.Figure()
                fig.add_trace(go.Box(y=df[df[TARGET]==0][col_sel].dropna().tolist(), name='Survived', marker_color=C1, boxmean=True))
                fig.add_trace(go.Box(y=df[df[TARGET]==1][col_sel].dropna().tolist(), name='Died',     marker_color=C2, boxmean=True))
                fig.update_layout(**PT, height=380, yaxis_title=col_sel)
                st.plotly_chart(fig, use_container_width=True)
                if int(outlier_mask.sum()) > 0:
                    st.subheader(f"Outlier rows for `{col_sel}`")
                    st.dataframe(df[outlier_mask].head(100), use_container_width=True)
                else:
                    ins("No outliers found for this column.")

        concl("Outliers",[
            "GCS columns: high outlier count but valid — patients at extremes of consciousness",
            "pre_icu_los_days: negative values are impossible clinically — fix before modelling",
            "Creatinine, BUN, bilirubin: extreme high values are real organ failure — do not remove",
            "apache_4a_death_prob: IQR flags due to skewed distribution, not errors",
        ])

    # ══ ⑤ CATEGORICAL ═════════════════════════════════════════════════════════
    elif page == PAGES[4]:
        sh("STEP 05","Categorical Columns","Summary across all columns + drill into any single column.")
        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single column drill-down"])

        with tab_ov:
            cat_summary = pd.DataFrame({
                'unique':    df[cat_cols].nunique(),
                'missing%':  (df[cat_cols].isnull().mean()*100).round(2),
                'top_value': [str(df[c].value_counts().index[0]) if df[c].dropna().shape[0]>0 else 'N/A' for c in cat_cols],
                'top_freq%': [round(df[c].value_counts(normalize=True).iloc[0]*100,1) if df[c].dropna().shape[0]>0 else 0 for c in cat_cols],
            }).sort_values('unique', ascending=False)
            st.dataframe(cat_summary, use_container_width=True)
            ins("High top_freq% (>90%) = low-variance column — may not help the model.")

        with tab_col:
            col_sel = st.selectbox("Select column", cat_cols, key='cat_col')
            top_n   = st.slider("Top N values", 3, 30, 15, key='cat_n')
            vc = df[col_sel].value_counts(dropna=False).head(top_n).reset_index()
            vc.columns = ['value','count']; vc['value'] = vc['value'].astype(str)
            c_left, c_right = st.columns([2,1])
            with c_left:
                fig = go.Figure(go.Bar(x=vc['count'].tolist(), y=vc['value'].tolist(),
                                       orientation='h', marker_color=C1))
                fig.update_layout(**PT, height=max(300,top_n*28), xaxis_title='Count', yaxis_title='', yaxis_autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)
            with c_right:
                st.dataframe(vc, use_container_width=True)
            divider()
            st.subheader(f"Death rate by `{col_sel}`")
            top_cats = df[col_sel].value_counts().head(top_n).index
            sub = df[df[col_sel].isin(top_cats)].copy()
            ct  = pd.crosstab(sub[col_sel], sub[TARGET], normalize='index') * 100
            if 1 in ct.columns:
                ct = ct.rename(columns={0:'Survived%',1:'Died%'})
                ct['n'] = df[col_sel].value_counts()
                ct = ct.sort_values('Died%', ascending=False).reset_index()
                ct[col_sel] = ct[col_sel].astype(str)
                fig2 = go.Figure(go.Bar(x=ct['Died%'].tolist(), y=ct[col_sel].tolist(),
                                        orientation='h', marker_color=C2,
                                        text=[f"{v:.1f}%" for v in ct['Died%'].tolist()],
                                        textposition='outside'))
                fig2.update_layout(**PT, height=max(300,len(ct)*30),
                                   xaxis_title='Death rate %', yaxis_title='', yaxis_autorange='reversed',
                                   xaxis=dict(range=[0, float(ct['Died%'].max())*1.3]))
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(ct, use_container_width=True)

        concl("Categorical Columns",[
            "icu_admit_source: OR/Recovery 3.7% vs Other ICU 14.4% — 4x difference, strong feature",
            "apache_3j_bodysystem: Sepsis 15.8% vs Metabolic 1.5% — strongest categorical signal",
            "hospital_admit_source: Step-Down Unit 18.8% — tracks patient deterioration trend",
            "gender: 8.8% vs 8.4% — nearly no signal",
            "apache_2_bodysystem: inconsistent casing needs cleaning before encoding",
        ])

    # ══ ⑥ FEATURE VS TARGET ═══════════════════════════════════════════════════
    elif page == PAGES[5]:
        sh("STEP 06","Feature vs Target","Which features separate survivors from non-survivors?")
        tab_num, tab_bin, tab_cat, tab_corr = st.tabs([
            "📈 Numeric — mean diff","🔢 Binary — death rates",
            "🏷️ Categorical — death rates","📐 Correlation with target"])

        with tab_num:
            gm = get_group_means(df, true_num_cols)
            st.subheader("All numeric features — mean by group")
            st.dataframe(gm.style.background_gradient(subset=['abs_diff'], cmap='Reds'), use_container_width=True)
            divider()
            st.subheader("Top 12 features — box plots")
            top12 = gm.head(12).index.tolist()
            for row_i in range(0, len(top12), 3):
                row_cols = st.columns(3)
                for ci, col in enumerate(top12[row_i:row_i+3]):
                    with row_cols[ci]:
                        q_lo,q_hi = float(df[col].quantile(.01)), float(df[col].quantile(.99))
                        pf = df[[col,TARGET]].copy(); pf[col] = pf[col].clip(q_lo,q_hi)
                        fig = go.Figure()
                        fig.add_trace(go.Box(y=pf[pf[TARGET]==0][col].dropna().tolist(), name='Survived', marker_color=C1, showlegend=False))
                        fig.add_trace(go.Box(y=pf[pf[TARGET]==1][col].dropna().tolist(), name='Died',     marker_color=C2, showlegend=False))
                        fig.update_layout(**PT, height=260, title=dict(text=col, font_size=11), margin=dict(t=35,b=10))
                        st.plotly_chart(fig, use_container_width=True)
            divider()
            st.subheader("Drill into a single column")
            col_sel = st.selectbox("Column", true_num_cols, key='fvt_num')
            c1,c2 = st.columns(2)
            c1.metric("Mean — Survived", f"{float(df[df[TARGET]==0][col_sel].mean()):.3g}")
            c2.metric("Mean — Died",     f"{float(df[df[TARGET]==1][col_sel].mean()):.3g}")
            q_lo,q_hi = float(df[col_sel].quantile(.01)), float(df[col_sel].quantile(.99))
            pf = df[[col_sel,TARGET]].copy(); pf[col_sel] = pf[col_sel].clip(q_lo,q_hi)
            fig = go.Figure()
            fig.add_trace(go.Box(y=pf[pf[TARGET]==0][col_sel].dropna().tolist(), name='Survived', marker_color=C1, boxmean=True))
            fig.add_trace(go.Box(y=pf[pf[TARGET]==1][col_sel].dropna().tolist(), name='Died',     marker_color=C2, boxmean=True))
            fig.update_layout(**PT, height=340, yaxis_title=col_sel)
            st.plotly_chart(fig, use_container_width=True)

        with tab_bin:
            rows = []
            for col in binary_cols:
                g = df.groupby(col)[TARGET].mean() * 100
                if 0.0 in g.index and 1.0 in g.index:
                    rows.append({'feature':col, 'death_rate_0':round(float(g[0.0]),2),
                                 'death_rate_1':round(float(g[1.0]),2),
                                 'difference':round(float(g[1.0]-g[0.0]),2),
                                 'pct_ones':round(float(df[col].mean()*100),2)})
            bin_df = pd.DataFrame(rows).sort_values('difference', key=abs, ascending=False)
            fig = go.Figure(go.Bar(x=bin_df['feature'].tolist(), y=bin_df['difference'].tolist(),
                                   marker_color=[C2 if v>0 else C1 for v in bin_df['difference'].tolist()],
                                   text=[f"{v:+.1f}pp" for v in bin_df['difference'].tolist()],
                                   textposition='outside'))
            fig.add_hline(y=0, line_color='black', line_width=1)
            max_diff = float(bin_df['difference'].abs().max())
            fig.update_layout(**PT, height=340, xaxis_tickangle=-40,
                              xaxis_title='', yaxis_title='Death rate difference (pp)',
                              yaxis=dict(range=[-max_diff*1.4, max_diff*1.4]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bin_df, use_container_width=True)
            ins("Negative = protective (elective_surgery, apache_post_operative).")

        with tab_cat:
            col_sel = st.selectbox("Categorical column", cat_cols, key='fvt_cat')
            top_n   = st.slider("Top N categories", 3, 30, 15, key='fvt_catn')
            top_cats = df[col_sel].value_counts().head(top_n).index
            sub = df[df[col_sel].isin(top_cats)].copy()
            ct  = pd.crosstab(sub[col_sel], sub[TARGET], normalize='index') * 100
            if 1 in ct.columns:
                ct = ct.rename(columns={0:'Survived%',1:'Died%'})
                ct['n'] = df[col_sel].value_counts()
                ct = ct.sort_values('Died%', ascending=False).reset_index()
                ct[col_sel] = ct[col_sel].astype(str)
                fig = go.Figure(go.Bar(x=ct['Died%'].tolist(), y=ct[col_sel].tolist(),
                                       orientation='h', marker_color=C2,
                                       text=[f"{v:.1f}%" for v in ct['Died%'].tolist()],
                                       textposition='outside'))
                fig.update_layout(**PT, height=max(300,len(ct)*30),
                                  xaxis_title='Death rate %', yaxis_title='', yaxis_autorange='reversed',
                                  xaxis=dict(range=[0, float(ct['Died%'].max())*1.3]))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ct, use_container_width=True)

        with tab_corr:
            corr_t = get_corr_target(df, true_num_cols, binary_cols, cat_cols)
            st.subheader("All features ranked by correlation with `hospital_death`")
            st.caption("Numeric & binary: Pearson |r|   ·   Categorical: Cramér's V")
            top_n = st.slider("Show top N", 10, min(60, len(corr_t)), 30, key='corr_n')
            top   = corr_t.head(top_n).copy()
            color_map = {'numeric':C1,'binary':'#5a9ad4','categorical':C2}
            colors = [color_map.get(str(t), C1) for t in top['type'].tolist()]
            fig = go.Figure(go.Bar(x=top['abs_corr'].tolist(), y=top['feature'].tolist(),
                                   orientation='h', marker_color=colors,
                                   text=[f"{v:.3f}" for v in top['abs_corr'].tolist()],
                                   textposition='outside'))
            fig.update_layout(**PT, height=max(400,top_n*22), yaxis_autorange='reversed',
                              xaxis_title='|correlation|', yaxis_title='',
                              xaxis=dict(range=[0, float(top['abs_corr'].max())*1.25]))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(top, use_container_width=True)
            ins("apache_4a_hospital_death_prob (r≈0.31) is a pre-calculated death probability — consider leakage risk.")

        concl("Feature vs Target",[
            "Strongest numeric: d1_lactate_min (r=0.40), gcs_motor_apache (r=0.28)",
            "Strongest binary: gcs_unable_apache (+14.9pp), ventilated_apache (+13.7pp), intubated_apache (+13.5pp)",
            "Strongest categorical: apache_3j_bodysystem, icu_admit_source",
            "Protective binary: elective_surgery (−6.8pp), apache_post_operative (−5.9pp)",
            "d1_ features consistently outperform h1_ equivalents",
        ])

    # ══ ⑦ CORRELATIONS ════════════════════════════════════════════════════════
    elif page == PAGES[6]:
        sh("STEP 07","Feature Correlations","Which top features are redundant with each other?")
        corr_t    = get_corr_target(df, true_num_cols, binary_cols, cat_cols)
        top_feats = [c for c in corr_t[corr_t['type']!='categorical'].head(30)['feature'].tolist() if c in df.columns]
        tab_heat, tab_pairs = st.tabs(["🗺️ Heatmap","📋 Redundant pairs"])

        with tab_heat:
            n_feat   = st.slider("Number of top features", 10, min(50,len(top_feats)), 30, key='heat_n')
            sel_cols = [c for c in corr_t[corr_t['type']!='categorical'].head(n_feat)['feature'].tolist() if c in df.columns]
            corr_mat = df[sel_cols].corr()
            mask_arr = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
            corr_display = corr_mat.values.tolist()
            for i in range(len(corr_display)):
                for j in range(len(corr_display[i])):
                    if mask_arr[i][j]:
                        corr_display[i][j] = None
            fig = go.Figure(go.Heatmap(z=corr_display, x=corr_mat.columns.tolist(),
                                        y=corr_mat.index.tolist(), colorscale='RdBu_r',
                                        zmid=0, zmin=-1, zmax=1, colorbar=dict(title='r')))
            fig.update_layout(**PT, height=max(500,n_feat*22), xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with tab_pairs:
            corr_mat2 = df[top_feats].corr().abs()
            upper = corr_mat2.where(np.triu(np.ones(corr_mat2.shape),k=1).astype(bool))
            pairs = (upper.stack().reset_index()
                         .rename(columns={'level_0':'col_A','level_1':'col_B',0:'abs_r'})
                         .sort_values('abs_r',ascending=False))
            thr   = st.slider("Show pairs with |r| ≥", 0.5, 1.0, 0.8, 0.05, key='pair_thr')
            strong = pairs[pairs['abs_r']>=thr].reset_index(drop=True)
            st.markdown(f"**{len(strong)} pairs with |r| ≥ {thr}** — consider keeping only one of each pair.")
            st.dataframe(strong, use_container_width=True)
            ins("Lactate group (d1/h1 min/max) all correlated — keep only d1_lactate_min. GCS components can be combined into gcs_total.")

        concl("Feature Correlations",[
            "Lactate group — all correlated, keep only d1_lactate_min",
            "apache_4a_hospital_death_prob ↔ apache_4a_icu_death_prob — nearly identical, keep one",
            "GCS motor/eyes/verbal — correlated, consider summing into gcs_total",
            "All invasive BP measurements — one representative is sufficient",
            "h1_inr_max ↔ d1_inr_max — correlation = 1.0, keep only d1_inr_max",
        ])

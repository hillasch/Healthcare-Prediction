import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="ICU EDA", page_icon="🏥", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Fraunces:wght@400;700;900&display=swap');
html, body, [class*="css"] { font-family: 'Fraunces', Georgia, serif; background:#f7f4ef; color:#1a1a1a; }
code, pre { font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="stSidebar"] { background:#1c1c1c; }
[data-testid="stSidebar"] * { color:#e8e0d0 !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid #e0d8cc; border-radius:6px; padding:.8rem; }
.sec-label { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.15em;
             color:#8a7a6a; text-transform:uppercase; margin-bottom:.2rem; }
.sec-title { font-family:'Fraunces',serif; font-weight:900; font-size:1.9rem;
             color:#1a1a1a; margin:0 0 .2rem 0; line-height:1.1; }
.sec-sub   { color:#6a5a4a; font-size:.9rem; margin-bottom:1.2rem; font-style:italic; }
.insight   { background:#fff8f0; border-left:4px solid #c84b2f; border-radius:0 6px 6px 0;
             padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; color:#2a1a0a; }
.warn      { background:#fffff0; border-left:4px solid #c8a02f; border-radius:0 6px 6px 0;
             padding:.7rem 1rem; margin:.5rem 0; font-size:.87rem; color:#2a2a0a; }
.concl     { background:#1c1c1c; color:#e8e0d0; border-radius:8px; padding:1rem 1.3rem;
             margin-top:1rem; font-size:.86rem; line-height:1.7; }
.concl h4  { font-family:'IBM Plex Mono',monospace; font-size:.68rem; letter-spacing:.12em;
             color:#e07a5f; margin-bottom:.5rem; }
</style>
""", unsafe_allow_html=True)

TARGET = "hospital_death"
C1, C2 = "#2f6bc8", "#c84b2f"
PT = dict(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)",
          plot_bgcolor="rgba(247,244,239,.4)", font_family="Fraunces", font_color="#1a1a1a")

# ── helpers ────────────────────────────────────────────────────────────────────
def sh(label, title, sub=""):
    st.markdown(f'<div class="sec-label">{label}</div>', unsafe_allow_html=True)
    st.markdown(f'<p class="sec-title">{title}</p>', unsafe_allow_html=True)
    if sub: st.markdown(f'<p class="sec-sub">{sub}</p>', unsafe_allow_html=True)

def ins(t):  st.markdown(f'<div class="insight">💡 {t}</div>', unsafe_allow_html=True)
def wrn(t):  st.markdown(f'<div class="warn">⚠️ {t}</div>', unsafe_allow_html=True)
def concl(title, bullets):
    li = "".join(f"<li>{b}</li>" for b in bullets)
    st.markdown(f'<div class="concl"><h4>CONCLUSION — {title.upper()}</h4><ul>{li}</ul></div>',
                unsafe_allow_html=True)
def divider(): st.markdown("<hr style='border:none;border-top:1px solid #e0d8cc;margin:1.5rem 0'>",
                           unsafe_allow_html=True)

def is_binary(s): v=s.dropna().unique(); return set(v).issubset({0,1}) and len(v)==2

@st.cache_data
def split_cols(df):
    num  = df.select_dtypes(include=np.number).columns.drop(TARGET,errors='ignore').tolist()
    cat  = df.select_dtypes(include=['object','category','bool']).columns.tolist()
    bn   = [c for c in num if is_binary(df[c])]
    tn   = [c for c in num if c not in bn]
    return num, cat, bn, tn

@st.cache_data
def get_miss(df):
    return pd.DataFrame({'missing':df.isnull().sum(),
                         'pct':(df.isnull().mean()*100).round(2)}
                        ).query('missing>0').sort_values('pct',ascending=False)

@st.cache_data
def get_outlier_counts(df, cols):
    res={}
    for c in cols:
        s=df[c].dropna(); Q1,Q3=s.quantile(.25),s.quantile(.75); IQR=Q3-Q1
        res[c]=int(((s<Q1-1.5*IQR)|(s>Q3+1.5*IQR)).sum())
    return pd.Series(res).sort_values(ascending=False)

@st.cache_data
def get_group_means(df, cols):
    gm = df[cols+[TARGET]].groupby(TARGET).mean().T
    gm.columns=['Survived (0)','Died (1)']
    gm['abs_diff']=(gm['Died (1)']-gm['Survived (0)']).abs()
    gm['rel_diff%']=((gm['Died (1)']-gm['Survived (0)'])/gm['Survived (0)'].replace(0,np.nan)*100).round(1)
    return gm

@st.cache_data
def get_corr_target(df, tn, bn, cat):
    pearson=(df[tn+bn].corrwith(df[TARGET]).abs()
             .reset_index().rename(columns={'index':'feature',0:'abs_corr'}))
    pearson['type']=pearson['feature'].apply(lambda x:'binary' if x in bn else 'numeric')
    rows=[]
    for col in cat:
        try:
            clean=df[[col,TARGET]].dropna(); ct=pd.crosstab(clean[col],clean[TARGET])
            chi2=chi2_contingency(ct)[0]; n=len(clean)
            rows.append({'feature':col,'abs_corr':round(np.sqrt(chi2/(n*(min(ct.shape)-1))),4),'type':'categorical'})
        except: pass
    return (pd.concat([pearson,pd.DataFrame(rows)]).sort_values('abs_corr',ascending=False)
              .reset_index(drop=True))

# ── sidebar ────────────────────────────────────────────────────────────────────
PAGES = ["① Upload & Overview","② Missing Values","③ Numeric Distributions",
         "④ Outliers","⑤ Categorical Columns","⑥ Feature vs Target","⑦ Correlations"]

with st.sidebar:
    st.markdown("## 🏥 ICU EDA")
    st.markdown("*Hospital Death Prediction*")
    st.divider()
    page = st.radio("", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("Each section mirrors your notebook exactly.\nUpload CSV on step ① to begin.")

# ══════════════════════════════════════════════════════════════════════════════
# ① UPLOAD & OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == PAGES[0]:
    sh("STEP 01","Upload & Overview","Load the dataset — shape, types, duplicates, target balance.")

    f = st.file_uploader("", type=["csv","xlsx"])
    if f:
        df = pd.read_csv(f) if f.name.endswith(".csv") else pd.read_excel(f)
        df = df.drop(columns=['patient_id','encounter_id'], errors='ignore')
        st.session_state['df'] = df
        st.success(f"✅ Loaded **{df.shape[0]:,} rows × {df.shape[1]} columns**")
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

    dups = df.duplicated().sum()
    ins("No duplicate rows.") if dups==0 else wrn(f"{dups:,} duplicate rows found.")

    # Target balance
    st.subheader("Target balance — `hospital_death`")
    vc  = df[TARGET].value_counts().sort_index()
    pct = df[TARGET].value_counts(normalize=True).sort_index()*100
    col_a, col_b = st.columns([1,2])
    with col_a:
        st.dataframe(pd.DataFrame({'count':vc,'percent':pct.round(2)}), use_container_width=True)
    with col_b:
        fig=px.bar(x=['Survived (0)','Died (1)'],y=pct.values,
                   color=['Survived (0)','Died (1)'],color_discrete_sequence=[C1,C2],
                   labels={'x':'','y':'%'},text=[f"{v:.1f}%" for v in pct.values],**PT)
        fig.update_layout(showlegend=False,height=260); fig.update_traces(textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

    dr = pct.get(1,0)
    wrn(f"Target is imbalanced — only {dr:.1f}% deaths. Consider class weights or SMOTE when modelling.")

    divider()

    # Column info table
    st.subheader("Column info")
    info=pd.DataFrame({'dtype':df.dtypes.astype(str),'nulls':df.isnull().sum(),
                       'null%':(df.isnull().mean()*100).round(2),'unique':df.nunique()})
    st.dataframe(info.style.background_gradient(subset=['null%'],cmap='Oranges'),
                 use_container_width=True)

    concl("Overview",[
        f"{df.shape[0]:,} rows, {df.shape[1]} columns after dropping patient/encounter IDs",
        f"Target is imbalanced: {dr:.1f}% deaths vs {100-dr:.1f}% survivors",
        f"{len(true_num_cols)} continuous numeric, {len(binary_cols)} binary, {len(cat_cols)} categorical features",
        "No duplicate rows" if dups==0 else f"{dups} duplicate rows detected",
    ])

elif 'df' not in st.session_state:
    st.warning("👈 Go to **① Upload & Overview** first."); st.stop()

else:
    df = st.session_state['df']
    num_cols, cat_cols, binary_cols, true_num_cols = split_cols(df)
    miss = get_miss(df)

    # ══════════════════════════════════════════════════════════════════════════
    # ② MISSING VALUES
    # ══════════════════════════════════════════════════════════════════════════
    if page == PAGES[1]:
        sh("STEP 02","Missing Values","Where the gaps are, how they cluster, and what they mean.")

        c1,c2,c3 = st.columns(3)
        c1.metric("Columns with missing", len(miss))
        c2.metric("Columns >50% missing", int((miss['pct']>50).sum()))
        sparse_pct=(df.isnull().sum(axis=1)>df.shape[1]*.5).mean()*100
        c3.metric("Rows >50% missing", f"{sparse_pct:.1f}%")

        divider()

        # Band summary
        bins_=[0,5,20,50,80,100]; labels_=['<5%','5–20%','20–50%','50–80%','>80%']
        miss2=miss.copy(); miss2['band']=pd.cut(miss2['pct'],bins=bins_,labels=labels_)
        bc=miss2['band'].value_counts().sort_index().reset_index(); bc.columns=['band','count']
        fig=px.bar(bc,x='band',y='count',color='band',
                   color_discrete_sequence=[C1,'#5a9ad4','#c8a02f','#d4752f',C2],
                   labels={'band':'Missing % band','count':'# columns'},**PT)
        fig.update_layout(showlegend=False,height=260)
        st.plotly_chart(fig, use_container_width=True)

        ins(f"**{int((miss['pct']>50).sum())} columns** have >50% missing (red) → drop these. "
            f"**{int(((miss['pct']>5)&(miss['pct']<=20)).sum())} columns** have 5–20% missing → impute with median.")

        divider()

        # Top 40 bar
        st.subheader("Top 40 columns by missing %")
        top40=miss.head(40)
        colors=[C2 if p>50 else C1 for p in top40['pct']]
        fig2=go.Figure(go.Bar(x=top40['pct'][::-1],y=top40.index[::-1],orientation='h',
                               marker_color=colors[::-1]))
        fig2.add_vline(x=50,line_dash='dash',line_color='red',annotation_text='50%')
        fig2.add_vline(x=20,line_dash='dash',line_color='orange',annotation_text='20%')
        fig2.update_layout(**PT,height=max(400,len(top40)*22),
                           xaxis_title='Missing %',yaxis_title='')
        st.plotly_chart(fig2, use_container_width=True)

        divider()

        # Sparse rows
        st.subheader("Sparse row analysis")
        sparse_mask=df.isnull().sum(axis=1)>df.shape[1]*.5
        dr_sp=df[sparse_mask][TARGET].mean()*100
        dr_nm=df[~sparse_mask][TARGET].mean()*100
        c1,c2,c3=st.columns(3)
        c1.metric("Sparse rows",f"{sparse_mask.sum():,}")
        c2.metric("Death rate — sparse",f"{dr_sp:.1f}%")
        c3.metric("Death rate — normal",f"{dr_nm:.1f}%")
        if abs(dr_sp-dr_nm)<3:
            ins("Sparse rows have nearly identical death rate to normal rows — missingness is random, no special treatment needed.")
        else:
            wrn(f"Sparse rows die at {dr_sp:.1f}% vs {dr_nm:.1f}% — consider flagging them as a feature.")

        divider()

        # Missingness correlation pairs
        st.subheader("Missingness correlation — highly correlated pairs")
        high_miss_cols=miss[miss['pct']>50].index.tolist()
        if high_miss_cols:
            miss_corr=df[high_miss_cols].isnull().corr()
            upper=miss_corr.where(np.triu(np.ones(miss_corr.shape),k=1).astype(bool))
            pairs=(upper.stack().reset_index()
                       .rename(columns={'level_0':'col_A','level_1':'col_B',0:'correlation'})
                       .sort_values('correlation',ascending=False))
            thr=st.slider("Show pairs with correlation ≥",0.7,1.0,0.9,0.05)
            strong=pairs[pairs['correlation']>=thr]
            st.dataframe(strong.head(40).reset_index(drop=True), use_container_width=True)
            ins(f"{len(strong)} pairs with missingness correlation ≥ {thr}. "
                "These columns go missing together — same clinical procedure (e.g. arterial line, blood panel).")

        concl("Missing Values",[
            f"{len(miss)} of {df.shape[1]} columns have missing values",
            f"{int((miss['pct']>50).sum())} columns >50% missing — mostly h1_ (first-hour) measurements → drop as a group",
            "Missing data follows clinical patterns: invasive BP, arterial blood gas, blood panels go missing together",
            f"Sparse rows ({sparse_mask.sum():,}, {sparse_pct:.1f}%) have similar death rate → no special treatment needed",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ③ NUMERIC DISTRIBUTIONS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[2]:
        sh("STEP 03","Numeric Distributions","Overview of all columns + deep-dive into any single column.")

        tab_overview, tab_drill = st.tabs(["📊 All columns overview", "🔍 Single column drill-down"])

        with tab_overview:
            desc=df[true_num_cols].describe().T
            desc['skew']=df[true_num_cols].skew()
            desc['kurtosis']=df[true_num_cols].kurtosis()
            desc['missing%']=(df[true_num_cols].isnull().mean()*100).round(2)
            high_skew=desc[desc['skew'].abs()>2].sort_values('skew',key=abs,ascending=False)

            c1,c2=st.columns(2)
            c1.metric("Highly skewed (|skew|>2)",len(high_skew))
            c2.metric("Total numeric columns",len(true_num_cols))

            st.subheader("Skew overview — all numeric columns")
            skew_df=(desc['skew'].reset_index().rename(columns={'index':'column',0:'skew'})
                     .sort_values('skew',key=abs,ascending=False))
            fig=px.bar(skew_df,x='column',y='skew',
                       color=skew_df['skew'].abs(),color_continuous_scale='RdBu_r',
                       labels={'column':'','skew':'Skewness'},**PT)
            fig.add_hline(y=2,line_dash='dash',line_color='red',annotation_text='threshold +2')
            fig.add_hline(y=-2,line_dash='dash',line_color='red',annotation_text='threshold -2')
            fig.update_layout(height=320,showlegend=False,xaxis_tickangle=-45,
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Top skewed columns — full stats")
            st.dataframe(high_skew[['mean','std','min','max','skew','missing%']].head(20)
                         .style.background_gradient(subset=['skew'],cmap='RdBu_r'),
                         use_container_width=True)

            divider()

            st.subheader("Binary columns — % of positive (1) values")
            bin_sum=pd.DataFrame({'pct_ones':(df[binary_cols].mean()*100).round(2),
                                  'missing%':(df[binary_cols].isnull().mean()*100).round(2)}
                                 ).sort_values('pct_ones',ascending=False)
            fig_b=px.bar(bin_sum.reset_index(),x='index',y='pct_ones',
                         color_discrete_sequence=[C2],
                         labels={'index':'','pct_ones':'% with value = 1'},**PT)
            fig_b.update_layout(height=300,xaxis_tickangle=-35)
            st.plotly_chart(fig_b, use_container_width=True)

            ins("Columns with very low pct_ones (e.g. aids=0.09%) are rare conditions — limited statistical power.")

        with tab_drill:
            st.markdown("**Select any numeric column to explore in depth.**")
            col_sel=st.selectbox("Column", true_num_cols, key='dist_col')
            series=df[col_sel].dropna()
            bins_n=st.slider("Histogram bins",10,150,40, key='dist_bins')

            # Stats strip
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("Mean",   f"{series.mean():.4g}")
            c2.metric("Median", f"{series.median():.4g}")
            c3.metric("Std",    f"{series.std():.4g}")
            c4.metric("Skew",   f"{series.skew():.3f}")
            c5.metric("Missing",f"{df[col_sel].isnull().mean()*100:.1f}%")

            # Histogram
            fig=px.histogram(series,nbins=bins_n,color_discrete_sequence=[C1],
                             labels={'value':col_sel,'count':'Count'},**PT)
            fig.update_layout(height=320,showlegend=False,bargap=0.02)
            st.plotly_chart(fig, use_container_width=True)

            # Box split by target
            st.subheader(f"`{col_sel}` by survival outcome")
            q_lo,q_hi=df[col_sel].quantile(.01),df[col_sel].quantile(.99)
            plot_df=df[[col_sel,TARGET]].copy()
            plot_df[col_sel]=plot_df[col_sel].clip(q_lo,q_hi)
            fig2=px.box(plot_df,x=TARGET,y=col_sel,color=TARGET,
                        color_discrete_sequence=[C1,C2],
                        labels={TARGET:'','col_sel':col_sel},
                        category_orders={TARGET:[0,1]},**PT)
            fig2.update_layout(height=320,showlegend=False,
                               xaxis=dict(ticktext=['Survived','Died'],tickvals=[0,1]))
            st.plotly_chart(fig2, use_container_width=True)

            # Insight
            sk=series.skew()
            if sk>2:   wrn(f"Skew = {sk:.2f} (positive). Consider log1p transform before modelling.")
            elif sk<-2: wrn(f"Skew = {sk:.2f} (negative). Distribution is heavily left-tailed.")
            else:       ins(f"Skew = {sk:.2f} — within ±2, approximately symmetric.")

            if col_sel=='pre_icu_los_days':
                neg=(df[col_sel]<0).sum()
                wrn(f"pre_icu_los_days has {neg} negative values (min={df[col_sel].min():.2f}) — data quality issue.")

        concl("Distributions",[
            f"{len(high_skew)} numeric columns have |skew|>2 — candidates for log transform",
            "pre_icu_los_days contains negative values — impossible clinically, needs cleaning",
            "SpO2 columns are negatively skewed (most patients near 100%) — clinically expected, not errors",
            "Binary features: ventilated_apache (32%), diabetes_mellitus (22%), elective_surgery (18%) most common",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ④ OUTLIERS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[3]:
        sh("STEP 04","Outliers","IQR-based detection — overview across all columns + per-column inspection.")

        out_series=get_outlier_counts(df, true_num_cols)
        out_pct=(out_series/len(df)*100).round(2)
        out_df=pd.DataFrame({'outliers':out_series,'pct_of_rows':out_pct})

        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single column drill-down"])

        with tab_ov:
            st.subheader("Top 30 columns by outlier count (IQR method)")
            top30=out_df.head(30).reset_index().rename(columns={'index':'column'})
            fig=px.bar(top30,x='column',y='pct_of_rows',
                       color='pct_of_rows',color_continuous_scale='Reds',
                       labels={'column':'','pct_of_rows':'% rows flagged'},**PT)
            fig.update_layout(height=340,xaxis_tickangle=-45,coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(out_df.head(30).style.background_gradient(subset=['pct_of_rows'],cmap='Reds'),
                         use_container_width=True)

            ins("In ICU data most 'outliers' are real — critically ill patients have extreme values by definition. "
                "Only negative pre_icu_los_days values are likely data errors.")

        with tab_col:
            st.markdown("**Select a column to see its box plot and outlier rows.**")
            col_sel=st.selectbox("Column", true_num_cols, key='out_col')
            s=df[col_sel].dropna()
            Q1,Q3=s.quantile(.25),s.quantile(.75); IQR=Q3-Q1
            lo,hi=Q1-1.5*IQR, Q3+1.5*IQR
            outlier_mask=df[col_sel].notna()&((df[col_sel]<lo)|(df[col_sel]>hi))

            c1,c2,c3,c4=st.columns(4)
            c1.metric("Outliers (IQR)",f"{outlier_mask.sum():,}")
            c2.metric("% of rows",f"{outlier_mask.mean()*100:.1f}%")
            c3.metric("Lower fence",f"{lo:.3g}")
            c4.metric("Upper fence",f"{hi:.3g}")

            # Box plot
            fig=px.box(df[[col_sel,TARGET]].dropna(),y=col_sel,color=TARGET,
                       color_discrete_sequence=[C1,C2],
                       category_orders={TARGET:[0,1]},**PT)
            fig.update_layout(height=350,showlegend=True,
                              legend=dict(title='',itemsizeref=1),
                              xaxis=dict(ticktext=['Survived','Died'],tickvals=[0,1]))
            st.plotly_chart(fig, use_container_width=True)

            # Outlier rows
            if outlier_mask.sum()>0:
                st.subheader(f"Outlier rows for `{col_sel}`")
                st.dataframe(df[outlier_mask].head(100), use_container_width=True)
            else:
                ins("No outliers found for this column using the IQR method.")

        concl("Outliers",[
            "GCS columns have high outlier counts but values are valid — patients at extremes of consciousness",
            "pre_icu_los_days outliers include clinically impossible negative values — needs fixing before modelling",
            "Creatinine, BUN, bilirubin: extreme high values are real (organ failure) — do not remove",
            "apache_4a_death_prob: IQR flags are due to skewed distribution, not errors",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ⑤ CATEGORICAL COLUMNS
    # ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[4]:
        sh("STEP 05","Categorical Columns","Summary across all categories + drill into any single column.")

        tab_ov, tab_col = st.tabs(["📊 All columns overview","🔍 Single column drill-down"])

        with tab_ov:
            cat_summary=pd.DataFrame({
                'unique':    df[cat_cols].nunique(),
                'missing%':  (df[cat_cols].isnull().mean()*100).round(2),
                'top_value': [df[c].value_counts().index[0] if df[c].dropna().shape[0]>0 else 'N/A' for c in cat_cols],
                'top_freq%': [(df[c].value_counts(normalize=True).iloc[0]*100).round(1) if df[c].dropna().shape[0]>0 else 0 for c in cat_cols],
            }).sort_values('unique',ascending=False)
            st.dataframe(cat_summary, use_container_width=True)
            ins("High top_freq% (e.g. >90%) = low-variance column — may not help the model.")

        with tab_col:
            col_sel=st.selectbox("Column", cat_cols, key='cat_col')
            top_n=st.slider("Top N values",3,30,15,key='cat_n')
            vc=df[col_sel].value_counts(dropna=False).head(top_n).reset_index()
            vc.columns=['value','count']

            c1,c2=st.columns([2,1])
            with c1:
                fig=px.bar(vc,x='count',y='value',orientation='h',
                           color='count',color_continuous_scale=[[0,'#ddd'],[1,C1]],
                           labels={'value':'','count':'Count'},**PT)
                fig.update_layout(height=max(300,top_n*28),coloraxis_showscale=False,
                                  yaxis_autorange='reversed')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                st.dataframe(vc, use_container_width=True, height=max(300,top_n*28))

            # Death rate by category
            divider()
            st.subheader(f"Death rate by `{col_sel}` category")
            top_cats=df[col_sel].value_counts().head(top_n).index
            sub=df[df[col_sel].isin(top_cats)]
            ct=pd.crosstab(sub[col_sel],sub[TARGET],normalize='index')*100
            if 1 in ct.columns:
                ct=ct.rename(columns={0:'Survived%',1:'Died%'})
                ct['n']=df[col_sel].value_counts()
                ct=ct.sort_values('Died%',ascending=False).reset_index()
                fig2=px.bar(ct,x='Died%',y=col_sel,orientation='h',
                            color='Died%',color_continuous_scale=[[0,'#ddd'],[1,C2]],
                            labels={col_sel:'','Died%':'Death rate %'},**PT)
                fig2.update_layout(height=max(300,len(ct)*28),coloraxis_showscale=False,
                                   yaxis_autorange='reversed')
                st.plotly_chart(fig2, use_container_width=True)
                st.dataframe(ct, use_container_width=True)

        concl("Categorical Columns",[
            "icu_admit_source: OR/Recovery patients die at 3.7% vs Other ICU at 14.4% — 4x difference, strong feature",
            "apache_3j_bodysystem: Sepsis 15.8% vs Metabolic 1.5% — strongest categorical signal",
            "hospital_admit_source: Step-Down Unit 18.8% vs direct admits 6.7% — tracks patient deterioration trend",
            "gender: 8.8% vs 8.4% — nearly no signal, use cautiously",
            "apache_2_bodysystem: 'Undefined Diagnoses' vs 'Undefined diagnoses' — inconsistent casing, needs cleaning",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ⑥ FEATURE VS TARGET
    # ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[5]:
        sh("STEP 06","Feature vs Target","Which features separate survivors from non-survivors?")

        tab_num, tab_bin, tab_cat, tab_corr = st.tabs([
            "📈 Numeric — mean diff",
            "🔢 Binary — death rates",
            "🏷️ Categorical — death rates",
            "📐 Correlation with target",
        ])

        # ── Numeric ──────────────────────────────────────────────────────────
        with tab_num:
            gm=get_group_means(df, true_num_cols)
            top_diff=gm.sort_values('abs_diff',ascending=False)

            st.subheader("All numeric features — mean by survival group")
            st.dataframe(top_diff.style.background_gradient(subset=['abs_diff'],cmap='Reds'),
                         use_container_width=True)

            divider()
            st.subheader("Top 12 features — box plots (clipped to 1st–99th percentile)")
            top12=top_diff.head(12).index.tolist()
            cols_per_row=3
            for row_i in range(0,len(top12),cols_per_row):
                row_cols=st.columns(cols_per_row)
                for ci,col in enumerate(top12[row_i:row_i+cols_per_row]):
                    with row_cols[ci]:
                        q_lo,q_hi=df[col].quantile(.01),df[col].quantile(.99)
                        pf=df[[col,TARGET]].copy(); pf[col]=pf[col].clip(q_lo,q_hi)
                        fig=px.box(pf,x=TARGET,y=col,color=TARGET,
                                   color_discrete_sequence=[C1,C2],**PT)
                        fig.update_layout(height=280,showlegend=False,title=col,
                                          title_font_size=11,
                                          xaxis=dict(ticktext=['Survived','Died'],tickvals=[0,1]))
                        st.plotly_chart(fig, use_container_width=True)

            divider()
            st.subheader("Drill into a single numeric column")
            col_sel=st.selectbox("Column",true_num_cols,key='fvt_num')
            q_lo,q_hi=df[col_sel].quantile(.01),df[col_sel].quantile(.99)
            pf=df[[col_sel,TARGET]].copy(); pf[col_sel]=pf[col_sel].clip(q_lo,q_hi)
            c1,c2=st.columns(2)
            survived_mean=df[df[TARGET]==0][col_sel].mean()
            died_mean=df[df[TARGET]==1][col_sel].mean()
            c1.metric("Mean — Survived",f"{survived_mean:.3g}")
            c2.metric("Mean — Died",f"{died_mean:.3g}")
            fig=px.box(pf,x=TARGET,y=col_sel,color=TARGET,
                       color_discrete_sequence=[C1,C2],**PT)
            fig.update_layout(height=340,showlegend=False,
                              xaxis=dict(ticktext=['Survived','Died'],tickvals=[0,1]))
            st.plotly_chart(fig, use_container_width=True)

        # ── Binary ───────────────────────────────────────────────────────────
        with tab_bin:
            rows=[]
            for col in binary_cols:
                g=df.groupby(col)[TARGET].mean()*100
                if 0.0 in g.index and 1.0 in g.index:
                    rows.append({'feature':col,'death_rate_0':round(g[0.0],2),
                                 'death_rate_1':round(g[1.0],2),
                                 'difference':round(g[1.0]-g[0.0],2),
                                 'pct_ones':round(df[col].mean()*100,2)})
            bin_df=(pd.DataFrame(rows).sort_values('difference',key=abs,ascending=False))

            st.subheader("Death rate when feature = 1 vs = 0")
            fig=px.bar(bin_df,x='feature',y='difference',
                       color='difference',color_continuous_scale='RdBu_r',
                       labels={'feature':'','difference':'Death rate difference (pp)'},**PT)
            fig.add_hline(y=0,line_color='black',line_width=1)
            fig.update_layout(height=320,xaxis_tickangle=-40,coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(bin_df, use_container_width=True)

            ins("Positive difference = feature=1 means higher death risk. "
                "Negative (elective_surgery, apache_post_operative) = protective.")

        # ── Categorical ───────────────────────────────────────────────────────
        with tab_cat:
            col_sel=st.selectbox("Select categorical column", cat_cols, key='fvt_cat')
            top_n=st.slider("Top N categories",3,30,15,key='fvt_catn')
            top_cats=df[col_sel].value_counts().head(top_n).index
            sub=df[df[col_sel].isin(top_cats)]
            ct=pd.crosstab(sub[col_sel],sub[TARGET],normalize='index')*100
            if 1 in ct.columns:
                ct=ct.rename(columns={0:'Survived%',1:'Died%'})
                ct['n']=df[col_sel].value_counts()
                ct=ct.sort_values('Died%',ascending=False).reset_index()
                fig=px.bar(ct,x=col_sel,y='Died%',color='Died%',
                           color_continuous_scale=[[0,'#ddd'],[1,C2]],
                           labels={col_sel:'','Died%':'Death rate %'},
                           text=ct['Died%'].apply(lambda x:f"{x:.1f}%"),**PT)
                fig.update_layout(height=340,xaxis_tickangle=-35,coloraxis_showscale=False)
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(ct, use_container_width=True)

        # ── Correlation with target ───────────────────────────────────────────
        with tab_corr:
            corr_t=get_corr_target(df, true_num_cols, binary_cols, cat_cols)
            st.subheader("All features ranked by correlation with `hospital_death`")
            st.caption("Numeric & binary: Pearson |r|. Categorical: Cramér's V.")

            top_n=st.slider("Show top N",10,len(corr_t),30,key='corr_n')
            top=corr_t.head(top_n)
            fig=px.bar(top,x='abs_corr',y='feature',orientation='h',color='type',
                       color_discrete_map={'numeric':C1,'binary':'#5a9ad4','categorical':C2},
                       labels={'abs_corr':'|correlation|','feature':''},**PT)
            fig.update_layout(height=max(400,top_n*22),yaxis_autorange='reversed',legend_title='')
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(corr_t.head(top_n), use_container_width=True)

            ins("apache_4a_hospital_death_prob (r=0.31) is a pre-calculated death probability — consider leakage risk before including in model.")

        concl("Feature vs Target",[
            "Strongest numeric signals: d1_lactate_min (r=0.40), d1_lactate_max (r=0.40), gcs_motor_apache (r=0.28)",
            "Strongest binary signals: ventilated_apache (+13.7pp), intubated_apache (+13.5pp), gcs_unable_apache (+14.9pp)",
            "Strongest categorical signals: apache_3j_bodysystem (Sepsis 15.8% vs Metabolic 1.5%), icu_admit_source (OR 3.7% vs Other ICU 14.4%)",
            "Protective binary features: elective_surgery (−6.8pp), apache_post_operative (−5.9pp)",
            "d1_ features consistently outperform h1_ equivalents — first-day values capture full clinical picture",
        ])

    # ══════════════════════════════════════════════════════════════════════════
    # ⑦ CORRELATIONS BETWEEN FEATURES
    # ══════════════════════════════════════════════════════════════════════════
    elif page == PAGES[6]:
        sh("STEP 07","Feature Correlations","Which features are redundant with each other?")

        corr_t=get_corr_target(df, true_num_cols, binary_cols, cat_cols)
        top30_cols=corr_t[corr_t['type']!='categorical'].head(30)['feature'].tolist()

        tab_heat, tab_pairs = st.tabs(["🗺️ Heatmap","📋 Redundant pairs"])

        with tab_heat:
            n_feat=st.slider("Number of top features to show",10,min(50,len(top30_cols)),30,key='corr_n2')
            sel_cols=corr_t[corr_t['type']!='categorical'].head(n_feat)['feature'].tolist()
            corr_mat=df[sel_cols].corr()
            mask=np.triu(np.ones_like(corr_mat,dtype=bool))
            corr_masked=corr_mat.where(~mask)

            fig=px.imshow(corr_masked,color_continuous_scale='RdBu_r',
                          zmin=-1,zmax=1,aspect='auto',
                          labels={'color':'Pearson r'},**PT)
            fig.update_layout(height=max(500,n_feat*22),
                              coloraxis_colorbar=dict(title='r'))
            st.plotly_chart(fig, use_container_width=True)

        with tab_pairs:
            corr_mat2=df[top30_cols].corr().abs()
            upper=corr_mat2.where(np.triu(np.ones(corr_mat2.shape),k=1).astype(bool))
            pairs=(upper.stack().reset_index()
                       .rename(columns={'level_0':'col_A','level_1':'col_B',0:'abs_r'})
                       .sort_values('abs_r',ascending=False))
            thr=st.slider("Show pairs with |r| ≥",0.5,1.0,0.8,0.05,key='pair_thr')
            strong=pairs[pairs['abs_r']>=thr]
            st.markdown(f"**{len(strong)} pairs with |r| ≥ {thr}** — consider keeping only one of each pair.")
            st.dataframe(strong.reset_index(drop=True), use_container_width=True)

            ins("Lactate group (d1/h1 min/max) are all highly correlated — keep only d1_lactate_min. "
                "GCS components can be summed into a single gcs_total score.")

        concl("Feature Correlations",[
            "Lactate group (d1_lactate_min/max, h1_lactate_min/max) — all correlated, keep only d1_lactate_min",
            "apache_4a_hospital_death_prob ↔ apache_4a_icu_death_prob — nearly identical, keep one or exclude as leakage",
            "GCS components (motor, eyes, verbal) — correlated, consider combining: gcs_total = sum of three",
            "All invasive BP measurements (sysbp/diasbp/mbp invasive) — correlated cluster, one representative sufficient",
            "h1_inr_max ↔ d1_inr_max — correlation = 1.0, keep only d1_inr_max",
        ])

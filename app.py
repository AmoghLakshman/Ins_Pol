
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None

st.set_page_config(layout='wide', page_title='Insurance Claims Dashboard v2')
st.title('Insurance Claims — Risk & Policy Status Dashboard (v2)')

@st.cache_data
def load_data():
    df = pd.read_csv('insurance_data.csv')
    return df

def sanitize_numeric(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def preprocess(df, drop_id_like=True):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if drop_id_like:
        id_like_keys = ('policy_no','policy no','policyno','id','name','no','number')
        drop_cols = [c for c in df.columns if any(k in c.lower() for k in id_like_keys)]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    for col in ['SUM_ASSURED','PI_AGE','PI_ANNUAL_INCOME']:
        if col in df.columns:
            df = sanitize_numeric(df, col)
    return df

def build_pipeline(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
    for c in num_cols.copy():
        if X[c].nunique() <= 10 and X[c].dtype.kind in 'iu':
            cat_cols.append(c); num_cols.remove(c)
    numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    categorical_transformer = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), 
                                       ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])
    preprocessor = ColumnTransformer([('num', numeric_transformer, num_cols),
                                      ('cat', categorical_transformer, cat_cols)], remainder='drop')
    return preprocessor, num_cols, cat_cols

def train_models(X, y, n_estimators=100, random_state=42):
    preprocessor, num_cols, cat_cols = build_pipeline(X)
    models = {
        'LogisticRegression': Pipeline([('preproc', preprocessor), ('clf', LogisticRegression(max_iter=1000))]),
        'RandomForest': Pipeline([('preproc', preprocessor), ('clf', RandomForestClassifier(n_estimators=n_estimators, random_state=random_state))])
    }
    if XGBClassifier is not None:
        models['XGBoost'] = Pipeline([('preproc', preprocessor), ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=n_estimators, random_state=random_state))])
    trained = {}
    for name, pipe in models.items():
        pipe.fit(X, y)
        trained[name] = pipe
    return trained, num_cols, cat_cols

def compute_metrics(pipe, X_train, y_train, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_train_pred = pipe.predict(X_train)
    proba = pipe.predict_proba(X_test) if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
    acc = accuracy_score(y_test, y_pred)
    avg = 'binary' if len(np.unique(y_test))==2 else 'macro'
    prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
    rec = recall_score(y_test, y_pred, average=avg, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=avg, zero_division=0)
    try:
        if proba is not None and proba.shape[1] > 1:
            auc_val = roc_auc_score(y_test, proba[:,1])
        else:
            auc_val = np.nan
    except Exception:
        auc_val = np.nan
    return {'accuracy':acc,'precision':prec,'recall':rec,'f1':f1,'auc':auc_val,'y_pred':y_pred,'proba':proba,'y_train_pred':y_train_pred}

def df_to_download(df):
    buf = BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return buf

df_raw = load_data()
df = preprocess(df_raw)

st.sidebar.header('Filters')
state_col = None
for c in df.columns:
    if c.lower() in ('pi_state','state','zone','region'):
        state_col = c; break
age_col = None
for c in df.columns:
    if c.lower() in ('pi_age','age'):
        age_col = c; break

selected_states = []
if state_col:
    all_states = sorted(df[state_col].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect('PI_STATE (multi-select)', options=all_states, default=all_states)
else:
    st.sidebar.info('PI_STATE not found; multi-select disabled.')

if age_col:
    min_age = int(np.nanmin(df[age_col].fillna(0)))
    max_age = int(np.nanmax(df[age_col].fillna(100)))
    age_range = st.sidebar.slider('PI_AGE range', min_value=min_age, max_value=max_age, value=(min_age, max_age))
else:
    age_range = None

df_filtered = df.copy()
if state_col and selected_states:
    df_filtered = df_filtered[df_filtered[state_col].isin(selected_states)]
if age_col and age_range is not None:
    df_filtered = df_filtered[df_filtered[age_col].between(age_range[0], age_range[1], inclusive='both')]

st.sidebar.markdown(f'Filtered rows: {df_filtered.shape[0]}')

tab1, tab2, tab3 = st.tabs(['Dashboard (Charts)', 'Modeling', 'Upload & Predict'])

with tab1:
    st.header('Actionable Insurance-Risk Charts')
    st.write('All charts update with sidebar filters. Charts are designed to give managerial insights for underwriting and claims processing.')

    st.subheader('1. Claim Approval Ratio by State (with volume)')
    if state_col and 'POLICY_STATUS' in df_filtered.columns:
        pivot = df_filtered.groupby(state_col)['POLICY_STATUS'].value_counts().unstack(fill_value=0)
        if pivot.shape[1] == 0:
            st.info('No POLICY_STATUS values to display.')
        else:
            positive = 'Approved Death Claim' if 'Approved Death Claim' in pivot.columns else pivot.columns[0]
            approval_rate = (pivot[positive] / pivot.sum(axis=1)).sort_values(ascending=False)
            volume = pivot.sum(axis=1).loc[approval_rate.index]
            fig, ax1 = plt.subplots(figsize=(10,4))
            ax1.bar(approval_rate.index, approval_rate.values, alpha=0.7)
            ax1.set_ylabel('Approval rate')
            ax1.set_xticklabels(approval_rate.index, rotation=45, ha='right')
            ax2 = ax1.twinx()
            ax2.plot(volume.index, volume.values, color='red', marker='o', label='Volume')
            ax2.set_ylabel('Volume')
            ax1.set_ylim(0,1)
            ax1.set_title('Approval rate by state (with volume)')
            st.pyplot(fig); plt.close(fig)
    else:
        st.info('PI_STATE or POLICY_STATUS missing for chart 1.')

    st.subheader('2. SUM_ASSURED vs PI_ANNUAL_INCOME by Policy Status (scatter + trend)')
    if 'SUM_ASSURED' in df_filtered.columns and 'PI_ANNUAL_INCOME' in df_filtered.columns and 'POLICY_STATUS' in df_filtered.columns:
        df_plot = df_filtered.dropna(subset=['SUM_ASSURED','PI_ANNUAL_INCOME'])
        if not df_plot.empty:
            fig, ax = plt.subplots(figsize=(8,5))
            sns.scatterplot(data=df_plot, x='PI_ANNUAL_INCOME', y='SUM_ASSURED', hue='POLICY_STATUS', alpha=0.7, ax=ax)
            ax.set_title('Sum Assured vs Annual Income by Policy Status')
            st.pyplot(fig); plt.close(fig)
            agg = df_plot.groupby('POLICY_STATUS')[['SUM_ASSURED','PI_ANNUAL_INCOME']].mean().round(2)
            st.dataframe(agg)
        else:
            st.info('Not enough numeric data for chart 2.')
    else:
        st.info('Required columns for chart 2 missing.')

    st.subheader('3. Claim Rejection Rate by Occupation (Top 10 risky occupations)')
    occ_col = None
    for c in df_filtered.columns:
        if 'occup' in c.lower():
            occ_col = c; break
    if occ_col and 'POLICY_STATUS' in df_filtered.columns:
        counts = df_filtered[occ_col].value_counts()
        top = counts.nlargest(20).index.tolist()
        df_occ = df_filtered[df_filtered[occ_col].isin(top)].copy()
        rej = df_occ.groupby(occ_col)['POLICY_STATUS'].apply(lambda s: (s!='Approved Death Claim').mean()).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8,4))
        rej.plot(kind='bar', ax=ax)
        ax.set_ylabel('Rejection rate'); ax.set_title('Top occupations by rejection rate (top 10)')
        st.pyplot(fig); plt.close(fig)
    else:
        st.info('Occupation or POLICY_STATUS missing for chart 3.')

    st.subheader('4. Age vs Sum Assured Risk Heatmap (bins)')
    if 'PI_AGE' in df_filtered.columns and 'SUM_ASSURED' in df_filtered.columns and 'POLICY_STATUS' in df_filtered.columns:
        tmp = df_filtered.dropna(subset=['PI_AGE','SUM_ASSURED'])
        if not tmp.empty:
            tmp['age_bin'] = pd.cut(tmp['PI_AGE'], bins=6)
            tmp['sum_bin'] = pd.qcut(tmp['SUM_ASSURED'], q=6, duplicates='drop')
            heat = tmp.groupby(['age_bin','sum_bin'])['POLICY_STATUS'].apply(lambda s: (s!='Approved Death Claim').mean()).unstack(fill_value=np.nan)
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(heat, annot=False, cmap='Reds', ax=ax)
            ax.set_title('Rejection rate heatmap: Age bin x SumAssured bin')
            st.pyplot(fig); plt.close(fig)
        else:
            st.info('Not enough data for heatmap.')
    else:
        st.info('Required columns for chart 4 missing.')

    st.subheader('5. Payment Mode vs Approval Rates')
    pay_col = None
    for c in df_filtered.columns:
        if 'payment' in c.lower():
            pay_col = c; break
    if pay_col and 'POLICY_STATUS' in df_filtered.columns:
        pivot = df_filtered.groupby(pay_col)['POLICY_STATUS'].value_counts().unstack(fill_value=0)
        if pivot.shape[1] > 0:
            pos = 'Approved Death Claim' if 'Approved Death Claim' in pivot.columns else pivot.columns[0]
            approval = (pivot[pos] / pivot.sum(axis=1)).sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(8,3))
            approval.plot(kind='bar', ax=ax)
            ax.set_ylabel('Approval rate'); ax.set_title('Approval rate by payment mode')
            st.pyplot(fig); plt.close(fig)
        else:
            st.info('No POLICY_STATUS categories found for payment mode chart.')
    else:
        st.info('Payment mode or POLICY_STATUS missing for chart 5.')

with tab2:
    st.header('Modeling — Train classifiers to predict POLICY_STATUS')
    st.write('Models: Logistic Regression, Random Forest, XGBoost (if available).')
    if 'POLICY_STATUS' not in df.columns:
        st.error('POLICY_STATUS column not found in dataset. Modeling disabled.')
    else:
        test_size = st.slider('Test set proportion', 0.1, 0.4, 0.2)
        n_estimators = st.slider('n_estimators (for RF/XGB)', 10, 300, 100)
        if st.button('Train models (stratified split)'):
            data = df.dropna(subset=['POLICY_STATUS']).copy()
            X = data.drop(columns=['POLICY_STATUS']).copy()
            y = data['POLICY_STATUS'].astype(str).copy()
            le = LabelEncoder(); y_enc = le.fit_transform(y)
            st.session_state['label_encoder'] = le
            X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=test_size, random_state=42, stratify=y_enc)
            with st.spinner('Training...'):
                trained, num_cols, cat_cols = train_models(X_train, y_train, n_estimators=n_estimators)
            st.session_state['models'] = trained
            st.session_state['num_cols'] = num_cols; st.session_state['cat_cols'] = cat_cols
            st.success('Models trained and saved in session_state.')
            rows = []
            fig_roc, ax = plt.subplots(figsize=(6,5))
            colors = {'LogisticRegression':'tab:blue','RandomForest':'tab:green','XGBoost':'tab:red'}
            for name, pipe in trained.items():
                metrics = compute_metrics(pipe, X_train, y_train, X_test, y_test)
                rows.append({'Model':name,'Accuracy':round(metrics['accuracy'],4),'Precision':round(metrics['precision'],4),
                             'Recall':round(metrics['recall'],4),'F1':round(metrics['f1'],4),'AUC':round(metrics['auc'],4)})
                st.subheader(f'{name} — Confusion Matrix (test)')
                cm = confusion_matrix(y_test, metrics['y_pred'])
                fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicted'); ax_cm.set_ylabel('True'); ax_cm.set_title(f'{name} CM')
                st.pyplot(fig_cm); plt.close(fig_cm)
                if metrics['proba'] is not None and metrics['proba'].shape[1] > 1:
                    fpr, tpr, _ = roc_curve(y_test, metrics['proba'][:,1])
                    ax.plot(fpr, tpr, label=f"{name} (AUC={metrics['auc']:.3f})", color=colors.get(name))
            ax.plot([0,1],[0,1],'k--'); ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curves'); ax.legend()
            st.pyplot(fig_roc); plt.close(fig_roc)
            st.table(pd.DataFrame(rows).set_index('Model'))

with tab3:
    st.header('Upload new CSV and Predict POLICY_STATUS')
    st.write('Upload a CSV file (columns should match or be compatible). Choose a trained model, predict, and download results.')
    upload = st.file_uploader('Upload CSV for prediction', type=['csv'])
    model_choice = st.selectbox('Choose trained model', options=list(st.session_state.get('models', {}).keys()) or ['RandomForest','LogisticRegression','XGBoost'])
    if st.button('Predict and download'):
        if 'models' not in st.session_state:
            st.error('No trained models found. Train models in Modeling tab first.')
        elif upload is None:
            st.error('Please upload a CSV file to predict.')
        else:
            df_new = pd.read_csv(upload)
            df_new_proc = preprocess(df_new)
            if model_choice not in st.session_state['models']:
                st.error(f'Model {model_choice} unavailable. Ensure it was trained (XGBoost may be missing).')
            else:
                pipe = st.session_state['models'][model_choice]
                try:
                    preds = pipe.predict(df_new_proc)
                    proba = pipe.predict_proba(df_new_proc) if hasattr(pipe.named_steps['clf'], 'predict_proba') else None
                    if 'label_encoder' in st.session_state:
                        le = st.session_state['label_encoder']
                        pred_labels = le.inverse_transform(preds.astype(int))
                    else:
                        pred_labels = preds.astype(str)
                    out = df_new.copy()
                    out['PREDICTED_POLICY_STATUS'] = pred_labels
                    if proba is not None:
                        if proba.shape[1] == 2:
                            out['PRED_PROB_POS'] = proba[:,1]
                        else:
                            out['PRED_PROB_MAX'] = proba.max(axis=1)
                    buf = df_to_download(out)
                    st.download_button('Download predictions CSV', data=buf, file_name='predictions_with_status.csv', mime='text/csv')
                    st.success(f'Predicted {len(out)} rows.')
                except Exception as e:
                    st.error('Prediction failed: ' + str(e))

st.sidebar.markdown('---')
st.sidebar.info('This app auto-drops ID-like columns and sanitizes numeric fields for stability.')

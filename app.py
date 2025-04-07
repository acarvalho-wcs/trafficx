import subprocess
import sys

# Força instalação de matplotlib caso não esteja instalado
try:
    import matplotlib
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import re
import streamlit as st
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import chi2_contingency
import networkx as nx
import seaborn as sns
from io import BytesIO

# Load and clean data (handles multiple species)
def load_and_clean_data(uploaded_file):
    df = pd.read_excel(uploaded_file)
    df.columns = df.columns.str.strip().str.replace('\xa0', '', regex=True)
    df['Year'] = df['Year'].astype(str).str.extract(r'(\d{4})').astype(float)

    def expand_multi_species_rows(df):
        expanded_rows = []
        for _, row in df.iterrows():
            matches = re.findall(r'(\d+)\s*([A-Z]{2,})', str(row['N seized specimens']))
            if matches:
                for qty, species in matches:
                    new_row = row.copy()
                    new_row['N_seized'] = float(qty)
                    new_row['Species'] = species
                    expanded_rows.append(new_row)
            else:
                expanded_rows.append(row)
        return pd.DataFrame(expanded_rows)

    df = expand_multi_species_rows(df)
    df = df.reset_index(drop=True)
    return df

# Compute crime score
def org_crime_score(df, binary_features, species_col='Species', year_col='Year',
                    count_col='N_seized', country_col='Country of offenders',
                    location_col='Location of seizure or shipment', weights=None):
    default_weights = {'trend': 0.25, 'chi2': 0.15, 'anomaly': 0.20, 'network': 0.30}
    if weights:
        default_weights.update({k: weights.get(k, v) for k, v in default_weights.items()})
    score = 0
    log = {}
    annual = df.groupby(year_col)[count_col].sum().reset_index()
    if len(annual) > 1:
        model = LinearRegression().fit(annual[[year_col]], annual[count_col])
        r2 = model.score(annual[[year_col]], annual[count_col])
        if r2 > 0.4:
            score += default_weights['trend']
            log['trend'] = f'+{default_weights["trend"]:.2f} (R² = {r2:.2f})'
        elif r2 < 0.05:
            score -= default_weights['trend']
            log['trend'] = f'-{default_weights["trend"]:.2f} (R² = {r2:.2f})'
        else:
            log['trend'] = f'0 (R² = {r2:.2f})'
    if 'Species' in df.columns:
        contingency = pd.crosstab(df['Species'], df[year_col] > 2022)
        if contingency.shape == (2, 2):
            chi2, p, _, _ = chi2_contingency(contingency)
            if p < 0.05:
                score += default_weights['chi2']
                log['chi2'] = f'+{default_weights["chi2"]:.2f} (p = {p:.3f})'
            else:
                log['chi2'] = f'0 (p = {p:.3f})'
    if all(f in df.columns for f in binary_features):
        X = StandardScaler().fit_transform(df[binary_features])
        iforest = IsolationForest(random_state=42).fit_predict(X)
        lof = LocalOutlierFactor().fit_predict(X)
        dbscan = DBSCAN(eps=1.2, min_samples=2).fit_predict(X)
        outlier_votes = sum(pd.Series([iforest, lof, dbscan]).apply(lambda x: (np.array(x) == -1).sum()))
        ratio = outlier_votes / (len(df) * 3)
        if ratio > 0.15:
            score += default_weights['anomaly']
            log['anomalies'] = f'+{default_weights["anomaly"]:.2f} ({int(ratio*100)}% consensus)'
        else:
            log['anomalies'] = '0 (low outlier consensus)'
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row['Case #'])
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            if i >= j:
                continue
            shared = sum([
                row1[year_col] == row2[year_col],
                row1[species_col] == row2[species_col],
                row1[country_col] == row2[country_col]
            ])
            if shared >= 2:
                G.add_edge(row1['Case #'], row2['Case #'])
    density = nx.density(G)
    components = nx.number_connected_components(G)
    if density > 0.2 and components < len(df) / 3:
        score += default_weights['network']
        log['network'] = f'+{default_weights["network"]:.2f} (density = {density:.2f}, {components} comps)'
    else:
        log['network'] = f'0 (density = {density:.2f}, {components} comps)'
    return max(-1.0, min(1.0, score)), log

def compute_cusum(data, target_mean=None):
    if target_mean is None:
        target_mean = np.mean(data)
    pos, neg = [0], [0]
    for i in range(1, len(data)):
        diff = data[i] - target_mean
        pos.append(max(0, pos[-1] + diff))
        neg.append(min(0, neg[-1] + diff))
    return pos, neg

# Streamlit Interface
# (continua, mas limitaremos aqui para não ultrapassar o tamanho máximo)

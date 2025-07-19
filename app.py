import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Jamaica Lotto Predictor", layout="centered")

st.title("🎯 Jamaica Lotto Predictor")

# Simulated data — replace later with real draw history
np.random.seed(42)
data = [np.random.choice(range(1, 39), size=6, replace=False).tolist() for _ in range(50)]
bonus = [np.random.choice([n for n in range(1, 39) if n not in row]) for row in data]
df = pd.DataFrame([row + [b] for row, b in zip(data, bonus)],
                  columns=["d1", "d2", "d3", "d4", "d5", "d6", "bonus"])

st.subheader("📅 Last Few Draws (Simulated)")
st.dataframe(df.tail(), use_container_width=True)

def make_features(df):
    X, y = [], []
    for i in range(len(df)-1):
        prev = df.loc[i, ["d1", "d2", "d3", "d4", "d5", "d6"]].values
        feat = np.bincount(prev, minlength=39)[1:]
        X.append(feat)
        next_nums = df.loc[i+1, ["d1", "d2", "d3", "d4", "d5", "d6"]].values
        y.append(np.bincount(next_nums, minlength=39)[1:])
    return np.array(X), np.array(y)

if st.button("🔮 Predict Next 6 Numbers"):
    X, y = make_features(df)
    tscv = TimeSeriesSplit(n_splits=3)
    models = [RandomForestClassifier(n_estimators=100, random_state=42) for _ in range(38)]
    for train_i, test_i in tscv.split(X):
        for n in range(38):
            models[n].fit(X[train_i], y[train_i, n] > 0)

    latest_feat = X[-1].reshape(1, -1)
    probs = np.array([m.predict_proba(latest_feat)[0][1] for m in models])
    top = np.argsort(probs)[-6:][::-1] + 1
    st.success(f"🎉 Your Next 6 Predicted Numbers: {list(top)}")

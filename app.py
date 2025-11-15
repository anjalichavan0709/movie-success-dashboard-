
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# ----------------- BASIC CONFIG -----------------
st.set_page_config(
    page_title="Movie Success – Executive Dashboard",
    layout="wide"
)

DATA_PATH = "movie_metadata_cleaned.csv"

@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    # Safety: if success not present, create it from IMDb ≥ 7.0
    if "success" not in df.columns:
        df["success"] = (df["imdb_score"] >= 7.0).astype(int)
    return df

df = load_data(DATA_PATH)

# numeric columns & features used in model
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

TARGET_COL = "success"
MODEL_FEATURES = [
    "duration",
    "num_voted_users",
    "movie_facebook_likes",
    "num_critic_for_reviews",
    "budget",
    "title_year",
]
MODEL_FEATURES = [c for c in MODEL_FEATURES if c in numeric_cols]

success_counts = df[TARGET_COL].value_counts().sort_index()
success_rate = df[TARGET_COL].mean() * 100.0

# ----------------- HEADER & METRICS -----------------
st.title("Movie Success – Executive Dashboard")

st.markdown(
    "**Goal:** classify movies into successful (1) vs not successful (0).  \n"
    "**Success definition:** a movie is a success if *IMDb score ≥ 7.0* "
    "(then `success = 1`, else `0`)."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Rows", f"{len(df):,}", "Total movies in cleaned dataset")
col2.metric("Columns", f"{df.shape[1]}", "Original + engineered features")
col3.metric("Numeric features used in model", f"{len(MODEL_FEATURES)}")
col4.metric("Success rate (IMDb ≥ 7)", f"{success_rate:0.1f}%")

st.markdown("---")

tab_overview, tab_eda, tab_model, tab_predict = st.tabs(
    ["Overview", "EDA", "Model", "Predict"]
)

# ----------------- OVERVIEW TAB -----------------
with tab_overview:
    st.subheader("Snapshot of cleaned dataset")
    st.caption(
        "Below is a quick look at the cleaned data (first 10 rows). "
        "This is the dataset used for EDA and to train the Random Forest model."
    )
    st.dataframe(df.head(10))

    st.subheader("Numeric feature dictionary (used in model)")
    st.caption("List of key numeric features and their data types.")
    numeric_info = (
        df[MODEL_FEATURES].dtypes.rename("Type")
        .reset_index()
        .rename(columns={"index": "Feature"})
    )
    st.dataframe(numeric_info)

# ----------------- EDA TAB -----------------
with tab_eda:
    st.subheader("EDA – success vs not-success")

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.countplot(x=TARGET_COL, data=df, palette="viridis", ax=ax)
    ax.set_xticklabels(["Not successful (0)", "Successful (1)"])
    ax.set_xlabel("")
    ax.set_ylabel("Number of movies")
    st.pyplot(fig)

    st.write("**Counts:**")
    st.write(success_counts.to_frame("count"))
    st.write(f"**Success rate:** {success_rate:0.2f}% of movies have IMDb ≥ 7.0.")

    st.subheader("Distributions of key numeric features")
    feature_for_hist = st.selectbox("Select feature for histogram", MODEL_FEATURES)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    ax2.hist(df[feature_for_hist].dropna(), bins=30)
    ax2.set_xlabel(feature_for_hist)
    ax2.set_ylabel("Number of movies")
    st.pyplot(fig2)

    st.subheader("Correlation heatmap (numeric features)")
    corr = df[MODEL_FEATURES + ["imdb_score"]].corr()
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    sns.heatmap(corr, cmap="YlOrRd", annot=False, ax=ax3)
    st.pyplot(fig3)

    st.caption(
        "- Darker colours → stronger positive relationship (closer to **+1.0**).  \n"
        "- Pale / mid colours around the diagonal → weak or no relationship (around **0**).  \n"
        "- If any cells were close to **-1.0**, that would indicate a strong negative relationship."
    )

# ----------------- HELPER: TRAIN MODEL -----------------
def train_random_forest(test_size: float, n_estimators: int):
    X = df[MODEL_FEATURES]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, acc, cm, X_test, y_test, y_pred

# ----------------- MODEL TAB -----------------
with tab_model:
    st.subheader("Random Forest – train & evaluate")

    test_size = st.slider(
        "Test size (validation split)",
        min_value=0.10,
        max_value=0.40,
        value=0.20,
        step=0.05,
    )
    n_trees = st.slider(
        "Number of trees",
        min_value=50,
        max_value=200,
        value=150,
        step=10,
    )

    if st.button("Train model", type="primary"):
        model, acc, cm, X_test, y_test, y_pred = train_random_forest(
            test_size=test_size,
            n_estimators=n_trees,
        )

        # store model so Predict tab can reuse it
        st.session_state["rf_model"] = model

        st.markdown(f"**Accuracy on validation data:** {acc * 100:0.2f}%")

        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax_cm)
        ax_cm.set_xlabel("Predicted label")
        ax_cm.set_ylabel("True label")
        ax_cm.set_title("Confusion matrix (Random Forest)")
        st.pyplot(fig_cm)

# ----------------- PREDICT TAB (ONLY HERE) -----------------
with tab_predict:
    st.subheader("Predict movie success")

    duration = st.number_input("Duration (minutes)", 60, 240, 120, step=5)
    num_votes = st.number_input("Number of votes", 0, 1_000_000, 50_000, step=1_000)
    movie_fb_likes = st.number_input("Movie Facebook likes", 0, 1_000_000, 5_000, step=1_000)
    num_critic = st.number_input("Number of critic reviews", 0, 1_000, 200, step=10)
    budget = st.number_input("Budget (USD approx)", 0, 500_000_000, 50_000_000, step=1_000_000)
    title_year = st.number_input("Release year", 1920, 2030, 2010, step=1)

    if st.button("Predict success", type="primary"):
        # use trained model if available, otherwise train a default one
        if "rf_model" in st.session_state:
            model = st.session_state["rf_model"]
        else:
            model, _, _, _, _, _ = train_random_forest(test_size=0.20, n_estimators=150)
            st.session_state["rf_model"] = model

        X_new = pd.DataFrame(
            [[duration, num_votes, movie_fb_likes, num_critic, budget, title_year]],
            columns=MODEL_FEATURES,
        )

        prob_success = float(model.predict_proba(X_new)[0, 1])

        st.markdown("### Prediction result")
        if prob_success >= 0.5:
            st.success(
                f"✅ The movie is **likely to succeed** (probability: {prob_success:0.2f})."
            )
        else:
            st.warning(
                f"⚠️ The movie is **unlikely to succeed** (probability: {prob_success:0.2f})."
            )

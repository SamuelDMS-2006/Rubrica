import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, classification_report
)

st.set_page_config(page_title="Iris Species Classification", layout="wide")

# Load dataset
def load_data():
    df = pd.read_csv("Iris.csv")

    # Normalize column names
    df.columns = [c.replace(" ", "_").lower() for c in df.columns]

    return df

df = load_data()

# Delete id column if it exists
if "id" in df.columns:
    df = df.drop(columns=["id"])

# Features and target
feature_names = [col for col in df.columns if col != "species"]


# Entrenar modelo
def train_model():
    X = df[feature_names]
    y = df["species"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "report": classification_report(y_test, y_pred)
    }

    return model, metrics

model, metrics = train_model()

# Header
st.title("🌸 Iris Species Classification — Rubric")
st.write("Interactive application with training, metrics, visualization and prediction.")

# Members
st.markdown("### 👥 Members")
st.write("- Samuel Mejía\n- Aaron Roa\n- Miguel Perez\n- Aldair Escobar")

# Visual exploration of the dataset
st.subheader("📊 Dataset Exploration (EDA)")
col1, col2 = st.columns(2)

with col1:
    fig_hist = px.histogram(df, x=feature_names[0], color="species", title="Distribution of the first attribute")
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    corr = df[feature_names].corr()
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)


# Model Metrics
st.subheader("📈 Model Metrics")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
c2.metric("Precision", f"{metrics['precision']:.4f}")
c3.metric("Recall", f"{metrics['recall']:.4f}")
c4.metric("F1 Score", f"{metrics['f1']:.4f}")

with st.expander("📄 View classification report"):
    st.text(metrics["report"])


# New Sample Prediction
st.subheader("🔮 New Sample Prediction")

with st.form("predict_form"):
    inputs = {}
    for feature in feature_names:
        inputs[feature] = st.number_input(
            feature.replace("_", " ").title(),
            value=float(df[feature].mean())
        )
    submit = st.form_submit_button("Predict")

if submit:
    X_new = pd.DataFrame([inputs])
    pred = model.predict(X_new)[0]
    st.success(f"🌼 Predicted Species: **{pred}**")


# 3D Visualization
st.subheader("📌 3D Visualization with your Sample")

axis_x = st.selectbox("Axis X", feature_names)
axis_y = st.selectbox("Axis Y", feature_names)
axis_z = st.selectbox("Axis Z", feature_names)

fig3d = px.scatter_3d(df, x=axis_x, y=axis_y, z=axis_z, color="species")

if submit:
    fig3d.add_scatter3d(
        x=[X_new.iloc[0][axis_x]],
        y=[X_new.iloc[0][axis_y]],
        z=[X_new.iloc[0][axis_z]],
        mode="markers",
        marker=dict(size=8, symbol="x"),
        name="New Sample"
    )

st.plotly_chart(fig3d, use_container_width=True)
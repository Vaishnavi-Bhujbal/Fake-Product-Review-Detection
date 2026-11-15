
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("fake_review_detection_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("fake_reviews.csv")

model = load_model()
df = load_data()

st.title("🔍 Fake Product Review Detection")
st.write("Machine Learning-based detection of fake and genuine product reviews.")

# Dashboard
st.header("Project Dashboard")

total = len(df)
fake = (df["label"] == "CG").sum()
genuine = (df["label"] == "OR").sum()

c1, c2, c3 = st.columns(3)
c1.metric("Total Reviews", total)
c2.metric("Fake Reviews", int(fake))
c3.metric("Genuine Reviews", int(genuine))

st.divider()

# Review detection
st.header("Review Detection")

review = st.text_area(
    "Enter a product review",
    placeholder="Type or paste a product review here..."
)

if st.button("Detect Review"):
    if review.strip():
        prediction = model.predict([review])[0]
        probabilities = model.predict_proba([review])[0]
        confidence = max(probabilities) * 100

        if prediction == "Fake":
            st.error("🚨 FAKE REVIEW")
        else:
            st.success("✅ GENUINE REVIEW")

        st.metric("Prediction Confidence", f"{confidence:.2f}%")
    else:
        st.warning("Please enter a review.")

st.divider()

# Dataset insights
st.header("Dataset Insights")

label_counts = df["label"].value_counts()
st.bar_chart(label_counts)

st.subheader("Product Categories")
st.bar_chart(df["category"].value_counts().head(10))

st.divider()

st.subheader("Product Categories")
st.bar_chart(df["category"].value_counts().head(10))

st.divider()

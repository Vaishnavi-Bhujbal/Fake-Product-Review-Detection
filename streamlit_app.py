
import streamlit as st
import pandas as pd
import joblib

# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Fake Review Detection",
    page_icon="🔍",
    layout="wide"
)

# ----------------------------------
# LOAD MACHINE LEARNING MODEL
# ----------------------------------

@st.cache_resource
def load_model():
    return joblib.load("fake_review_detection_model.pkl")


# ----------------------------------
# LOAD DATASET
# ----------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("fake_reviews.csv")


# ----------------------------------
# LOAD FILES SAFELY
# ----------------------------------

try:
    model = load_model()
    df = load_data()

except FileNotFoundError as e:
    st.error(f"Required file not found: {e}")
    st.stop()

except Exception as e:
    st.error(f"Error loading project files: {e}")
    st.stop()


# ----------------------------------
# CHECK DATASET COLUMNS
# ----------------------------------

required_columns = ["label", "category"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    st.error(f"Missing dataset columns: {missing_columns}")
    st.stop()


# ----------------------------------
# WEBSITE TITLE
# ----------------------------------

st.title("🔍 Fake Product Review Detection")

st.write(
    "Machine Learning-based detection of fake "
    "and genuine product reviews."
)

st.divider()


# ----------------------------------
# PROJECT DASHBOARD
# ----------------------------------

st.header("📊 Project Dashboard")

total = len(df)

fake = (df["label"] == "CG").sum()
genuine = (df["label"] == "OR").sum()

c1, c2, c3 = st.columns(3)

c1.metric("Total Reviews", total)
c2.metric("Fake Reviews", int(fake))
c3.metric("Genuine Reviews", int(genuine))

st.divider()


# ----------------------------------
# REVIEW DETECTION
# ----------------------------------

st.header("📝 Review Detection")

review = st.text_area(
    "Enter a product review",
    placeholder="Type or paste a product review here..."
)

if st.button("Detect Review", type="primary"):

    if review.strip():

        try:
            prediction = model.predict([review])[0]

            st.subheader("Detection Result")

            # Handle different model labels
            fake_labels = ["CG", "Fake", "fake", 1]
            genuine_labels = ["OR", "Genuine", "genuine", 0]

            if prediction in fake_labels:
                st.error("🚨 FAKE REVIEW")

            elif prediction in genuine_labels:
                st.success("✅ GENUINE REVIEW")

            else:
                st.info(f"Model Prediction: {prediction}")

            # Confidence score
            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba([review])[0]

                confidence = max(probabilities) * 100

                st.metric(
                    "Prediction Confidence",
                    f"{confidence:.2f}%"
                )

                st.progress(float(confidence / 100))

            else:
                st.info(
                    "Confidence score is not available "
                    "for this model."
                )

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.warning("Please enter a product review.")


st.divider()


# ----------------------------------
# DATASET INSIGHTS
# ----------------------------------

st.header("📈 Dataset Insights")

label_counts = df["label"].value_counts()

st.subheader("Fake vs Genuine Reviews")

st.bar_chart(label_counts)


# ----------------------------------
# PRODUCT CATEGORIES
# ----------------------------------

st.subheader("📦 Product Categories")

category_counts = df["category"].value_counts().head(10)

st.bar_chart(category_counts)

st.divider()


# ----------------------------------
# ABOUT THE PROJECT
# ----------------------------------

st.header("ℹ️ About the Project")

st.write("""
**Objective:**

The project identifies fake and genuine product reviews
using Natural Language Processing and Machine Learning.

**Technologies Used:**

- Python
- Pandas
- TF-IDF Vectorization
- Logistic Regression
- Scikit-learn
- Streamlit
- Joblib

**Working:**

1. The user enters a product review.
2. The review is processed by the trained ML model.
3. The model predicts whether the review is fake or genuine.
4. The prediction result is displayed on the website.
""")

st.divider()

st.caption(
    "Fake Product Review Detection System | Machine Learning Mini Project"
)

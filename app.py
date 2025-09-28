import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
import textwrap


GROQ_API_KEY = "Put your API key here"

# -------------------- Streamlit Page Config --------------------
st.set_page_config(page_title="CSVInsights", layout="wide")
st.title("📊 CSVInsights – LLM Powered EDA Tool")
st.markdown("Upload a CSV and explore your data using Groq's AI-powered insights!")

# -------------------- File Upload --------------------
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

@st.cache_data
def load_csv(file):
    return pd.read_csv(file)

def make_summary(df: pd.DataFrame) -> str:
    buf = []
    buf.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    buf.append("Column data types:")
    for col, dtype in df.dtypes.items():
        buf.append(f"- {col}: {dtype}")
    if df.isna().sum().sum() > 0:
        buf.append("Missing values:")
        for col, cnt in df.isna().sum().items():
            if cnt > 0:
                buf.append(f"- {col}: {cnt}")
    buf.append("\nSample rows:")
    buf.append(df.head(5).to_csv(index=False))
    return "\n".join(buf)

def query_llm(prompt: str):
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Use Groq-supported model
        messages=[
            {"role": "system", "content": "You are a data assistant that analyzes CSVs."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=400,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# -------------------- Auto Clean Function --------------------
def auto_clean_dataset(df):
    # Drop duplicates
    df = df.drop_duplicates()

    # Drop columns with >40% missing values
    threshold = len(df) * 0.4
    df = df.dropna(thresh=threshold, axis=1)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            df[col] = df[col].fillna(df[col].mean())  # Numeric → mean
        else:
            df[col] = df[col].fillna(df[col].mode()[0])  # Categorical → mode
    return df

# -------------------- Main App --------------------
if uploaded_file:
    df = load_csv(uploaded_file)

    st.subheader("🔎 Data Preview")
    st.dataframe(df.head())

    # Dataset info
    st.subheader("📌 Dataset Info")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write("**Data Types**")
        st.table(df.dtypes.astype(str))
    with c2:
        st.write("**Missing Values**")
        st.table(df.isna().sum())
    with c3:
        st.write("**Summary (Numeric)**")
        st.table(df.describe().T.round(2))

    # -------------------- Cleaning Section --------------------
    st.subheader("🧹 Clean Dataset")
    cleaned_df = auto_clean_dataset(df)

    st.success("✅ Dataset cleaned automatically (duplicates removed, missing values handled).")
    st.write("Preview of Cleaned Dataset:")
    st.dataframe(cleaned_df.head())

    # Download button
    csv = cleaned_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="💾 Download Cleaned Dataset",
        data=csv,
        file_name="cleaned_dataset.csv",
        mime="text/csv"
    )

    # -------------------- Visualizations --------------------
    st.subheader("📊 Visualizations")
    col = st.selectbox("Select a column for visualization", df.columns)

    if pd.api.types.is_numeric_dtype(df[col]):
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=30, color="skyblue", edgecolor="black")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        df[col].value_counts().nlargest(20).plot(kind="bar", ax=ax, color="lightcoral")
        ax.set_title(f"Value Counts of {col}")
        st.pyplot(fig)

    if st.checkbox("Show Correlation Heatmap"):
        num_cols = df.select_dtypes(include=[np.number]).columns
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # -------------------- LLM Q&A --------------------
    st.subheader("🤖 Ask your CSV (Groq-powered)")
    user_q = st.text_area("Type a question about the dataset")
    if st.button("Ask"):
        summary = make_summary(df)
        prompt = textwrap.dedent(f"""
        You are given a dataset with the following summary:
        {summary}

        Question: {user_q}

        Provide a concise answer based ONLY on the dataset.
        """)
        with st.spinner("Thinking..."):
            ans = query_llm(prompt)
        st.success(ans)
else:
    st.info("⬆️ Upload a CSV to begin.")

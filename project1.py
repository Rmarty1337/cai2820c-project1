import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# GitHub raw file URL
GITHUB_FILE_URL = "https://github.com/Rmarty1337/cai2820c-project1/raw/refs/heads/main/AllITBooks_DataSet.xlsx"

st.title("📚 Book Categorization App")

@st.cache_data
def load_data(url):
    """Download and load dataset from GitHub."""
    response = requests.get(url)
    if response.status_code == 200:
        df = pd.read_excel(BytesIO(response.content), engine="openpyxl")
        return df
    else:
        st.error("Failed to load dataset. Check your GitHub URL.")
        return None

df = load_data(GITHUB_FILE_URL)

if df is not None:
    st.write("### Preview of Dataset:")
    st.write(df.head())

    if "Description" in df.columns:
        # Convert text descriptions into TF-IDF features
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        X = vectorizer.fit_transform(df["Description"].astype(str))

        # Apply K-Means clustering
        num_clusters = 5  # Adjust as needed
        kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        df["Category"] = kmeans.fit_predict(X)

        st.write("### Categorized Books:")
        if "Title" in df.columns:
            st.write(df[["Title", "Category"]])
        else:
            st.write(df[["Category"]])

        # Allow downloading of categorized file
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Categorized Data", data=csv, file_name="categorized_books.csv", mime="text/csv")

    else:
        st.error("The dataset must contain a 'Description' column.")


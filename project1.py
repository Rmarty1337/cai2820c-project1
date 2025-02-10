import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import subprocess

# Function to update a package
def update_package(package_name):
    try:
        subprocess.check_call(["pip", "install", "--upgrade", package_name])
        st.write(f"{package_name} is updated successfully.")
    except Exception as e:
        st.write(f"Error updating {package_name}: {e}")

# Update pip first
update_package("pip")

# Update and verify installations
packages = ["streamlit", "pandas", "requests", "openpyxl", "scikit-learn"]
for package in packages:
    update_package(package)

# Check pip version
try:
    pip_version = subprocess.check_output(["pip", "--version"]).decode("utf-8").strip()
    st.write(f"pip version: {pip_version}")
except Exception as e:
    st.write(f"Error checking pip version: {e}")

# Verify installations
try:
    import streamlit as st
    st.write("Streamlit is installed correctly.")
except ImportError:
    st.write("Streamlit is not installed.")

try:
    import pandas as pd
    st.write("Pandas is installed correctly.")
except ImportError:
    st.write("Pandas is not installed.")

try:
    import requests
    st.write("Requests is installed correctly.")
except ImportError:
    st.write("Requests is not installed.")

try:
    import openpyxl
    st.write("Openpyxl is installed correctly.")
except ImportError:
    st.write("Openpyxl is not installed.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    st.write("Scikit-learn is installed correctly.")
except ImportError:
    st.write("Scikit-learn is not installed.")

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

        # Display top 5 books in each category
        st.write("### Top 5 Books in Each Category:")
        for category in range(num_clusters):
            st.write(f"#### Category {category}")
            top_books = df[df["Category"] == category].head(5)
            if "Title" in df.columns:
                st.write(top_books[["Title", "Description"]])
            else:
                st.write(top_books[["Description"]])

    else:
        st.error("The dataset must contain a 'Description' column.")


import streamlit as st
import pandas as pd
import requests
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

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
        # Check if necessary columns exist
        if all(col in df.columns for col in ["Title", "Sub_title", "Description"]):
            # Combine relevant text columns
            df["ConsolidatedText"] = df["Title"] + " " + df["Sub_title"] + " " + df["Description"]

            # Define stop words
            stop_words = set(stopwords.words('english'))

            # Function to preprocess text
            def preprocess_text(text):
                text = str(text).lower()
                tokens = word_tokenize(text)
                filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
                return " ".join(filtered_tokens)

            # Apply preprocessing
            df["CleanedDescription"] = df["ConsolidatedText"].apply(preprocess_text)

            # Convert text descriptions into TF-IDF features
            vectorizer = TfidfVectorizer(max_features=500)
            tfidf_matrix = vectorizer.fit_transform(df["CleanedDescription"])

            # Apply NMF for topic modeling
            nmf_model = NMF(n_components=10, random_state=42)
            nmf_model.fit(tfidf_matrix)

            # Get topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(nmf_model.components_):
                top_words = [feature_names[i] for i in topic.argsort()[:-11:-1]]
                topics.append(", ".join(top_words))

            # Assign topics to documents
            topic_assignments = nmf_model.transform(tfidf_matrix).argmax(axis=1)
            df["AssignedTopic"] = topic_assignments
            df["Topic_Keywords"] = [topics[i] for i in topic_assignments]

            # Define topic categories
            categories = {
                0: "Content Management Systems (CMS)",
                1: "Web Development and Frameworks",
                2: "Data Analysis and Big Data",
                3: "Game Development",
                4: "Network and Security Administration",
                5: "Programming Languages and Functional Programming",
                6: "Mobile App Development",
                7: "Java and Enterprise Applications",
                8: "Python and Machine Learning",
                9: "Databases and SQL Administration"
            }
            df["Topic"] = df["AssignedTopic"].map(categories)

            st.write("### Categorized Books:")
            st.write(df[["Title", "Category", "Topic", "Topic_Keywords"]])

            # Allow downloading of categorized file
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Categorized Data", data=csv, file_name="categorized_books.csv", mime="text/csv")

            # Display top 5 books in each category
            st.write("### Top 5 Books in Each Category:")
            for category in range(10):
                st.write(f"#### Category {category}: {categories[category]}")
                top_books = df[df["AssignedTopic"] == category].head(5)
                st.write(top_books[["Title", "Description"]])
        else:
            st.error("The dataset must contain 'Title', 'Sub_title', and 'Description' columns.")
    else:
        st.error("The dataset must contain a 'Description' column.")


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

@st.cache_data
def load_data(url):
    """Download and load dataset from GitHub."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_excel(BytesIO(response.content), engine="openpyxl")
        return df
    except Exception as e:
        st.error(f"Failed to load dataset: {e}")
        return None

df = load_data(GITHUB_FILE_URL)

# Define stop words
stop_words = set(stopwords.words('english'))

# Function to preprocess text
def preprocess_text(text):
    try:
        text = str(text).lower()
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
        return " ".join(filtered_tokens)
    except Exception as e:
        st.warning(f"Text preprocessing failed: {e}")
        return "N/A"

# Main Streamlit app
st.title("📚 Book Categorization App")

if df is not None:
    st.write("### Preview of Dataset:")
    st.write(df.head())

    if "Description" in df.columns:
        try:
            # Combine relevant text columns
            df["ConsolidatedText"] = df["Book_name"] + " " + df["Sub_title"] + " " + df["Description"]

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
            st.write(df[["Book_name", "Category", "Topic", "Topic_Keywords"]])

            # Allow downloading of categorized file
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Categorized Data", data=csv, file_name="categorized_books.csv", mime="text/csv")

            # Display top 5 books in each category
            st.write("### Top 5 Books in Each Category:")
            for category in range(10):
                st.write(f"#### Category {category}: {categories[category]}")
                top_books = df[df["AssignedTopic"] == category].head(5)
                st.write(top_books[["Book_name", "Description"]])
        except Exception as e:
            st.error(f"Error processing dataset: {e}")
    else:
        st.error("The dataset must contain a 'Description' column.")


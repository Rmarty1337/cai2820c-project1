import subprocess

# Check pip version
try:
    pip_version = subprocess.check_output(["pip", "--version"]).decode("utf-8").strip()
    print(f"pip version: {pip_version}")
except Exception as e:
    print(f"Error checking pip version: {e}")

# Verify installations
try:
    import streamlit as st
    print("Streamlit is installed correctly.")
except ImportError:
    print("Streamlit is not installed.")

try:
    import pandas as pd
    print("Pandas is installed correctly.")
except ImportError:
    print("Pandas is not installed.")

try:
    import requests
    print("Requests is installed correctly.")
except ImportError:
    print("Requests is not installed.")

try:
    import openpyxl
    print("Openpyxl is installed correctly.")
except ImportError:
    print("Openpyxl is not installed.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF
    print("Scikit-learn is installed correctly.")
except ImportError:
    print("Scikit-learn is not installed.")

try:
    import nltk
    print("NLTK is installed correctly.")
except ImportError:
    print("NLTK is not installed.")

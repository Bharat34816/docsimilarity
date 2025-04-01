import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

def load_documents_from_urls(urls):
    """Load text documents from URLs."""
    documents = []
    filenames = []

    for url in urls:
        try:
            response = requests.get(url)
            response.raise_for_status()  
            documents.append(response.text)
            
            
            filename = url.split('/')[-1]
            filenames.append(filename)
            
        except Exception as e:
            print(f"Failed to load {url}: {e}")

    return documents, filenames

def preprocess_text(documents):
    """Preprocess text documents with TF-IDF encoding."""
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(documents)
    
    return vectors, vectorizer

import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Streamlit UI
st.title("Document Similarity Score")

# Upload documents
uploaded_files = st.file_uploader("Upload multiple text files", accept_multiple_files=True, type=["txt"])

# Function to read and decode uploaded files
def read_files(files):
    documents = []
    filenames = []
    for file in files:
        content = file.read().decode("utf-8")
        documents.append(content)
        filenames.append(file.name)
    return documents, filenames

# Process the uploaded files
if uploaded_files:
    documents, filenames = read_files(uploaded_files)

    if len(documents) >= 2:
        # Preprocessing and vectorizing
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)

        # Compute cosine similarity
        similarity_matrix = cosine_similarity(vectors)

        # Display similarity matrix
        st.subheader("Similarity Matrix")
        similarity_df = pd.DataFrame(similarity_matrix, columns=filenames, index=filenames)
        st.dataframe(similarity_df)

        # Display similarity decision
        st.subheader("Similarity Results")

        threshold = st.slider("Set Similarity Threshold:", 0.0, 1.0, 0.5, 0.05)

        results = []
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                score = similarity_matrix[i, j]
                result = {
                    "Document 1": filenames[i],
                    "Document 2": filenames[j],
                    "Similarity Score": score,
                    "Similar": "✅ YES" if score >= threshold else "❌ NO"
                }
                results.append(result)

        results_df = pd.DataFrame(results)
        st.write(results_df)

    else:
        st.warning("Please upload at least 2 documents.")

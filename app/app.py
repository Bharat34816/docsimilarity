import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

def read_files(files):
    documents = []
    filenames = []
    for file in files:
        if file.type == "application/pdf":
            doc = fitz.open(stream=file.read())
            text = "\n".join([page.get_text("text") for page in doc])
        else:
            text = file.read().decode("utf-8")
        documents.append(text)
        filenames.append(file.name)
    return documents, filenames

st.title("Document Similarity Score")

uploaded_files = st.file_uploader("Upload text or PDF files", accept_multiple_files=True, type=["txt", "pdf"])
text_input = st.text_area("Or enter text manually (separate documents with ---)")

if uploaded_files or text_input:
    documents, filenames = read_files(uploaded_files) if uploaded_files else ([], [])
    if text_input:
        split_texts = text_input.split("---")
        documents.extend(split_texts)
        filenames.extend([f"Text {i+1}" for i in range(len(split_texts))])
    
    if len(documents) >= 2:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(documents)
        similarity_matrix = cosine_similarity(vectors)
        
        st.subheader("Similarity Matrix")
        similarity_df = pd.DataFrame(similarity_matrix, columns=filenames, index=filenames)
        st.dataframe(similarity_df)
        
        fig, ax = plt.subplots()
        sns.heatmap(similarity_df, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
        st.subheader("Similarity Results")
        threshold = st.slider("Set Similarity Threshold:", 0.0, 1.0, 0.5, 0.05)
        results = []
        for i in range(len(filenames)):
            for j in range(i + 1, len(filenames)):
                score = similarity_matrix[i, j]
                result = {"Document 1": filenames[i], "Document 2": filenames[j], "Similarity Score": score, "Similar": "✅ YES" if score >= threshold else "❌ NO"}
                results.append(result)
        st.write(pd.DataFrame(results))
    else:
        st.warning("Please provide at least two documents or text inputs.")

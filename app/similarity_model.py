from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compute_similarity(vectors, filenames):
    """Calculate similarity scores between documents."""
    similarity_matrix = cosine_similarity(vectors)

   
    df = pd.DataFrame(similarity_matrix, columns=filenames, index=filenames)
    return df

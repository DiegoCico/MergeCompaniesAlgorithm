import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def group_similar_companies(data, name_threshold=0.8, address_threshold=0.8):
    """Group similar companies based on shipper_name and first3_addresses."""
    vectorizer = TfidfVectorizer().fit(data['shipper_name'] + ' ' + data['first3_addresses'])
    tfidf_matrix = vectorizer.transform(data['shipper_name'] + ' ' + data['first3_addresses'])
    similarity_matrix = cosine_similarity(tfidf_matrix)

    groups = [-1] * len(data)  # -1 means ungrouped
    group_id = 1

    for i in range(len(data)):
        if groups[i] == -1:  # If ungrouped
            groups[i] = group_id
            for j in range(i + 1, len(data)):
                if groups[j] == -1 and similarity_matrix[i, j] >= max(name_threshold, address_threshold):
                    groups[j] = group_id
            group_id += 1

    data['group'] = groups
    return data

# Load the CSV file
data = pd.read_csv('./shipper_name_chunk_0.csv')

# Ensure group column is initialized
data['group'] = None

# Apply the grouping function
data = group_similar_companies(data)

# Filter out rows where group is not assigned
data = data[data['group'] != -1]

# Save the grouped data to a new CSV file
output_file = './grouped_shippers2.csv'
data.to_csv(output_file, index=False)

output_file

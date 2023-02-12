import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def find_clustered_keywords(corpus, num_clusters):
    """
    This function takes a list of documents (corpus) and the number of clusters
    as input and returns a list of clustered keywords.
    """
    # Initialize the TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')

    # Transform the corpus into a term frequency-inverse document frequency matrix
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Initialize the KMeans clustering model
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)

    # Fit the model to the tfidf matrix
    kmeans.fit(tfidf_matrix)

    # Get the cluster centers
    cluster_centers = kmeans.cluster_centers_

    # Get the sorted indices of the top keywords in each cluster
    sorted_indices = cluster_centers.argsort()[:, ::-1]

    # Get the feature names
    feature_names = vectorizer.get_feature_names_out()

    # Initialize a dictionary to store the clustered keywords
    clustered_keywords = {}

    # Iterate over each cluster
    for i in range(num_clusters):
        # Initialize a list to store the keywords in the current cluster
        keywords = []

        # Iterate over the top keywords in the current cluster
        for index in sorted_indices[i, :10]:
            # Add the keyword to the list
            keywords.append(feature_names[index])

        # Add the list of keywords to the dictionary
        clustered_keywords[i] = keywords

    return clustered_keywords

# Define a list of documents
corpus = [
    'This is the first document',
    'This is the second document',
    'This is the third document',
    'This is the fourth document',
    'This is the fifth document',
    'kashi is know as python coder'
]

# Find the clustered keywords
clustered_keywords = find_clustered_keywords(corpus, 2)

# Print the clustered keywords
print(clustered_keywords)

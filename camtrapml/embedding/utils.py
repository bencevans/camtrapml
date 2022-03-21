"""
Embeddings Utilities
"""

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def plot_embeddings(embeddings, labels=None, title=None):
    """
    Plots the embeddings of the images in the dataset.
    """

    # Normalize the embeddings
    embeddings = embeddings / embeddings.std(axis=1).reshape(-1, 1)

    # Reduce the dimensionality of the embeddings
    pca = PCA(n_components=2)
    embeddings = pca.fit_transform(embeddings)

    # Visualize the embeddings
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
    embeddings = tsne.fit_transform(embeddings)

    # Plot the embeddings
    plt.figure(figsize=(8, 8))
    for i in range(len(embeddings)):
        plt.scatter(
            embeddings[i, 0],
            embeddings[i, 1],
            c=labels[i] if labels else None,
            alpha=0.5,
        )

    if title is not None:
        plt.title(title)

    plt.show()

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

from makeFeatureVector import makeFeatureVec

# === MAIN ===

if __name__ == "__main__":
    image_folder = "Lab3.1"
    #image_folder = 'C:/Users/almal/Desktop/termin8/TNM098/lab 3/TNM098_Lab3/Lab3.1'

    image_paths = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))])[:12]

    feature_vectors = []

    # extract features using makeFeatureVec
    for path in image_paths:
        img = cv2.imread(path)
        vec = makeFeatureVec(img)
        feature_vectors.append(vec)

    feature_matrix = np.array(feature_vectors)

    # compute the cosine similarity distance between all vectors
    similarity_matrix = cosine_similarity(feature_matrix)

    # TODO: save results in a 12x12 matrix of cosine similarity distances
    print("12x12 Cosine Similarity Matrix:")
    print(similarity_matrix)

    # TODO: use the matrix to rank the 11 images in similarity to one chosen image
    chosen_index = 0  # välj t.ex. första bilden
    scores = similarity_matrix[chosen_index]
    ranked_indices = np.argsort(scores)[::-1]

    print(f"\nLikhetsranking till bilden: {os.path.basename(image_paths[chosen_index])}")
    for i in ranked_indices[1:]:  # hoppa över bilden själv
        print(f"{os.path.basename(image_paths[i])} — Likhet: {scores[i]:.4f}")

    # visualize the distance results using a heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(similarity_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title("Likhetsmatris (cosine similarity)")
    plt.xticks(ticks=range(12), labels=[os.path.basename(p) for p in image_paths], rotation=45)
    plt.yticks(ticks=range(12), labels=[os.path.basename(p) for p in image_paths])
    plt.tight_layout()
    plt.show()

# Smart Laptop Upgrade Advisor

A data mining and intelligent recommendation project designed to analyze laptop user problem descriptions and support personalized upgrade recommendations. The project applies text mining, clustering, similarity analysis, and graph mining techniques to identify recurring laptop performance issues and uncover relationships between similar user cases.

## Project Overview

The Smart Laptop Upgrade Advisor analyzes user stories describing laptop problems, such as slow performance, storage limitations, multitasking issues, and gaming-related problems. These unstructured text descriptions are transformed into machine-readable features using TF-IDF, enabling the system to identify similar cases, discover recurring patterns, group related problems, and support suitable upgrade recommendations such as RAM, SSD, or GPU upgrades.

## Key Techniques

- Text Preprocessing
- TF-IDF Vectorization
- K-Means Clustering
- PCA Dimensionality Reduction
- Cosine Similarity
- Graph Mining
- Network Analysis
- PageRank
- Degree Centrality
- Betweenness Centrality
- Closeness Centrality

## Key Functionality

- Converts unstructured laptop problem descriptions into numerical TF-IDF features.
- Groups similar laptop issues using K-Means clustering.
- Uses PCA to reduce feature dimensions and visualize clusters.
- Measures similarity between laptop user cases using cosine similarity.
- Builds similarity graphs to analyze relationships between related cases.
- Identifies representative, highly connected, and bridge cases using graph centrality measures.
- Supports personalized upgrade recommendations based on recurring issue patterns and similar historical cases.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- K-Means
- TF-IDF
- PCA
- Cosine Similarity
- NetworkX
- Matplotlib

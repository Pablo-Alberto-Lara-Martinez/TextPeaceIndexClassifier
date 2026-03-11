# Clustering and Validation Methodology

This document details the unsupervised learning approach (K-Means) and the cross-validation strategy (Leave-One-Out) implemented to analyze text embeddings and classify the peace levels of countries.

## 1. K-Means Clustering for Unsupervised Classification

While the Neural Network provides a supervised approach to classification, we also employ K-Means clustering as an unsupervised baseline. This algorithm attempts to find inherent structures within the high-dimensional space of the textual embeddings without prior knowledge of the true peace labels during the fitting process.

### Mathematical Foundation

K-Means partitions the embedding space into $K$ distinct, non-overlapping clusters. For this binary classification problem (Peaceful vs. Non-Peaceful), we set $K=2$. 

The algorithm aims to minimize the within-cluster sum of squares (variance), often referred to as inertia. Given a set of text embeddings $(x_1, x_2, \dots, x_n)$, the objective is to find the set of clusters $C = \{C_1, C_2\}$ and their respective centroids $\mu = \{\mu_1, \mu_2\}$ that minimize the following objective function:

$$J = \sum_{j=1}^{K} \sum_{x_i \in C_j} \| x_i - \mu_j \|^2$$

Where:
* $K = 2$ represents the two target clusters.
* $x_i$ is a data point (embedding vector) belonging to cluster $C_j$.
* $\mu_j$ is the centroid (mean vector) of cluster $C_j$.
* $\| x_i - \mu_j \|^2$ is the squared Euclidean distance between the data point and the centroid.

### Cluster Mapping Protocol

Because K-Means is unsupervised, the resulting clusters (e.g., Cluster 0 and Cluster 1) do not inherently map to "Peaceful" (1) or "Non-Peaceful" (0). To evaluate the clustering performance, we implement a **Majority Voting Mapping**:
1. After the clusters are formed using the training set, we examine the true labels of the countries within each cluster.
2. A cluster is assigned the class (0 or 1) that represents the majority of its members.
3. This mapped logic is then used to predict the label of the test data based on which centroid it is closest to.

## 2. Leave-One-Out Cross-Validation (LOO-CV)

When dealing with macro-level datasets where the primary instances are distinct countries, the sample size $N$ is naturally constrained. Traditional train-test splits (like 80/20) are highly volatile and unreliable on small datasets, as the random exclusion of even one or two specific countries from the training set can drastically alter the model's learned boundaries.

To ensure robustness, we utilize Leave-One-Out Cross-Validation (LOO-CV).

### The LOO-CV Process

In LOO-CV, the number of folds equals the number of instances in the dataset ($N$ countries). For each fold $i$:
1. **Test Set:** A single country ($x_i$) is isolated to serve as the test set.
2. **Training Set:** The remaining $N-1$ countries are used to train the model (both the CNN and the K-Means algorithms).
3. **Evaluation:** The trained model predicts the peace class of the isolated country $x_i$.
4. **Iteration:** This process is repeated $N$ times, ensuring every single country serves as the test set exactly once.

### Justification

This approach is computationally expensive for massive datasets, but it is the gold standard for small sample sizes. It maximizes the amount of data available for training in each iteration ($N-1$ samples), allowing the algorithms to learn the most generalized patterns of language associated with peace levels, while providing a completely unbiased evaluation of the model's performance on unseen data.
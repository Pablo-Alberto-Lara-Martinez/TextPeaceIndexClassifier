# src/models.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from .embeddings import prepare_country_level
from .data_utils import majority_label, mean_label_value

def red_neuronal_categorica(input_shape):
    """
    Defines and compiles the 1D Convolutional Neural Network architecture.
    """
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def map_clusters_to_class(cluster_labels, y_train):
    """
    Maps cluster_id -> {0,1} using majority voting based on true labels.
    """
    mapping = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        labels = y_train[mask]
        mapping[int(c)] = int(majority_label(labels))
    return mapping

def map_clusters_to_mean_value(cluster_labels, y_train):
    """
    Maps cluster_id -> average label (0 or 1) based on true labels.
    """
    mapping = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        labels = y_train[mask]
        mapping[int(c)] = mean_label_value(labels)
    return mapping

def leave_one_out_kmeans_country(df, n_init=30, random_state=42):
    """
    Executes Leave-One-Out cross-validation at the country level using KMeans.
    - Trains KMeans(k=2) with all countries except one.
    - Maps clusters to peace status by majority.
    - Predicts the class of the left-out country.
    """
    countries, X_country, y_country, _ = prepare_country_level(df)
    n_c = len(countries)
    results = []
    
    for i in range(n_c):
        # Split
        train_idx = [j for j in range(n_c) if j != i]
        test_idx = i

        X_train, y_train = X_country[train_idx], y_country[train_idx]
        X_test = X_country[test_idx].reshape(1, -1)
        true_label = int(y_country[test_idx])

        # Train
        kmeans = KMeans(n_clusters=2, n_init=n_init, random_state=random_state)
        cluster_labels = kmeans.fit_predict(X_train)

        # Mapping cluster -> class
        mapping = map_clusters_to_class(cluster_labels, y_train)
        mapping_mean = map_clusters_to_mean_value(cluster_labels, y_train)

        # Predict
        cluster_test = int(kmeans.predict(X_test)[0])
        predicted_label = int(mapping[cluster_test])

        # Stats
        train_pos_frac = float((y_train == 1).mean())
        train_neg_frac = float((y_train == 0).mean())

        results.append({
            "fold": i, "country": countries[test_idx],
            "true_peace": true_label, "predicted_peace": predicted_label,
            "train_pos_frac": round(train_pos_frac, 4),
            "train_neg_frac": round(train_neg_frac, 4),
            "cluster_test": cluster_test,
            "mean_peace_cluster_test": mapping_mean[cluster_test]
        })

    return pd.DataFrame(results)

def leave_one_out_red_neuronal_categorica(df, model):
    """
    Executes Leave-One-Out cross-validation at the country level using the CNN.
    - Trains the model with all countries except one.
    - Predicts the class of the left-out country.
    """
    countries, X_country, y_country, _ = prepare_country_level(df)
    n_c = len(countries)
    results = []
    
    for i in range(n_c):
        # Split
        train_idx = [j for j in range(n_c) if j != i]
        test_idx = i
        
        X_train, y_train = X_country[train_idx], y_country[train_idx]
        X_test = X_country[test_idx].reshape(1, -1)
        true_label = int(y_country[test_idx])

        # Train
        model.fit(X_train, y_train, verbose=0)
        
        # Predict
        predicted_values = model.predict(X_test, verbose=0)
        predicted_label = int(round(predicted_values[0][0]))

        # Stats
        train_pos_frac = float((y_train == 1).mean())
        train_neg_frac = float((y_train == 0).mean())

        results.append({
            "country": countries[test_idx],
            "true_peace": true_label, "predicted_peace": predicted_label,
            "train_pos_frac": round(train_pos_frac, 4),
            "train_neg_frac": round(train_neg_frac, 4),
            "mean_peace_cluster_test": predicted_values[0][0]
        })

    return pd.DataFrame(results)
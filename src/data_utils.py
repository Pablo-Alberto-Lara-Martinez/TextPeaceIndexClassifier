# src/data_utils.py
import pandas as pd
from collections import Counter
import numpy as np

def df_sample_from_random_rows(df, n, random_state):
    """
    Selects n random elements for each unique country in the dataframe.
    If a country has fewer than n elements, all its elements are selected.
    """
    unique_countries = df['country'].unique()
    selected_rows = pd.DataFrame(columns=df.columns)
    
    for country in unique_countries:
        country_rows = df[df['country'] == country]
        if len(country_rows) >= n:
            selected_rows = pd.concat([selected_rows, country_rows.sample(n=n, random_state=random_state)])
        else:
            selected_rows = pd.concat([selected_rows, country_rows])
            
    return selected_rows.reset_index(drop=True)

def majority_label(labels):
    """
    Returns the majority label (0 or 1) from a list of labels.
    Handles exact ties by defaulting to 1 in a stable manner.
    """
    c = Counter(labels)
    if len(c) == 0:
        return None
    # If there is an exact tie, decide in a stable way
    if len(c) > 1 and c.most_common(1)[0][1] == c.most_common(2)[1][1]:
        return 1
    return c.most_common(1)[0][0]

def mean_label_value(labels):
    """
    Returns the average label value (0 or 1) from a list of labels.
    """
    return np.mean(labels)
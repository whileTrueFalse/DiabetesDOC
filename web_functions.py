"""This module contains necessary functions needed for loading data, training models, and making predictions."""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_data():
    """Load the Diabetes dataset into a DataFrame, perform feature and target split.
    
    Returns:
        tuple: DataFrame, Features DataFrame, Target Series.
    """
    df = pd.read_csv('diabetes.csv')
    X = df[["Pregnancies", "FastingGlc", "AfterGlc", "BloodPressure", "SkinThickness", "Insulin", "BMI", "GeneticCorr", "Age"]]
    y = df['Outcome']

    # Optionally, return a copy to avoid unexpected side effects
    return df.copy(), X.copy(), y.copy()

@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def train_model(X, y):
    """Train a decision tree classifier with specified parameters and return the model and its accuracy score.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target values.

    Returns:
        tuple: Trained model, accuracy score.
    """
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1, 
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(X, y)
    score = model.score(X, y)

    return model, score

def predict(X, y, features):
    """Predict the outcome for a new sample using a trained model.
    
    Args:
        X (DataFrame): Feature set.
        y (Series): Target values.
        features (list): Input features for which prediction is needed.

    Returns:
        tuple: Prediction result, model score.
    """
    model, score = train_model(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score

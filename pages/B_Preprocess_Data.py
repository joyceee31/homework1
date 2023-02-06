import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

# Checkpoint 1
def restore_dataset():
    """
    Input: 
    Output: 
    """
    df=None
    return df

# Checkpoint 3
def remove_features(X,removed_features):
    """
    Input: 
    Output: 
    """
    return X

# Checkpoint 4
def impute_dataset(X, impute_method):
    """
    Input: 
    Output: 
    """
    return X

# Checkpoint 5
def compute_descriptive_stats(X, stats_feature_select, stats_select):
    """
    Input: 
    Output: 
    """
    output_str=''
    out_dict = {
        'mean': None,
        'median': None,
        'max': None,
        'min': None
    }
    return output_str, out_dict

# Checkpoint 6
def split_dataset(X, number):
    """
    Input: 
    Output: 
    """
    train=[]
    test=[]
    return train, test

# Restore Dataset
df = restore_dataset()

if df is not None:

    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values

    ############################################# MAIN BODY #############################################

    #numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect

    # Remove the features using the remove_features function

    # Display updated dataframe

    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}

    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset
    
    # Display updated dataframe
    
    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')

    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect

    # Provide option to select multiple descriptive statistics to show using Streamit multiselect
 
    # Compute Descriptive Statistics including mean, median, min, max
    # ... = compute_descriptive_stats(...)
        
    # Display updated dataframe
    st.markdown('### Result of the imputed dataframe')

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')

    # Compute the percentage of test and training data
    
    # Print dataset split result

    # Save state of train and test split

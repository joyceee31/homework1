import streamlit as st                  
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from pandas.plotting import scatter_matrix
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

feature_lookup = {
    'longitude':'**longitude** - longitudinal coordinate',
    'latitude':'**latitude** - latitudinal coordinate',
    'housing_median_age':'**housing_median_age** - median age of district',
    'total_rooms':'**total_rooms** - total number of rooms per district',
    'total_bedrooms':'**total_bedrooms** - total number of bedrooms per district',
    'population':'**population** - total population of district',
    'households':'**households** - total number of households per district',
    'median_income':'**median_income** - median income',
    'ocean_proximity':'**ocean_proximity** - distance from the ocean',
    'median_house_value':'**median_house_value**'
}

#############################################

st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Explore Dataset')

#############################################

st.markdown('### Import Dataset')

# Checkpoint 1
def load_dataset(data):
    """
    Input: data is the filename or path to file (string)
    Output: pandas dataframe df
    - Checkpoint 1 - Read .csv file containing a dataset
    """
    pass
    return 0

# Checkpoint 2
def compute_correlation(X,features):
    """
    Input: X is pandas dataframe, features is a list of feature name (string) ['age','height']
    Output: correlation coefficients between one or more features
    """
    pass
    return 0

# Helper Function
def user_input_features(df):
    """
    Input: pnadas dataframe containing dataset
    Output: dictionary of sidebar filters on features
    """
    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    side_bar_data = {}
    for feature in numeric_columns:
        try:
            f = st.sidebar.slider(str(feature), float(df[str(feature)].min()), float(df[str(feature)].max()), float(df[str(feature)].mean()))
        except Exception as e:
            print(e)
        side_bar_data[feature] = f
    return side_bar_data

# Helper Function
def display_features(df,feature_lookup):
    """
    This function displayes feature names and descriptions (from feature_lookup).
    
    Inputs:
    df (pandas.DataFrame): The input DataFrame to be whose features to be displayed.
    feature_lookup (dict): A dictionary containing the descriptions for the features.
    """
    for idx, col in enumerate(df.columns):
        for f in feature_lookup:
            if f in df.columns:
                st.markdown('Feature %d - %s'%(idx, feature_lookup[col]))
                break
            else:
                st.markdown('Feature %d - %s'%(idx, col))
                break

# Helper Function
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    
    """
    This function fetches a dataset from a URL, saves it in .tgz format, and extracts it to a specified directory path.
    
    Inputs:
    
    housing_url (str): The URL of the dataset to be fetched.
    housing_path (str): The path to the directory where the extracted dataset should be saved.
    """
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

###################### FETCH DATASET #######################

# Create two columns for dataset upload
# Call functions to upload data or restore dataset
col1, col2 = st.columns(2)
# with(col1):
# with(col2):

data=None
if data:
    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')

    # Load dataset
    #df = load_dataset(...)

    # Restore dataset if already in memory

    # Display feature names and descriptions (from feature_lookup)
    #display_features(...,feature_lookup)
    
    # Display dataframe as table using streamlit dataframe function

    # Select feature to explore

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection

    # Draw plots including Scatterplots, Histogram, Lineplots, Boxplot
    
    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")

    # Collect features for correlation analysis using multiselect

    # Compute correlation between selected features 
    #correlation = compute_correlation(...)
    #st.write(correlation) 

    # Display correlation of all feature pairs 
    
    st.markdown('Continue to Preprocess Data')
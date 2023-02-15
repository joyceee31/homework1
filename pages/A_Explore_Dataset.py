import streamlit as st                  
import pandas as pd
import numpy as np
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
    #error checking - check for string
    df = pd.read_csv(data)

    return df

# Checkpoint 2
def compute_correlation(X,features):
    """
    Input: X is pandas dataframe, features is a list of feature name (string) ['age','height']
    Output: correlation coefficients between one or more features
    """
    
    correlation = X[features].corr() #feature is name of feature (string type)
   
    return correlation

# Helper Function
def user_input_features(df):
    """
    Input: pnadas dataframe containing dataset
    Output: dictionary of sidebar filters on features
    """
    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    # categorical_columns = list(df.select_dtypes(['string']).columns)
    side_bar_data = {}
    for feature in numeric_columns:
        try:
            f = st.sidebar.slider(str(feature), float(df[str(feature)].min()), float(df[str(feature)].max()), float(df[str(feature)].mean()))
        except Exception as e:
            print(e)
        side_bar_data[feature] = f

    # for feature in categorical_columns:
    #     try:
    #         f = st.sidebar.selectbox(str(feature))
    #     except Exception as e:
    #         print(e)
    #     side_bar_data[feature] = f
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
with(col1): #uploading from local machine
    data = st.file_uploader('Upload your data', type = ['csv'])
with(col2): #uoload from cloud
    data_path = st.text_input("Eneter data url", "", key = "data_url")


    if(data_path):
        fetch_housing_data()
        data = os.path.join(HOUSING_PATH, "housing.csv")
        st.write("You entered: ", data_path)

if data:
    ###################### EXPLORE DATASET #######################
    st.markdown('### Explore Dataset Features')

    # Load dataset
    df = load_dataset(data)
    st.write(df)

    # Restore dataset if already in memory
    st.session_state['house_df'] = df

    # Display feature names and descriptions (from feature_lookup)
    display_features(df,feature_lookup)
    
    # Display dataframe as table using streamlit dataframe function

    # Select feature to explore

    ###################### VISUALIZE DATASET #######################
    st.markdown('### Visualize Features')

    # Specify Input Parameters
    st.sidebar.header('Specify Input Parameters')

    # Collect user plot selection
    st.sidebar.header('Select type of chart')
    chart_select = st.sidebar.selectbox(
        label = 'Types of charts',
        options = ['Scatterplot', 'Historgram', 'Boxplot', 'Lineplot']
    )

    st.write(chart_select)
    numeric_columns = list(df.select_dtypes(['float','int']).columns)
    categorical_columns = list(df.select_dtypes(['string']).columns)
    # Draw plots including Scatterplots, Histogram, Lineplots, Boxplot
    if(chart_select == 'Scatterplot'):
        try: 
            x_values = st.sidebar.selectbox('X axis', options = numeric_columns)
            y_values = st.sidebar.selectbox('Y axis', options = numeric_columns)
            side_bar_data = user_input_features(df)
            plot = px.scatter(df, x = x_values, y = y_values, range_x = [df[x_values].min(),
                                side_bar_data[x_values]],
                                range_y = [df[y_values].min(), side_bar_data[y_values]])

            st.write(plot)
        except Exception as e:
            print(e)

    if(chart_select == 'Historgram'):
        try: 

            x_values = st.sidebar.selectbox('X axis', options=numeric_columns)
            fig = px.histogram(df, x=x_values)
            st.plotly_chart(fig)

        except Exception as e:
            print(e)

    if(chart_select == 'Boxplot'):
        try: 
            x_values = st.sidebar.selectbox('X axis', options = numeric_columns)
            side_bar_data = user_input_features(df)
            fig = px.box(df, x=x_values)
            st.plotly_chart(fig)
            # plot = px.histogram(df, x = x_values, nbins=30)
            # st.write(plot)
        except Exception as e:
            print(e)

    if (chart_select == 'Lineplot'):
        try: 
            x_value = st.sidebar.selectbox('X axis', options=numeric_columns)
            y_value = st.sidebar.selectbox('Y axis', options=numeric_columns)
            plot = px.line(df, x=x_value, y=y_value)
            st.write(plot)
        except Exception as e:
            print(e)


    ###################### CORRELATION ANALYSIS #######################
    st.markdown("### Looking for Correlations")

    # Collect features for correlation analysis using multiselect
    st.markdown("### Select secondary features for visualize of correlation analysis")

    # Use Streamlit to allow the user to select up to 4 features
    st.header("Select up to 4 features for Correlation")
    selected_features = st.multiselect("Select features to view the scatter matrix", df.columns.tolist(), default=df.columns.tolist()[:2])

    # Compute the correlation for the selected features
    correlation = compute_correlation(df, selected_features)

    # Display the correlation to the user
    st.write("The correlation matrix between the selected features is:")
    st.write(correlation)





    # Compute correlation between selected features 
    # correlation = compute_correlation(df, select_features)

    # st.write(correlation)
    # Display correlation of all feature pairs 

    if selected_features:
        try:

            corr_matrix = df[selected_features].corr()
            st.write(corr_matrix)

            # sns.set_style("darkgrid")
            # scatter_matrix = sns.pairplot(df[selected_features],diag_kind="hist", diag_kws=dict(bins=30))
            # scatter_matrix = sns.pairplot(df[selected_features])
            # st.pyplot(scatter_matrix.fig)
            # scatter_matrix = px.scatter_matrix(df[selected_features], dimensions=selected_features, labels={col: col for col in selected_features})
            # scatter_matrix.update_traces(diagonal_visible=True)
            # Create scatter matrix
#             scatter_matrix = px.scatter_matrix(
#             df[selected_features], 
#             dimensions=selected_features, 
#             color=selected_features[0],  # set color to first selected feature
#             labels={col: col for col in selected_features},
#             opacity=0.7
# )

#             # Set diagonal to histogram
#             scatter_matrix.update_traces(diagonal_visible=True)
            # fig = pd.plotting.scatter_matrix(df[selected_features], alpha=0.2)
            # st.pyplot(fig
            fig = scatter_matrix(df[selected_features], figsize=(12,8)) 
            st.pyplot(fig[0][0].get_figure()) 
            # st.plotly_chart(scatter_matrix)
      
        except Exception as e:
            st.write(e)
   
    st.markdown('Continue to Preprocess Data')
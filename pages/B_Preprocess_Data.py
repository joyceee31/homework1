import numpy as np                      # pip install numpy
import pandas as pd                     # pip install pandas
import matplotlib.pyplot as plt        # pip install matplotlib
from sklearn.model_selection import train_test_split
import streamlit as st                  # pip install streamlit
import plotly.express as px


st.markdown("# Practical Applications of Machine Learning (PAML)")

#############################################

st.markdown("### Homework 1 - End-to-End ML Pipeline")

#############################################

st.markdown('# Preprocess Dataset')

#############################################

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

def restore_dataset():
    """
    Input: 
    Output: 
    """
    if 'house_df' in st.session_state:
        data = st.session_state['house_df']
    else:
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
             data = load_dataset(data)
    
    return data

# Checkpoint 3
def remove_features(X,removed_features):
    """
    Input: 
    Output: 

    """
    X = X.drop(columns=removed_features)

    return X

# Checkpoint 4
def impute_dataset(X, impute_method):
    """
    Input: 
    Output: 
    """
    # impute_method_lower = [x.lower() for x in impute_method]
    if impute_method == "Zero":
        X.fillna(0,inplace=True )
    elif impute_method== "Mean":
        X.fillna(X.mean(), inplace=True)
    elif impute_method == "Median":
        X.fillna(X.median(), inplace=True)
    else:
        raise ValueError("Invalid imputation method. Options are: 'zero', 'mean', 'median'.")

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
    stats_feature_select = [feat.lower() for feat in stats_feature_select]
    stats_select = [stat.lower() for stat in stats_select]

    for feature in stats_feature_select:
        output_str = ''
        for stat in stats_select:
            if stat == 'mean':
                out_dict['mean'] = round(X[feature].mean(), 2)
                output_str += f"Mean of {feature}: {out_dict['mean']}\n"
            if stat == 'median':
                out_dict['median'] = round(X[feature].median(), 2)
                output_str += f"Median of {feature}: {out_dict['median']}\n"
            if stat == 'min':
                out_dict['min'] = round(X[feature].min(), 2)
                output_str += f"Minimum of {feature}: {out_dict['min']}\n"
            if stat == 'max':
                out_dict['max'] = round(X[feature].max(), 2)
                output_str += f"Maximum of {feature}: {out_dict['max']}\n"
    
        st.write(output_str)

    return output_str, out_dict

# Checkpoint 6
def split_dataset(X, number):

    train=[]
    test=[]
    train, test = train_test_split(X, test_size=float(number/100), random_state=np.random.seed(42))
    train_size = len(train)
    test_size = len(test)
    total_size = len(X)

    # Print the size of the training and testing datasets
    st.markdown("The training dataset contains {} line observations ({:.2f}% of total data)".format(train_size, 100*train_size/total_size))
    st.markdown("The testing dataset contains {} line observations ({:.2f}% of total data)".format(test_size, 100*test_size/total_size))
    

    return train, test

# Restore Dataset
df = restore_dataset()




if df is not None:

   
   
    st.markdown('View initial data with missing values or invalid inputs')

    # Display original dataframe
    st.write(df)

    # Show summary of missing values including the 1) number of categories with missing values, average number of missing values per category, and Total number of missing values
    # Get the number of missing values in each column
    missing_values = df.isna().sum()
    missing_values.rename("Number of Missing data", inplace=True)
    # Get the number of categories with missing values
    categories_with_missing = missing_values[missing_values > 0].count()
    
    # Get the average number of missing values per category
    avg_missing_values = missing_values.mean()
    
    # Get the total number of missing values
    total_missing = missing_values.sum()
    
    # Display the summary of missing values
    st.markdown('Number of categories with missing values: {0:.2f}'.format(categories_with_missing))
    st.markdown('Average number of missing values per category: {0:.2f}'.format(avg_missing_values))
    st.markdown('Total number of missing values: {0:.2f}'.format(total_missing))
    st.write("Number of missing values in each column:")
    st.write(missing_values)

    ############################################# MAIN BODY #############################################
    # st.markdown('Number of categories with missing values: {0:.2f}'.format())
    numeric_columns = list(df.select_dtypes(['float','int']).columns)

    # Provide the option to select multiple feature to remove using Streamlit multiselect

    # Remove the features using the remove_features function
    st.markdown('### Remove irrelevant data')
    # Get the list of columns
    columns = df.columns

    # Get the list of features to be removed using the Streamlit multiselect function
    removed_features = st.multiselect('Select features to remove', columns)

    # Update the dataframe by removing the selected features
    df = remove_features(df, removed_features)
   

    # Display updated dataframe
    st.write(df)
    # Clean dataset
    st.markdown('### Impute data')
    st.markdown('Transform missing values to 0, mean, or median')

    # Use selectbox to provide impute options {'Zero', 'Mean', 'Median'}
    st.markdown('Select cleaning method')
    # Call impute_dataset function to resolve data handling/cleaning problems by calling impute_dataset

    impute_method = st.selectbox(
        label='Impute method:',
        options=['Zero', 'Mean', 'Median']
    )
    # Display updated dataframe
    if st.button('Impute Missing Values'):
        X_imputed = impute_dataset(df, impute_method)
        st.write('Data with missing values imputed:')
        st.write(X_imputed)




    # Descriptive Statistics 
    st.markdown('### Summary of Descriptive Statistics')

    # Provide option to select multiple feature to show descriptive statistics using Streamit multiselect
    selectmultiple  = st.multiselect('Select features for statistics', columns)
    # Provide option to select multiple descriptive statistics to show using Streamit multiselect

    selectforstas = st.multiselect(
        label='Select statistics to display',
        options=['median', 'max', 'mean', 'min']
    )
    # Compute Descriptive Statistics including mean, median, min, max
    X_descriptive = compute_descriptive_stats(df,selectmultiple, selectforstas)
    # st.write(X_descriptive)    
    # Display updated dataframe
    # st.markdown('### Result of the imputed dataframe')

    # Split train/test
    st.markdown('### Enter the percentage of test data to use for training the model')

    # Compute the percentage of test and training data
    test_size = st.number_input("Enter the test set (X %): ", value = 20)

    # Print dataset split result
    X_train, X_test = split_dataset(df, test_size)



    # Save state of train and test split

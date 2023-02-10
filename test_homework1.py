from pages import A_Explore_Dataset, B_Preprocess_Data
import pandas as pd
import numpy as np

######## CheckPoint1 ##############
student_filepath="datasets/housing/housing.csv"
test_filepath= "test_dataframe_file/inital_housing.csv"
s_dataframe = pd.read_csv(student_filepath)
e_dataframe = pd.read_csv(test_filepath)
e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
numeric_columns = list(e_X.select_dtypes(['float', 'int']).columns)
nan_colns = e_X.columns[e_X.isna().any()].tolist()



def test_load_dataframe():
    s_dataframe = A_Explore_Dataset.load_dataset(student_filepath)
    e_dataframe =pd.read_csv(test_filepath)
    pd.testing.assert_frame_equal(s_dataframe,e_dataframe)

###################################


######## CheckPoint2 ##############
## You have to round to two decimal places
def test_compute_descriptive_stats():
    _, out_dict=B_Preprocess_Data.compute_descriptive_stats(e_dataframe,['latitude'],['Mean','Median','Max','Min'])
    e_dict = {
        'mean': 35.63,
        'median': 34.26,
        'max': 41.95,
        'min': 32.54
    }
    assert out_dict==e_dict
    

###################################


######## CheckPoint3 ##############
def test_compute_corr():
    e_corr = np.array([[1, -0.0360996], [-0.0360996, 1]])
    test_corr = A_Explore_Dataset.compute_correlation(e_dataframe,['latitude','total_rooms'])
    print(test_corr)
    print(e_corr)
    assert np.allclose(e_corr, test_corr)
    # assert test_corr == 

###################################


######## CheckPoint4 ##############

def test_impute_zero():
    e_zero_df = pd.read_csv("test_dataframe_file/Zero.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_zero_df = B_Preprocess_Data.impute_dataset(e_X, 'Zero')#, numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_zero_df,s_zero_df)


def test_impute_median():
    e_median_df = pd.read_csv("test_dataframe_file/Median.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_median_df = B_Preprocess_Data.impute_dataset(e_X, 'Median')#, numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_median_df,s_median_df)


def test_impute_mean():
    e_mean_df = pd.read_csv("test_dataframe_file/Mean.csv")
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]

    s_mean_df = B_Preprocess_Data.impute_dataset(e_X, 'Mean')#, numeric_columns, nan_colns)
    pd.testing.assert_frame_equal(e_mean_df,s_mean_df)


###################################


######## CheckPoint5 ##############

def test_remove_features():
    e_remove= pd.read_csv("./test_dataframe_file/remove.csv")
    e_X= e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_remove = B_Preprocess_Data.remove_features(e_X, ['latitude', 'longitude'])
    pd.testing.assert_frame_equal(s_remove,e_remove)

###################################


######## CheckPoint6 ##############

def test_split_dataset():
    e_X = e_dataframe.loc[:, ~e_dataframe.columns.isin(['median_house_value'])]
    s_split_train, s_split_test = B_Preprocess_Data.split_dataset(e_X,30)

    assert s_split_train.shape == (14448, 9)

###################################

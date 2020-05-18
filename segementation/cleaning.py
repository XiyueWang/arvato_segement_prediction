import numpy as np
import pandas as pd



def get_attribute(excel_filepath):
    '''Processes attribute description data
    Args:
        excel - attribute information
    Returns:
        dict - dictionary contains attribute names and Values
    '''
    att_values = pd.read_excel(filepath, header=1)
    att_values = att_values.fillna('')
    att_values.drop('Unnamed: 0', axis=1, inplace=True)

    # find unique values of each attributes
    idx = []
    for i in range(att_values.shape[0]):
        if len(att_values.Attribute[i]) > 0:
            idx.append(i)

    attr_dict = {}
    for i in range(len(idx)-1):
        key_name = att_values.Attribute[idx[i]]
        attr_dict[key_name] = att_values.Value[idx[i]:idx[i+1]].tolist()
        last_key = att_values.Attribute[idx[-1]]
        attr_dict[last_key] = att_values.Value[idx[i]:].tolist()

    return attr_dict

def check_value(x):
    '''check the values for missing value'''
    if type(x) == float:
        return x
    elif x == 'X' or (x == 'XX'):
        return np.nan
    else:
        return float(x)

def clean_data(filepath, attr_dict=attr_dict):
    '''Processes data
        - Converts missing values to np.nan using loaded features table
        - Drops unwanted columns and rows
        - Convert mixed datatype to float
        - Perfroms feature enginerring

    Args:
        df (pd.Dataframe): data to be cleaned
        feat_info (to_dict): feature information

    Returns:
        cleaned_df (pd.Dataframe): cleaned rows
    '''
    clean_df = df.copy()

    cols = clean_df.columns[18:20]
    for col in cols:
        clean_df[col] = clean_df[col].apply(lambda x: check_value(x))


    col_nulls = clean_df.isnull().sum()/clean_df.shape[0]
    row_nulls = clean_df.isnull().sum(axis=1)/clean_df.shape[1]
    # remove columns with more than 20% nulls in azdias dataframe
    cols = col_nulls[col_nulls<=0.2].index.tolist()
    # remove rows that has more than 10% nulls
    rows = row_nulls[row_nulls<=0.1].index.tolist()
    clean_df = clean_df.loc[rows, cols]

    # remove columns with kba
    kba_cols = clean_df.columns[clean_df.columns.str.startswith('KBA')]
    clean_df.drop(list(kba_cols), axis=1, inplace=True)

    # onky keep rows with meaningful attributes
    common_col = set(attr_dict.keys()).intersection(set(clean_df.columns))
    common_col = list(common_col)
    clean_df = clean_df.loc[:,common_col]

    # get the dummy for region
    dummy = pd.get_dummies(clean_df['OST_WEST_KZ'])
    clean_df.drop('OST_WEST_KZ', axis=1, inplace=True)
    clean_df = pd.concat([clean_df, dummy], axis=1)

    # re-engineer PRAEGENDE_JUGENDJAHRE
    to_replace = {1:4, 2:4, 3:5, 4:5, 5:6, 6:6, 7:6, 8:7, 9:7, 10:8, 11:8, 12:8, 13:8, 14:9, 15:9}
    clean_df['decade'] = clean_df['PRAEGENDE_JUGENDJAHRE'].replace(to_replace)

    clean_df.drop(['CAMEO_DEU_2015', 'PRAEGENDE_JUGENDJAHRE'] , axis=1, inplace=True)

    return clean_df

def fill_null(clean_df):
    '''This function takes the cleaned df, fill numerical columns with mean, and
    categorical columns with median.
    Args: clean df
    Return: df without missing values
    '''
    # select columns with numerical  values
    num_col = []
    for key, item in attr_dict.items():
        if item[0] == '…':
            num_col.append(key)
        # fill mean for numerical columns
    for col in num_col:
        try:
            az_mean = clean_df[col].mean()
            clean_df[col] = clean_df[col].fillna(az_mean)
        except KeyError:
            continue
        # fill median for categorical columns
        # fill all other columns with mode
        for col in clean_df.columns:
            try:
                az_median = clean_df[col].median()
                clean_df[col] = clean_df[col].fillna(az_median)
            except KeyError:
                continue

    return clean_df

def  clean_customers(df_cus):
    '''Processes data
        - Converts missing values to np.nan using loaded features table
        - Drops unwanted columns and rows
        - Convert mixed datatype to float
        - Perfroms feature enginerring

    Args:
        df (pd.Dataframe): data to be cleaned

    Returns:
        cleaned_df (pd.Dataframe): cleaned rows
    '''
    cus_null = df_cus.isnull().sum(axis=1)/df_cus.shape[1]
    rows = cus_null[cus_null<=0.1].index.tolist()
    df_cus = df_cus.iloc[rows, :]

    cols = df_cus.columns[18:20]
    for col in cols:
        df_cus[col] = df_cus[col].apply(lambda x: check_value(x))
    # get dummy regions
    dummy = pd.get_dummies(df_cus['OST_WEST_KZ'])
    df_cus.drop('OST_WEST_KZ', axis=1, inplace=True)
    df_cus = pd.concat([df_cus, dummy], axis=1)

    # replace decade
    to_replace = {1:4, 2:4, 3:5, 4:5, 5:6, 6:6, 7:6, 8:7, 9:7, 10:8, 11:8, 12:8, 13:8, 14:9, 15:9}
    df_cus['decade'] = df_cus['PRAEGENDE_JUGENDJAHRE'].replace(to_replace)

    # drop unused row
    df_cus.drop(['CAMEO_DEU_2015', 'PRAEGENDE_JUGENDJAHRE'] , axis=1, inplace=True)
    return df_cus


if __name__ == '__main__':
    '''Cleans and saves data to new files. New files will have "_clean"
    and "_dropped" appended to original data filename.

    Args:
        data_filepath (str): filepath to data
        features_filepath (str): filepath to feature information
    '''

    # Load feature info
    attr_dict = get_attribute('DIAS Attributes - Values 2017.xlsx')

    # Load data
    print('Loading data...')
    df = pd.read_csv('azdias.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)

    print('Cleaning data...')
    clean_df = clean_data(df)

    print('Filling null values...')
    clean_df = fill_null(clean_df)

    print('Cleaning customer dataframe...')
    df_cus = pd.read_csv('custromers.csv')
    customers_clean = clean_customers(df_cus)
    cols = clean_df.columns
    customers_clean = df_cus.loc[:, cols]
    customers_clean = fill_null(customers_clean)

    print('Writing clean data...')

    clean_df.to_csv('azdias_clean.csv')
    customers_clean.to_csv('customers_clean.csv')

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

def get_attribute(excel_filepath='DIAS Attributes - Values 2017'):
    '''Processes attribute description data
    Args:
        excel - attribute information
    Returns:
        dict - dictionary contains attribute names and Values
    '''
    att_values = pd.read_excel(excel_filepath, header=1)
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


def clean_data(df, attr_dict):
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
    cols = col_nulls[col_nulls<=0.22].index.tolist()
    clean_df = clean_df.loc[:, cols]

    # remove columns with kba
    kba_cols = clean_df.columns[clean_df.columns.str.startswith('KBA')]
    clean_df.drop(list(kba_cols), axis=1, inplace=True)

    # get the dummy for region
    dummy = pd.get_dummies(clean_df['OST_WEST_KZ'])
    clean_df.drop('OST_WEST_KZ', axis=1, inplace=True)
    clean_df = pd.concat([clean_df, dummy], axis=1)

    # re-engineer PRAEGENDE_JUGENDJAHRE
    to_replace = {1:4, 2:4, 3:5, 4:5, 5:6, 6:6, 7:6, 8:7, 9:7, 10:8, 11:8, 12:8, 13:8, 14:9, 15:9}
    clean_df['decade'] = clean_df['PRAEGENDE_JUGENDJAHRE'].replace(to_replace)

    clean_df.drop(['CAMEO_DEU_2015', 'PRAEGENDE_JUGENDJAHRE', 'D19_LETZTER_KAUF_BRANCHE', 'EINGEFUEGT_AM'] , axis=1, inplace=True)

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

def build_model(model):
    '''
    Creates pipeline with  two steps: column transformer (ct) introduced in preprocessing step and classifier (model).

    Input:
        scaler: scale the features
        model: object type that implements the “fit” and “predict” methods

    Output:
        pipeline: object type with "fit" and "predict" methods
    '''

    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('clf', model)
                        ])
    parameters = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.001, 0.01, 0.1],
        'clf__boosting_type': ['gbdt','dart'],
        'clf__num_leaves': [31, 62]
        #'clf__base_estimator__min_samples_split': [2, 5, 10],
        #'clf__base_estimator__max_depth': [1, 3, 5]
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv

def clean_test(df_cus):
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

    cols = df_cus.columns[18:20]
    for col in cols:
        df_cus[col] = df_cus[col].apply(lambda x: ml.check_value(x))
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


def evaluate_model(cv, X_test, y_test):
    """Draw ROC curve for the model
    Args:
        Classification Model
        X_test, y_test, Array-like
    return: ROC curve and model pickles
    """
    y_pred = cv.predict_proba(X_test)[:,1]
    print('\nBest Parameters:', cv.best_params_)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr) # compute area under the curve
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % (roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # create the axis of thresholds (scores)
    ax2 = plt.gca().twinx()
    ax2.plot(fpr, thresholds, markeredgecolor='r',linestyle='dashed', color='r')
    ax2.set_ylabel('Threshold',color='r')
    ax2.set_ylim([thresholds[-1],thresholds[0]])
    ax2.set_xlim([fpr[0],fpr[-1]])

    print('Saving model...')
    f = open('parameters'+'.pkl', 'wb')
    pickle.dump(cv.best_params_, f)

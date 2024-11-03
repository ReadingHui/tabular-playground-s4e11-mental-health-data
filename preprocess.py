import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.base import TransformerMixin
from typing import Literal
from eda_util import feature_target


class DataImport:
    def get_train_test(path: str, index_col: int = None, target: str = None, random_state: int = 1048576, verbose: int = 1) -> tuple[pd.DataFrame]:
        train_csv = pd.read_csv(path, index_col=index_col)
        X, y = feature_target.feature_target_split(train_csv, target=target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.1, random_state=random_state)
        if verbose > 0:
            print(f'Shape of X_train is: {X_train.shape}; shape of y_train is: {y_train.shape}')
            print(f'Shape of X_test is: {X_test.shape}; shape of y_test is: {y_test.shape}')
        return X_train, X_test, y_train, y_test

class Outliers:
    def get_income_outliers(X: pd.DataFrame):
        loan_income = X[['person_income' ,'loan_percent_income', 'loan_amnt']].copy()
        loan_income['cal_loan_amnt'] = loan_income['person_income'] * loan_income['loan_percent_income']
        loan_income['diff'] = loan_income['cal_loan_amnt'] - loan_income['loan_amnt']
        loan_income['percent_diff'] = round(loan_income['diff'] / loan_income['cal_loan_amnt'] * 100, 2)
        return loan_income[abs(loan_income['percent_diff']) > 50].index
    
    def get_age_outliers(X: pd.DataFrame):
        work_time = X[['person_age', 'person_emp_length', 'cb_person_cred_hist_length']].copy()
        work_time['start_working_age'] = work_time['person_age'] - work_time['person_emp_length']
        work_time['start_cred'] = work_time['person_age'] - work_time['cb_person_cred_hist_length']
        return work_time[work_time['start_working_age'] < 14].index # US legal working age starts at 14
    
    def remove_outliers(X: pd.DataFrame, y: pd.Series | np.ndarray, verbose: int = 1) -> tuple:
        income_idx = Outliers.get_income_outliers(X)
        age_idx = Outliers.get_age_outliers(X)
        drop_ind = np.concatenate([income_idx, age_idx])
        X, y = X.copy(), y.copy()
        X, y = X.drop(drop_ind), y.drop(drop_ind)
        if verbose > 0:
            print(f'New shape of X_train is: {X.shape}; New shape of y_train is: {y.shape}')
        return X, y
    
class CategoryEncoder(TransformerMixin):
    def __init__(self, cat_features: list[str], method: Literal['ordinal', 'one_hot']) -> None:
        self.cat_features = cat_features
        self.method = method
        if method == 'ord':
            ct_list = [
                    ('cat_encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan), self.cat_features)
                ]
        elif method == 'one_hot':
            ct_list = [
                    ('cat_encoder', OneHotEncoder(sparse_output=False, handle_unknown='infrequent_if_exist'), self.cat_features)
                ]

        
        self.encoder = ColumnTransformer(
            ct_list, 
            remainder='passthrough'
        )
        self.encoder.set_output(transform='pandas')

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray = None) -> None:
        self.encoder.fit(X)
        return self
    
    def transform(self, X:pd.DataFrame, y: pd.Series | np.ndarray = None) -> pd.DataFrame:
        X = X.copy()
        X = self.encoder.transform(X)
        X = X.rename(columns=lambda x:x.split('__', 1)[-1]) # Discard the 'cat_encoder__' and 'passthrough__' prefices
        return X

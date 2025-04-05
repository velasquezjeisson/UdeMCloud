import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

def main(args):
    df = pd.read_csv(args.input_file, delimiter=',', header=0)
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    pd.set_option('display.max_columns', 500)
    df.dropna(inplace=True)
    df.set_index('customerID', inplace=True)

    num_vars = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
    ohe_vars = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
    
    ohe = OneHotEncoder(sparse_output=False, drop='first')  
    oe_Contract = OrdinalEncoder(categories=[['Month-to-month', 'One year', 'Two year']])
    ss = StandardScaler()

    ohe_array = ohe.fit_transform(df[ohe_vars])
    ohe_df = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(ohe_vars), index=df.index)
    df.drop(columns=ohe_vars, inplace=True)
    df = pd.concat([df, ohe_df], axis=1)

    df['Contract'] = oe_Contract.fit_transform(df[['Contract']])

    df[num_vars] = ss.fit_transform(df[num_vars])

    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
    df.dropna(subset=['Churn'], inplace=True)

    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Verificación
    assert X.dtypes.apply(lambda dt: np.issubdtype(dt, np.number)).all(), "X contiene columnas no numéricas"
    assert y.apply(lambda v: isinstance(v, (int, float))).all(), "y contiene valores no numéricos"

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)

    os.makedirs(args.train_output, exist_ok=True)
    os.makedirs(args.validation_output, exist_ok=True)
    os.makedirs(args.test_output, exist_ok=True)
    
    train_df = pd.concat([y_train, X_train], axis=1)
    val_df = pd.concat([y_val, X_val], axis=1)
    test_df = pd.concat([y_test, X_test], axis=1)
    
    train_df.to_csv(os.path.join(args.train_output, 'train.csv'), index=False, header=False, sep=',')
    val_df.to_csv(os.path.join(args.validation_output, 'validation.csv'), index=False, header=False, sep=',')
    #test_df.to_csv(os.path.join(args.test_output, 'test.csv'), index=False, header=False, sep=',')
    X_test.to_csv(os.path.join(args.test_output, 'X_test.csv'), index=False, header=False, sep=',')
    y_test.to_csv(os.path.join(args.test_output, 'y_test.csv'), index=False, header=False, sep=',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--train_output', type=str, required=True)
    parser.add_argument('--validation_output', type=str, required=True)
    parser.add_argument('--test_output', type=str, required=True)
    args = parser.parse_args()
    main(args)

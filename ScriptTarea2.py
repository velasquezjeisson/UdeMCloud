import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split

def main(args):
    # Leer el archivo CSV
    df = pd.read_csv(args.input_file, delimiter=',', header=0, on_bad_lines='skip')
    df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df.dropna(inplace=True)
    # Separar las características y la variable objetivo
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Dividir los datos en conjuntos de entrenamiento, validación y prueba
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=1)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=1)
    
    # Guardar los conjuntos de datos en archivos CSV
    X_train.to_csv(os.path.join(args.train_output, 'X_train.csv'), index=False)
    y_train.to_csv(os.path.join(args.train_output, 'y_train.csv'), index=False)
    X_val.to_csv(os.path.join(args.validation_output, 'X_val.csv'), index=False)
    y_val.to_csv(os.path.join(args.validation_output, 'y_val.csv'), index=False)
    X_test.to_csv(os.path.join(args.test_output, 'X_test.csv'), index=False)
    y_test.to_csv(os.path.join(args.test_output, 'y_test.csv'), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--train_output', type=str, required=True)
    parser.add_argument('--validation_output', type=str, required=True)
    parser.add_argument('--test_output', type=str, required=True)
    args = parser.parse_args()
    
    main(args)

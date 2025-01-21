#dependencies
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
#machine learning
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet

# guess data types
def guess_data_types(df):
    # create a dictionary to hold the data types
    data_types = {}
    # loop through each column
    for col in df.columns:
        # get the unique values in the column
        unique_values = df[col].dropna().unique()
        # check if the column has any values
        if len(unique_values) > 0:
            # check if the column has any strings
            if any(isinstance(value, str) for value in unique_values):
                data_types[col] = "string"
            # check if the column has any dates
            elif any(isinstance(value, datetime) for value in unique_values):
                data_types[col] = "date"
            # check if the column has any floats
            elif any(isinstance(value, float) for value in unique_values):
                data_types[col] = "float"
           # check if the column has any integers
            elif all(isinstance(value, (int, np.integer)) for value in unique_values):
                data_types[col] = "integer"
            # if none of the above, assume it's a string
            else:
                data_types[col] = "string"
        # if the column has no values, assume it's a string
        else:
            data_types[col] = "string"

    return data_types

# update data types
def update_data_types(df, data_types):
    # loop through each column and data type
    for col, data_type in data_types.items():
        # check if the column exists in the dataframe
        if col in df.columns:
            # check if the data type is a string
            if data_type == "string":
                df[col] = df[col].astype(str)
            # check if the data type is a date
            elif data_type == "date":
                df[col] = pd.to_datetime(df[col])
            # check if the data type is a float
            elif data_type == "float":
                df[col] = df[col].astype(float)
            # check if the data type is an integer
            elif data_type == "integer":
                try:
                    df[col] = df[col].astype(int)
                except:
                    df[col] = df[col].astype(bool)
    return df

# label encode categorical variables and keep the encoder for later
def label_encode(df, cat_cols):
    # create a dictionary to hold the encoders
    encoders = {}
    # loop through each categorical column
    for col in cat_cols:
        # create a label encoder
        encoder = LabelEncoder()
        # fit the encoder on the column
        df[col] = encoder.fit_transform(df[col])
        # store the encoder in the dictionary
        encoders[col] = encoder
    return df, encoders

# one-hot encode categorical variables and keep the encoder for later
def one_hot_encode(df, cat_cols):
    # create a dictionary to hold the encoders
    encoders = {}
    # loop through each categorical column
    for col in cat_cols:
        # create a one-hot encoder
        encoder = OneHotEncoder()
        # fit the encoder on the column
        encoded = encoder.fit_transform(df[col].values.reshape(-1, 1)).toarray()
        # create a new dataframe with the encoded values
        df_encoded = pd.DataFrame(encoded, columns=[col + "_" + str(i) for i in range(encoded.shape[1])])
        # concatenate the new dataframe with the original dataframe
        df = pd.concat([df, df_encoded], axis=1)
        # drop the original column
        df = df.drop(col, axis=1)
        # store the encoder in the dictionary
        encoders[col] = encoder
    return df, encoders

# scale the data using standard scaler and keep the scaler for later
def scale_data(df, scaler=None):
    # check if a scaler is provided
    if not scaler:
        # create a standard scaler
        scaler = StandardScaler()
        # fit the scaler on the dataframe
        df_scaled = scaler.fit_transform(df)
    else:
        # transform the dataframe using the provided scaler
        df_scaled = scaler.transform(df)
    return df_scaled, scaler

# normalize the data using min-max scaler and keep the scaler for later
def normalize_data(df, scaler=None):
    # check if a scaler is provided
    if not scaler:
        # create a min-max scaler
        scaler = MinMaxScaler()
        # fit the scaler on the dataframe
        df_normalized = scaler.fit_transform(df)
    else:
        # transform the dataframe using the provided scaler
        df_normalized = scaler.transform(df)
    return df_normalized, scaler

# impute missing values using mean, median, or mode
def impute_missing_values(df, strategy="mean"):
    # check if the strategy is mean
    if strategy == "mean":
        # fill missing values with the mean of the column
        df = df.fillna(df.mean(numeric_only=True))
    # check if the strategy is median
    elif strategy == "median":
        # fill missing values with the median of the column
        df = df.fillna(df.median(numeric_only=True))
    # check if the strategy is mode
    elif strategy == "mode":
        # fill missing values with the mode of the column
        for col in df.columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# impute missing values of a categorical variable based on user input or most frequent value
def impute_missing_values_categorical(df, col, value=None):
    # check if a value is provided
    if value:
        # fill missing values with the provided value
        df[col] = df[col].fillna(value)
    else:
        # fill missing values with the most frequent value
        df[col] = df[col].fillna(df[col].mode().iloc[0])
    return df

# select appropriate machine learning problem (regression, binary classification, multi-class classification, time-series)
def select_problem(df, y_col, ts_col=None):
    # check if the ts_col is provided and is a datetime column
    if ts_col and df[ts_col].dtype == "datetime64[ns]":
        return "time-series"
    # check if the target column is a string
    elif df[y_col].dtype == "string" or df[y_col].dtype == "object":
        # check if the target column has only two unique values
        if len(df[y_col].unique()) == 2:
            return "binary-classification"
        else:
            return "multi-class-classification"
    # if none of the above, assume it's a regression problem
    else:
        return "regression"

# split into train, validation, and test sets
def split_data(df, y_col, test_size=0.2, validation_size=0.2):
    # split the dataframe into features and target
    X = df.drop(y_col, axis=1)
    y = df[y_col]
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # split the train set into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=validation_size)
    return X_train, X_val, X_test, y_train, y_val, y_test

# get machine learning models based on the problem type
def get_models(problem):
    # create a dictionary to hold the models
    models = {}
    # check if the problem is regression
    if problem == "regression":
        models["Linear Regression"] = LinearRegression()
        models["Random Forest"] = RandomForestRegressor()
        models["Gradient Boosting"] = GradientBoostingRegressor()
        models["Neural Network"] = MLPRegressor()
    # check if the problem is binary classification
    elif problem == "binary-classification":
        models["Logistic Regression"] = LogisticRegression()
        models["Random Forest"] = RandomForestClassifier()
        models["Gradient Boosting"] = GradientBoostingClassifier()
        models["Neural Network"] = MLPClassifier()
    # check if the problem is multi-class classification
    elif problem == "multi-class-classification":
        models["Logistic Regression"] = LogisticRegression()
        models["Random Forest"] = RandomForestClassifier()
        models["Gradient Boosting"] = GradientBoostingClassifier()
        models["Neural Network"] = MLPClassifier()
    # check if the problem is time-series
    elif problem == "time-series":
        models["ARIMA"] = ARIMA
        models["Prophet"] = Prophet()
    return models

# import training data
data = {
    "feature1": np.linspace(0, 10, 10000) + np.random.uniform(-8, 7, 10000),
    "feature2": np.linspace(0, 5, 10000) + np.random.uniform(-8, 8, 10000),
    "feature3": np.linspace(0, 1, 10000) + np.random.uniform(-1, 1, 10000),
    "target": 3 * (np.linspace(0, 10, 10000) + np.random.uniform(-3, 3, 10000))**2 + 2 * (np.linspace(0, 10, 10000) + np.random.uniform(-1, 1, 10000)) + 1 + np.random.normal(0, 1, 10000) + np.random.uniform(-5, 5, 10000),  # Polynomial relationship with noise and added randomness
    "binary": np.random.choice(['N', 'Y'], size=10000),  # Random binary classification target
    "multiclass": np.random.choice(['A', 'AB', 'B', 'O'], size=10000)  # Random binary classification target
}

target_str = 'multiclass'
df = pd.DataFrame(data)[['feature1', 'feature2', 'feature3', target_str]]

# guess data types whatever
dt = guess_data_types(df)
df = update_data_types(df, dt)

# select the appropriate machine learning problem
problem = select_problem(df, target_str)
models = get_models(problem)

# preprocess the data
df = impute_missing_values(df, strategy="mean")
df = impute_missing_values_categorical(df, "feature1", value="unknown")
cat_cols = [col for col, dtype in dt.items() if dtype == "string" or dtype == "date" or dtype == "object"]
df, encoders = label_encode(df, cat_cols)

# split the data into train, validation, and test sets
X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, target_str)

# fit the models and evaluate on validation and test sets
results = {}
print(problem.title(), '\n')
for name, model in models.items():
    print(name)
    try:
        # fit the model on the training data
        model.fit(X_train, y_train)
        # evaluate the model on the validation data
        val_score = model.score(X_val, y_val)
        # evaluate the model on the test data
        test_score = model.score(X_test, y_test)
        # store the results in the dictionary
        results[name] = {"validation_score": val_score, "test_score": test_score}
        print(f"Validation Score: {val_score}, Test Score: {test_score}")
    except Exception as e:
        print(f"Error with model {name}: {e}")
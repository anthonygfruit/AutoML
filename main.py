#dependencies
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

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
        df[col] = df[col].replace([pd.NaT, None, "None", np.nan, "", float('inf'), -float('inf')], value).fillna(value)
    else:
        # fill missing values with the most frequent value
        mode_value = df[col].mode().iloc[0]
        df[col] = df[col].replace([pd.NaT, None, "None", np.nan, "", float('inf'), -float('inf')], mode_value).fillna(mode_value)

# impute missing values for multiple columns based on a dictionary of col/value pairs
def impute_missing_values_categorical_bulk(df, imputation_dict):
    for col, value in imputation_dict.items():
        df = impute_missing_values_categorical(df, col, value)
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
def split_data(X, y, test_size=0.2, validation_size=0.2):
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

# plot correlations
def corplot(df, target_str):
    # Plot target_str variable against all other variables in separate subplots in the same figure
    num_features = len(df.columns.difference([target_str]))
    ncols = int(np.ceil(np.sqrt(num_features)))  # Dynamic number of columns
    nrows = int(np.ceil(num_features / ncols))  # Dynamic number of rows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows), sharey=True)
    axes = axes.flatten()  # Flatten axes for dynamic indexing
    colors = plt.cm.tab10.colors  # Use a colormap for different colors
    for ax, (feature, color) in zip(axes, zip(df.columns.difference([target_str]), colors)):
        ax.scatter(df[feature], df[target_str], alpha=0.5, color=color)
        ax.set_title(f'{feature} vs {target_str}')
        ax.set_xlabel(feature)
        ax.set_ylabel(target_str)
    for ax in axes[num_features:]:
        ax.set_visible(False)  # Hide unused subplots
    plt.tight_layout()
    plt.show()

# create frequency table for NaN and None values
def freq_table(df):
    return df.isna().sum().to_frame('NaN Values').join(
        df.apply(lambda x: (x == None).sum() + (x == "None").sum()).to_frame('None Values'))

# identify categorical/date columns
def get_cat_cols(df):
    return [col for col in df.columns if df[col].dtype == "string" or df[col].dtype == "date" or df[col].dtype == "object"]

# fit and run cross-validation
def auto_fit(models, problem, X_train, y_train, X_val, y_val):
    best_model_name = None
    best_model = None
    best_val_score = float("inf") if problem in ["regression", "time-series"] else float("-inf")
    best_val_metrics = {}

    for name, model in models.items():
        try:
            # fit the model on the training data
            model.fit(X_train, y_train)
            # make predictions
            y_val_pred = model.predict(X_val)

            # determine evaluation metrics based on problem type
            if problem == "binary-classification":
                val_metrics = {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "f1": f1_score(y_val, y_val_pred),
                    "precision": precision_score(y_val, y_val_pred),
                    "recall": recall_score(y_val, y_val_pred)
                }
                score = val_metrics["accuracy"]  # Use accuracy as a comparison metric
            elif problem == "multi-class-classification":
                val_metrics = {
                    "accuracy": accuracy_score(y_val, y_val_pred),
                    "f1_macro": f1_score(y_val, y_val_pred, average='macro'),
                    "f1_weighted": f1_score(y_val, y_val_pred, average='weighted')
                }
                score = val_metrics["accuracy"]  # Use accuracy as a comparison metric
            elif problem in ["regression", "time-series"]:
                val_metrics = {
                    "mean_squared_error": mean_squared_error(y_val, y_val_pred),
                    "mean_absolute_error": mean_absolute_error(y_val, y_val_pred),
                    "r2_score": r2_score(y_val, y_val_pred)
                }
                score = val_metrics["mean_squared_error"]  # Use MSE as a comparison metric
            else:
                val_metrics = {}
                score = None

            # Determine the best model based on the validation metric
            if (problem in ["binary-classification", "multi-class-classification"] and score > best_val_score) or \
                (problem in ["regression", "time-series"] and score < best_val_score):
                best_model_name = name
                best_model = model
                best_val_score = score
                best_val_metrics = val_metrics

            print(f"Validation Metrics for {name}: {val_metrics}")
        except Exception as e:
            print(f"Error fitting model {name}: {e}")

    return (best_model_name, best_model)

# test evaluation for the best model
def test_fit(best_model_tuple, problem, X_test, y_test):
    best_model = best_model_tuple[1]
    best_model_name = best_model_tuple[0]

    if best_model:
        print(f"\nBest Model: {best_model_name}")
        y_test_pred = best_model.predict(X_test)
        if problem == "binary-classification":
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "f1": f1_score(y_test, y_test_pred),
                "precision": precision_score(y_test, y_test_pred),
                "recall": recall_score(y_test, y_test_pred)
            }
        elif problem == "multi-class-classification":
            test_metrics = {
                "accuracy": accuracy_score(y_test, y_test_pred),
                "f1_macro": f1_score(y_test, y_test_pred, average='macro'),
                "f1_weighted": f1_score(y_test, y_test_pred, average='weighted')
            }
        elif problem in ["regression", "time-series"]:
            test_metrics = {
                "mean_squared_error": mean_squared_error(y_test, y_test_pred),
                "mean_absolute_error": mean_absolute_error(y_test, y_test_pred),
                "r2_score": r2_score(y_test, y_test_pred)
            }
        else:
            test_metrics = {}

        print(f"Test Metrics for Best Model ({best_model_name}): {test_metrics}")

        return test_metrics

# run automl
def main(df, target_str, imputation_dict):
    # guess data types
    dt = guess_data_types(df)
    df = update_data_types(df, dt)
    # impute missing values
    if imputation_dict is not None:
        df = impute_missing_values_categorical_bulk(df, imputation_dict)
    df = impute_missing_values(df, strategy="mean")
    # show frequency table for Nan/None values
    print(freq_table(df))
    # select appropriate machine learning problem
    problem = select_problem(df, target_str)
    # get machine learning models based on the problem type
    models = get_models(problem)
    # visualize correlations
    corplot(df, target_str)
    # find categorical variables
    cat_cols = get_cat_cols(df)
    # encode categorical variables
    df, encoders = label_encode(df, cat_cols)
    # scale the data
    y = df[target_str]
    X, scaler = scale_data(df.drop(target_str, axis=1))
    # split the data into train, validation, and test sets
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, test_size=0.2, validation_size=0.4)
    # fit the models
    best_model_tuple = auto_fit(models, problem, X_train, y_train, X_val, y_val)
    # test the best model
    test_metrics = test_fit(best_model_tuple, problem, X_test, y_test)

    return test_metrics, best_model_tuple

# import example training data
data = {
    "feature1": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 10, 5000) + np.random.uniform(-8, 7, 5000)),
    "feature2": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 5, 5000) + np.random.uniform(-8, 8, 5000)),
    "feature3": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 1, 5000) + np.random.uniform(-1, 1, 5000)),
    "feature4": np.where(np.random.rand(5000) < 0.05, np.nan, np.random.uniform(0, 100, 5000)),
    "feature5": np.where(np.random.rand(5000) < 0.05, np.nan, np.random.normal(50, 10, 5000)),
    "feature6": np.random.choice(['A', 'B', 'C'], size=5000, p=[0.4, 0.4, 0.2]),
    "target": np.random.choice(['Yes', 'No'], size=5000, p=[0.6, 0.4]),
    "feature8": np.random.randint(0, 100, size=5000),
    "feature7": np.random.choice(['Low', 'Medium', 'High'], size=5000, p=[0.3, 0.5, 0.2]),
    "feature10": np.random.uniform(-5, 5, 5000),
    "feature9": np.where(np.random.rand(5000) < 0.05, None,
                       np.exp(0.5 * np.linspace(0, 10, 5000)) +
                       0.1 * np.random.choice([-1, 1], size=5000) *
                       np.random.uniform(0, 1, 5000))
}

#run main function
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    target_str = 'target'
    df = pd.DataFrame(data)[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', target_str]]
    test_metrics, best_model_tuple = main(df, target_str, imputation_dict=None)
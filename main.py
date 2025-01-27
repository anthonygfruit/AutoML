#dependencies
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
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

# identify categorical/date columns
def get_cat_cols(df):
    return [col for col in df.columns if df[col].dtype == "string" or df[col].dtype == "date" or df[col].dtype == "object"]

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
    return df

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
def split_data(X, y, test_size=0.2):
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test

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

# get descriptive statistics including NaN and None values
def get_descriptives(df):
    descriptives = df.describe(include='all').transpose()
    descriptives['NaN Count'] = df.isna().sum()
    descriptives['None Count'] = df.apply(lambda x: (x == None).sum() + (x == "None").sum())
    descriptives['Unique Count'] = df.nunique()
    return descriptives

# hyperparameter tuning and cross-validation with custom parameter grids
def tune_params(models, problem, X_train, y_train, verbose=0, custom_param_grids=None):
    # Default parameter grids
    param_grids = {
        "Linear Regression": {
            "fit_intercept": [True, False],
            #"normalize": [True, False]
        },
        "Random Forest": {
                     "n_estimators": [10, 50, 100],
        # "max_depth": [None, 5, 10],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [3, 5, 10],
        # "max_features": ["sqrt", "log2", None],
        # "bootstrap": [True, False]
        },
        "Gradient Boosting": {
            "n_estimators": [10, 50, 100, 200],
            # "learning_rate": [0.01, 0.1, 0.5],
            # "max_depth": [3, 5, 10],
            # "min_samples_split": [2, 5, 10],
            # "min_samples_leaf": [3, 5, 10],
        },
        "Neural Network": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            # "activation": ["relu", "tanh", "logistic"],
            # "solver": ["adam", "sgd", "lbfgs"],
            # "alpha": [0.0001, 0.001, 0.01],
            # "learning_rate": ["constant", "invscaling", "adaptive"]
        },
        "Logistic Regression": {
            "penalty": ["l1", "l2", "elasticnet", "none"],
            #"C": [0.1, 1, 10, 100],
            #"solver": ["liblinear", "saga", "lbfgs"],
            #"max_iter": [100, 200, 500]
        },
        "ARIMA": {
            "order": [(1, 0, 0), (0, 1, 1), (1, 1, 1), (2, 1, 2)],
            #"seasonal_order": [(0, 0, 0, 0), (1, 1, 1, 12)],
            #"trend": ["n", "c", "t", "ct"]
        },
        "Prophet": {}
    }

    # Merge custom grids with default grids (if provided)
    if custom_param_grids:
        param_grids.update(custom_param_grids)

    tuned_models = {}

    for name, model in models.items():
        try:
            print(f"Training {name} model...")
            param_grid = param_grids.get(name, {})
            if param_grid:
                grid_search = GridSearchCV(
                    model,
                    param_grid,
                    verbose=verbose,
                    cv=4,
                    scoring="r2" if problem in ["regression", "time-series"] else "accuracy"
                )
                grid_search.fit(X_train, y_train)
                tuned_models[name] = grid_search.best_estimator_
                if verbose > 0:
                    print(f"Best {grid_search.scoring} score for {name}: {grid_search.best_score_}")
            else:
                tuned_models[name] = model
        except Exception as e:
            print(f"Error tuning model {name}: {e}")
            tuned_models[name] = model

    return tuned_models

# evaluate with test best models
def evaluate(tuned_models, problem, X_test, y_test):
    best_model_name = None
    best_model = None
    best_test_score = float("inf") if problem in ["regression", "time-series"] else float("-inf")
    best_test_metrics = {}

    for name, model in tuned_models.items():
        try:
            # Make predictions on the test set
            y_test_pred = model.predict(X_test)

            # Determine evaluation metrics based on problem type
            if problem == "binary-classification":
                test_metrics = {
                    "accuracy": accuracy_score(y_test, y_test_pred),
                    "f1": f1_score(y_test, y_test_pred),
                    "precision": precision_score(y_test, y_test_pred),
                    "recall": recall_score(y_test, y_test_pred)
                }
                score = test_metrics["accuracy"]  # Use accuracy as a comparison metric
            elif problem == "multi-class-classification":
                test_metrics = {
                    "accuracy": accuracy_score(y_test, y_test_pred),
                    "f1_macro": f1_score(y_test, y_test_pred, average='macro'),
                    "f1_weighted": f1_score(y_test, y_test_pred, average='weighted')
                }
                score = test_metrics["accuracy"]  # Use accuracy as a comparison metric
            elif problem in ["regression", "time-series"]:
                test_metrics = {
                    "mean_squared_error": mean_squared_error(y_test, y_test_pred),
                    "mean_absolute_error": mean_absolute_error(y_test, y_test_pred),
                    "r2_score": r2_score(y_test, y_test_pred)
                }
                score = test_metrics["mean_squared_error"]  # Use MSE as a comparison metric
            else:
                test_metrics = {}
                score = None

            # Determine the best model based on the test metric
            if (problem in ["binary-classification", "multi-class-classification"] and score > best_test_score) or \
                    (problem in ["regression", "time-series"] and score < best_test_score):
                best_model_name = name
                best_model = model
                best_test_score = score
                best_test_metrics = test_metrics
        except Exception as e:
            print(f"Error evaluating model {name}: {e}")

    # Print only the best model's results
    if best_model_name:
        print(f"Best Model: {best_model_name}", f"\nScore: {best_test_metrics}")

    return (best_model_name, best_model)

# run automl
def main(df, target_str, descriptives=True, verbose=0, encode_method='label', custom_param_grids=None, imputation_dict=None, impute_method="mean", test_size=0.2, scale=True):
    # guess data types
    dt = guess_data_types(df)
    df = update_data_types(df, dt)
    # impute missing values
    if imputation_dict is not None:
        df = impute_missing_values_categorical_bulk(df, imputation_dict)
    df = impute_missing_values(df, strategy=impute_method)
    # show descriptives summary
    if verbose > 0:
        print(get_descriptives(df).to_string(), '\n')
    # select appropriate machine learning problem
    problem = select_problem(df, target_str)
    # get machine learning models based on the problem type
    models = get_models(problem)
    # visualize correlations
    corplot(df, target_str)
    # find categorical variables
    cat_cols = get_cat_cols(df)
    # encode categorical variables
    if encode_method=='label':
        df, encoders = label_encode(df, cat_cols)
    elif encode_method=='onehot':
        cat_cols.remove(target_str)
        df, encoders = one_hot_encode(df, cat_cols)
        df, target_encoder = label_encode(df, [target_str])
        encoders[target_str] = target_encoder[target_str]
    # scale the data
    y = df[target_str]
    if scale:
        X, scaler = scale_data(df.drop(target_str, axis=1))
    else:
        X = df.drop(target_str, axis=1)
    # split the data into train, validation, and test sets
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)
    # fit the models with cv
    tuned_models = tune_params(models, problem, X_train, y_train, verbose=verbose, custom_param_grids=custom_param_grids)
    # test the best model
    best_model_tuple = evaluate(tuned_models, problem, X_test, y_test)

    return best_model_tuple, tuned_models, encoders, scaler

# un-encode df
def un_encode(df, encoders):
    for col, encoder in encoders.items():
        if isinstance(encoder, LabelEncoder):
            try:
                # Inverse transform using LabelEncoder
                df[col] = encoder.inverse_transform(df[col])
            except ValueError:
                # Ignore unseen labels
                df[col] = df[col].apply(lambda x: None if x not in encoder.classes_ else x)
        elif isinstance(encoder, OneHotEncoder):
            # Inverse transform using OneHotEncoder
            ohe_columns = [c for c in df.columns if c.startswith(col + "_")]
            ohe_array = df[ohe_columns].values
            try:
                # Find the original category
                original_categories = encoder.inverse_transform(ohe_array)
                df[col] = original_categories
            except ValueError:
                # Ignore unseen labels
                df[col] = None
            # Drop the one-hot encoded columns
            df = df.drop(columns=ohe_columns, axis=1)
    return df

# make predictions on new data
def predict_new(df, target_str, best_model_tuple, encoders, scaler, impute_method, imputation_dict=None):
    # guess data types
    dt = guess_data_types(df)
    df = update_data_types(df, dt)
    # impute missing values
    if imputation_dict is not None:
        df = impute_missing_values_categorical_bulk(df, imputation_dict)
    df = impute_missing_values(df, strategy=impute_method)

    # Encode the data using encoders
    for col, encoder in encoders.items():
        if col in df.columns and isinstance(encoder, LabelEncoder):
            # Transform using LabelEncoder
            df[col] = encoder.transform(df[col])
        elif col in df.columns and isinstance(encoder, OneHotEncoder):
            # Transform using OneHotEncoder
            encoded_df = pd.DataFrame(
                encoder.transform(df[[col]]).toarray(),
                columns=[f"{col}_{cat}" for cat in encoder.categories_[0]],
                index=df.index
            )
            df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)

        # Scale the data using the provided scaler
    X_new = scaler.transform(df)

    # Make predictions using the best model
    model = best_model_tuple[1]
    predictions = model.predict(X_new)
    df[target_str] = predictions
    
    # Attempt to un-encode the predictions DataFrame
    try:
        df = un_encode(df, encoders)
    except Exception as e:
        print(f"Error during un-encoding: {e}")
    return df

#SAMPLE USAGE

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
    "feature9": np.random.choice(['Yes', 'No'], size=5000, p=[0.6, 0.4]),
    "feature8": np.random.randint(0, 100, size=5000),
    "feature7": np.random.choice(['Low', 'Medium', 'High'], size=5000, p=[0.3, 0.5, 0.2]),
    "feature10": np.random.uniform(-5, 5, 5000),
    "target": np.where(np.random.rand(5000) < 0.05, None,
                       np.exp(0.5 * np.linspace(0, 10, 5000)) +
                       0.1 * np.random.choice([-1, 1], size=5000) *
                       np.random.uniform(0, 1, 5000))
}

# import new data for predictions
new_data = {
    "feature1": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 10, 5000) + np.random.uniform(-8, 7, 5000)),
    "feature2": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 5, 5000) + np.random.uniform(-8, 8, 5000)),
    "feature3": np.where(np.random.rand(5000) < 0.05, np.nan,
                         np.linspace(0, 1, 5000) + np.random.uniform(-1, 1, 5000)),
    "feature4": np.where(np.random.rand(5000) < 0.05, np.nan, np.random.uniform(0, 100, 5000)),
    "feature5": np.where(np.random.rand(5000) < 0.05, np.nan, np.random.normal(50, 10, 5000)),
    "feature6": np.random.choice(['A', 'B', 'C'], size=5000, p=[0.4, 0.4, 0.2]),
    "feature9": np.random.choice(['Yes', 'No'], size=5000, p=[0.6, 0.4]),
    "feature8": np.random.randint(0, 100, size=5000),
    "feature7": np.random.choice(['Low', 'Medium', 'High'], size=5000, p=[0.3, 0.5, 0.2]),
    "feature10": np.random.uniform(-5, 5, 5000)
}

# run main function
if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    #sample data
    target_str = 'target'
    df = pd.DataFrame(data)[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7', target_str]]
    new_df = pd.DataFrame(new_data)[['feature1', 'feature2', 'feature3', 'feature4', 'feature5', 'feature6', 'feature7']]

    #automl
    best_model_tuple, fitted_models, encoders, scaler = main(df, target_str, descriptives=True, verbose=2,
                                                         encode_method='label', custom_param_grids=None,
                                                         imputation_dict={'feature1' : 10, 'feature7': 'high'},
                                                         impute_method="mode", test_size=0.2, scale=True)
    #make predictions
    new_df_pred = predict_new(new_df, target_str, best_model_tuple, encoders, scaler, impute_method='mode', imputation_dict={'feature1' : 10, 'feature7': 'high'})
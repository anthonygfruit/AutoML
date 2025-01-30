# dependencies
import itertools
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dateutil.parser import parse
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, \
    GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score, \
    precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX


# DATA PROCESSING

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
    return [col for col in df.columns if
            df[col].dtype == "string" or np.issubdtype(df[col].dtype, np.datetime64) or df[col].dtype == "object"]

# fuzzy date match
def is_date(df, fuzzy=False):
    date_cols = []
    for col in df.columns:
        # Check if any value in the column is a valid date
        def is_parsable_date(val):
            try:
                # Try to parse the value as a date
                if isinstance(val, str):
                    parse(val, fuzzy=fuzzy)
                    return True
                elif pd.notna(val) and isinstance(val, datetime):
                    return True
                return False
            except:
                return False

        if df[col].apply(is_parsable_date).any():
            date_cols.append(col)

    return date_cols


# dates to ordinal
def dates_to_ordinal(df):
    converted_cols = []
    date_cols = is_date(df)
    for col in date_cols:
        try:
            df[col] = pd.to_datetime(df[col], errors='coerce')
            if df[col].notna().any():
                df[col] = df[col].map(lambda x: x.toordinal() if pd.notna(x) else 0).astype(int)
                converted_cols.append(col)
        except:
            pass
    return df, converted_cols


# ordinal to dates
def ordinal_to_dates(df, converted_cols):
    for col in converted_cols:
        try:
            df[col] = pd.to_datetime(df[col].astype(int).map(lambda x: datetime.fromordinal(x) if x > 0 else None), errors='coerce')
        except:
            pass
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
            if ohe_columns:
                try:
                    # Use the encoder to reverse one-hot encoding
                    reverse_transformed = encoder.inverse_transform(df[ohe_columns].values)
                    df[col] = [category[0] for category in reverse_transformed]
                except Exception as e:
                    df[col] = None
                # Drop the one-hot encoded columns
                df = df.drop(columns=ohe_columns, axis=1)
    return df


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
        df[col] = df[col].replace([pd.NaT, None, "None", np.nan, "", float('inf'), -float('inf')], mode_value).fillna(
            mode_value)
    return df


# impute missing values for multiple columns based on a dictionary of col/value pairs
def impute_missing_values_categorical_bulk(df, imputation_dict):
    for col, value in imputation_dict.items():
        df = impute_missing_values_categorical(df, col, value)
    return df


# prepare time-series data
def prepare_ts_df(df, ts_col, target_str):
    # check if the ts_col is provided and is a datetime column
    if ts_col and df[ts_col].dtype == "datetime64[ns]":
        # set the ts_col as the index
        df = df.set_index(ts_col).sort_index()
    return df[[target_str]]


# split into train, validation, and test sets
def split_data(X, y, test_size=0.2):
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    return X_train, X_test, y_train, y_test


# VISUALIZATION
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


# AUTOML
# select appropriate machine learning problem (regression, binary classification, multi-class classification)
def select_problem(df, y_col):
    # check if the target column is a string
    if df[y_col].dtype == "string" or df[y_col].dtype == "object":
        # check if the target column has only two unique values
        if len(df[y_col].unique()) == 2:
            return "binary-classification"
        else:
            return "multi-class-classification"
    # if none of the above, assume it's a regression problem
    else:
        return "regression"


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
        models["ARIMA"] = None
        models["Prophet"] = Prophet()
    return models


# TRAINING
# tune arima
def tune_arima(df, target_str, season_periods=12, test_periods=365, verbose=0):
    # Define the p, d, and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # Generate all different combinations of p, d, and q triplets
    pdq = list(itertools.product(p, d, q))

    # Generate all different combinations of seasonal p, d, and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], season_periods) for x in list(itertools.product(p, d, q))]

    warnings.filterwarnings("ignore")  # Ignore warnings

    best_mse = float("inf")
    best_params = None

    # Split the data into training and testing sets
    train_data = df[:-test_periods]
    test_data = df[-test_periods:]

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                start_time = datetime.now()
                mod = SARIMAX(train_data[target_str],
                              order=param,
                              seasonal_order=param_seasonal,
                              enforce_stationarity=False,
                              enforce_invertibility=False)

                results = mod.fit()

                # Make predictions on the test data
                predictions = results.get_forecast(steps=test_periods).predicted_mean
                mse = mean_squared_error(test_data[target_str], predictions)
                end_time = datetime.now()

                if verbose > 1:
                    print(
                        f"Params: {param}, Seasonal Params: {param_seasonal}, Time taken: {end_time - start_time}, MSE: {mse}")

                if mse < best_mse:
                    best_mse = mse
                    best_params = (param, param_seasonal)

            except Exception as e:
                continue

    return ("ARIMA", best_params)


#tune prophet
def tune_prophet(df, target_str, test_periods=365, verbose=0):
    # Define a list of parameter grids to search over
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'yearly_seasonality': [True, False],
        'weekly_seasonality': [True, False],
        'daily_seasonality': [True, False]
    }

    # Initialize the best parameters
    best_params = None
    best_score = float("inf")

    # Split the data into training and testing sets
    train_data = df[:-test_periods]
    test_data = df[-test_periods:]

    # Iterate over all combinations of parameters
    for params in itertools.product(*param_grid.values()):
        param_dict = dict(zip(param_grid.keys(), params))
        model = Prophet(**param_dict)

        # Fit the model
        start_time = datetime.now()
        model.fit(train_data[[target_str]].reset_index().rename(columns={"index": "ds", target_str: "y"}))

        # Make predictions on the test set
        future = model.make_future_dataframe(periods=test_periods)
        forecast = model.predict(future)
        forecast = forecast.set_index('ds').join(test_data[target_str], how='inner')

        # Calculate the mean squared error on the test set
        mse = mean_squared_error(forecast['yhat'], forecast[target_str])
        end_time = datetime.now()

        # Print details if verbose is greater than 1
        if verbose > 1:
            print(f"Params: {param_dict}, Time taken: {end_time - start_time}, MSE: {mse}")

        # Update best_params and best_model if the current model is better
        if mse < best_score:
            best_score = mse
            best_params = param_dict

    return ("Prophet", best_params)


# tune regression and classification
def tune_params(models, problem, X_train, y_train, verbose=0, custom_param_grids=None, cv_folds=4):
    # Default parameter grids
    param_grids = {
        "Linear Regression": {
            "fit_intercept": [True, False],
            # "normalize": [True, False]
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
            "penalty": ["l2"],
            # "C": [0.1, 1, 10, 100],
            # "solver": ["liblinear", "saga", "lbfgs"],
            # "max_iter": [100, 200, 500]
        }
    }

    # Merge custom grids with default grids (if provided)
    if custom_param_grids:
        for model_name, custom_grid in custom_param_grids.items():
            param_grids[model_name] = custom_grid

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
                    cv=cv_folds,
                    scoring="r2" if problem == "regression" else "accuracy"
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


# TESTING
# evaluate regression and classification models on test set
def evaluate(tuned_models, problem, X_test, y_test):
    best_model_name = None
    best_model = None
    best_test_score = float("inf") if problem in ["regression"] else float("-inf")
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
            elif problem == "regression":
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
                    (problem in ["regression"] and score < best_test_score):
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


# PREDICTION
# make predictions on new data
def predict_unseen(df, target_str, model, encoders, scaler, impute_method, imputation_dict=None):
    # Guess data types
    dt = guess_data_types(df)
    df = update_data_types(df, dt)

    # Impute missing values
    if imputation_dict is not None:
        df = impute_missing_values_categorical_bulk(df, imputation_dict)
    df = impute_missing_values(df, strategy=impute_method)

    # Convert dates to ordinals
    df, _ = dates_to_ordinal(df)

    # Encode the data using encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            if isinstance(encoder, LabelEncoder):
                df[col] = df[col].apply(lambda x: x if x in encoder.classes_ else 'unknown')
                if 'unknown' not in encoder.classes_:
                    # Convert to object dtype to handle mixed types
                    encoder.classes_ = np.append(encoder.classes_.astype(object), 'unknown')
                df[col] = encoder.transform(df[col])
            elif isinstance(encoder, OneHotEncoder):
                df[col] = df[col].fillna("unknown").astype(str)
                known_categories = set(encoder.categories_[0])
                df[col] = df[col].apply(lambda x: x if x in known_categories else "unknown")
                ohe_encoded = encoder.transform(df[[col]]).toarray()
                ohe_columns = [f"{col}_{i}" for i in range(len(encoder.categories_[0]))]
                ohe_df = pd.DataFrame(ohe_encoded, columns=ohe_columns, index=df.index)
                df = pd.concat([df.drop(columns=[col]), ohe_df], axis=1)

    # Scale the data using the provided scaler
    X_new = scaler.transform(df)

    # Make predictions using the best model
    predictions = model.predict(X_new)
    df[target_str] = predictions

    # Attempt to un-encode the predictions DataFrame
    try:
        df = un_encode(df, encoders)
    except Exception as e:
        print(f"Error during un-encoding: {e}")

    return df


# make predictions on new time series data
def predict_ts(df, best_params, f_periods=365):
    model_name = best_params[0]
    params = best_params[1]

    if model_name == "ARIMA":
        # Fit the SARIMA model using the best parameters
        model = SARIMAX(df, order=params[0], seasonal_order=params[1])
        model = model.fit()
        # Make predictions using the best model
        forecast = model.get_forecast(steps=f_periods)
        return forecast.predicted_mean
    elif model_name == "Prophet":
        # Fit the Prophet model using the best parameters
        model = Prophet(**params)
        model.fit(df.reset_index().rename(columns={"index": "ds", df.columns[0]: "y"}))
        # Make predictions using the best model
        future = model.make_future_dataframe(periods=f_periods)
        forecast = model.predict(future)
        return forecast.set_index('ds').tail(f_periods)["yhat"]


# PIPELINE
def main(df, target_str, new_df=None, ts_col=None, ts_periods=365, impute_method="mean", imputation_dict=None,
         encode_method='label', scale_method='scale', test_size=0.2, custom_param_grids=None, cv_folds=4, verbose=0):
    # Get descriptive statistics
    print(get_descriptives(df).to_string())

    # Time-series case
    if ts_col:
        df = prepare_ts_df(df, ts_col, target_str)

        best_sarima = tune_arima(df, target_str, season_periods=12, test_periods=int(ts_periods), verbose=verbose)
        best_prophet = tune_prophet(df, target_str, test_periods=int(ts_periods), verbose=verbose)

        # Make time-series predictions
        sarima_predictions = predict_ts(df, best_sarima, n_periods=int(ts_periods))
        prophet_predictions = predict_ts(df, best_prophet, n_periods=int(ts_periods))

        return (sarima_predictions, prophet_predictions)

    # Non-time-series case
    else:

        # guess data types
        dt = guess_data_types(df)
        df = update_data_types(df, dt)

        # impute missing values
        if imputation_dict is not None:
            df = impute_missing_values_categorical_bulk(df, imputation_dict)
        df = impute_missing_values(df, strategy=impute_method)

        # select appropriate machine learning problem
        problem = select_problem(df, target_str)
        print(f"\n{problem.title()}\n")

        # Plot correlations
        corplot(df, target_str)

        df, converted_cols = dates_to_ordinal(df)

        # Select appropriate machine learning problem
        problem = select_problem(df, target_str)

        # Get machine learning models based on the problem type
        models = get_models(problem)

        # find categorical variables
        cat_cols = get_cat_cols(df)

        encoders = None
        # encode categorical variables
        if encode_method.lower() == 'label':
            df, encoders = label_encode(df, cat_cols)
        elif encode_method.lower() == 'onehot':
            if target_str in cat_cols:
                cat_cols.remove(target_str)
            df, encoders = one_hot_encode(df, cat_cols)
            df, target_encoder = label_encode(df, [target_str])
            encoders[target_str] = target_encoder[target_str]

        scaler = None
        # scale the data
        y = df[target_str]
        X = df.drop(target_str, axis=1)
        if scale_method.lower() == 'scale':
            X, scaler = scale_data(X)
        elif scale_method.lower() == 'normalize':
            X, scaler = normalize_data(X)

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size)

        # Tune the models
        tuned_models = tune_params(models, problem, X_train, y_train, verbose=verbose,
                                   custom_param_grids=custom_param_grids, cv_folds=cv_folds)

        # Evaluate the models on the test set
        best_model_name, best_model = evaluate(tuned_models, problem, X_test, y_test)

        # Make predictions on new_df
        if new_df is not None:
            preds = predict_unseen(new_df, target_str, best_model, encoders, scaler, impute_method, imputation_dict)

            # Convert ordinals back to dates
            preds = ordinal_to_dates(preds, converted_cols)

            return (preds, best_model_name, best_model)
        else:
            return (best_model_name, best_model, encoders, scaler, converted_cols, cat_cols)

# examples of how to use the thang
def example_dict():
    return {"Non Time Series Usage": '''if __name__ == "__main__":

        df = pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=1000, freq="D"),
            "target_column": np.random.rand(1000),
            "categorical_column": np.random.choice(["A", "B", "C", None], 1000),
            "numerical_column": np.random.choice([1, 2, 3, None], 1000),
            "string_column": np.random.choice(["X", "Y", "Z", None], 1000)
        })

        new_df = pd.DataFrame({
            "date": pd.date_range(start="2020-01-01", periods=100, freq="D"),
            "categorical_column": np.random.choice(["A", "B", "C", None], 100),
            "numerical_column": np.random.choice([1, 2, 3, None], 100),
            "string_column": np.random.choice(["X", "Y", "Z", None], 100)
        })


        # Run main function
        info = main(df,
                     target_str="target_column",
                     new_df=new_df,
                     ts_col=None,
                     ts_periods=365,
                     impute_method="mean",
                     imputation_dict=None,
                     encode_method='label',
                     scale_method='scale',
                     test_size=0.2,
                     custom_param_grids=None,
                     cv_folds=4,
                     verbose=0)

        # Display predictions
        print(info[0].head().to_string())''', "Time Series Usage": '''# Example usage for time series
            if __name__ == "__main__":
                # Create sample training data for time series with nan and none values
                date_rng = pd.date_range(start='2022-01-01', end='2024-12-31', freq='D')
                df = pd.DataFrame({
                    "date": date_rng,
                    "target_column": np.random.rand(len(date_rng)),
                    "categorical_column": np.random.choice(["A", "B", "C", None], len(date_rng)),
                    "numerical_column": np.random.choice([1, 2, 3, None], len(date_rng))
                })
        
                # Create sample new data for predictions
                new_date_rng = pd.date_range(start='2025-01-01', end='2025-12-31', freq='D')
                new_df = pd.DataFrame({
                    "date": new_date_rng,
                    "target_column": np.random.rand(len(new_date_rng)),
                    "numerical_column": np.random.choice([1, 2, 3, None], len(new_date_rng))
                })
        
                # Run main function
                info = main(df,
                             target_str="target_column",
                             new_df=new_df
                             ts_col="date",
                             ts_periods=365,
                             impute_method="mean",
                             imputation_dict=None,
                             encode_method='label',
                             scale_method='scale',
                             test_size=0.2,
                             custom_param_grids=None,
                             cv_folds=4,
                             verbose=2)
        
                # Display predictions
                print(info[0].head().to_string())''', "Custom Param Grid": ''' param_grids = {
        "Linear Regression": {
            "fit_intercept": [True, False],
            "normalize": [True, False]
        },
        "Random Forest": {
            "n_estimators": [10, 50, 100],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [3, 5, 10],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False]
        },
        "Gradient Boosting": {
            "n_estimators": [10, 50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.5],
            "max_depth": [3, 5, 10],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [3, 5, 10],
        },
        "Neural Network": {
            "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50)],
            "activation": ["relu", "tanh", "logistic"],
            "solver": ["adam", "sgd", "lbfgs"],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate": ["constant", "invscaling", "adaptive"]
        },
        "Logistic Regression": {
            "penalty": ["l2"],
            "C": [0.1, 1, 10, 100],
            "solver": ["liblinear", "saga", "lbfgs"],
            "max_iter": [100, 200, 500]
        }
    }'''}
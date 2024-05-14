"""
This is a toolkit for GARCH model estimation and forecasting.
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# calculate returns
import numpy as np

def wrangle_data(df):
    """
    Wrangles the given DataFrame by sorting it by date, creating a 'log return' column,
    and returning the 'log return' values.

    Parameters:
    df (DataFrame): The input DataFrame containing the 'close' column.

    Returns:
    Series: The 'log return' values with NaN values dropped.
    """
    # sort DataFrame by date
    df.sort_index(ascending=True, inplace=True)
    # create "log return" column
    df['log return'] = np.log(df['close']).diff() * 100
    # return return
    return df["log return"].dropna()

def clean_prediction(prediction):
    """
    Cleans the given prediction by performing the following steps:
    1. Calculates the forecast start date by adding one day to the first index of the prediction.
    2. Creates a date range using business days starting from the forecast start date.
    3. Converts the date range into ISO 8601 format and assigns it as the index labels for the prediction.
    4. Extracts the predictions from the DataFrame and takes the square root of each value.
    5. Combines the square root values with the prediction index labels into a Series.
    6. Converts the Series into a dictionary and returns it.

    Parameters:
    - prediction (DataFrame): The prediction data to be cleaned.

    Returns:
    - prediction_formatted (dict): The cleaned prediction data in the form of a dictionary.
    """
    # Calculate forecast start date
    start = prediction.index[0] + pd.DateOffset(days=1)

    # Create date range
    prediction_dates = pd.bdate_range(start=start, periods=prediction.shape[1])

    # Create prediction index labels, ISO 8601 format
    prediction_index = [d.isoformat() for d in prediction_dates]

    # Extract predictions from DataFrame, get square root
    data = prediction.values.flatten() ** 0.5

    # Combine `data` and `prediction_index` into Series
    prediction_formatted = pd.Series(data, index=prediction_index).to_dict()

    # Return Series as dictionary
    return prediction_formatted

def time_series_plot(df_train, name="name", model_con_vol=None):
    """
    Generate a time series plot with the train data and conditional volatility.

    Parameters:
    df_train (DataFrame): The train data for plotting.
    name (str): The name of the plot. Default is "name".
    model_con_vol (Series): The conditional volatility for plotting. Default is None.

    Returns:
    Time series plot with train data and conditional volatility.
    """
    # create a time series plot with the train data
    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot train data
    df_train.plot(ax=ax, label=f"{name} Daily Returns")

    # Plot conditional volatility * 2
    (2 * model_con_vol).plot(
        ax=ax, color="C1", label="2 SD Conditional Volatility", linewidth=3
    )

    # Plot conditional volatility * -2
    (-2 * model_con_vol.rename()).plot(
        ax=ax, linewidth=3
    )

    # Add axis labels
    plt.xlabel("Date")

    # Add legend
    plt.legend();

def walk_forward_valid(df, p=1, q=1, name="df"):
    """
    Perform walk-forward validation for GARCH model.

    Parameters:
    - df (pandas.DataFrame): The input dataframe containing the time series data.
    - p (int): The order of the autoregressive component of the GARCH model. Default is 1.
    - q (int): The order of the moving average component of the GARCH model. Default is 1.
    - name (str): The name of the dataframe. Default is "df".

    Returns:
    - y_test_wfv (pandas.Series): The predicted volatility values for the test data.

    This function performs walk-forward validation for a GARCH model. It splits the input dataframe into training and test data,
    trains a GARCH model on the training data, and generates volatility predictions for the test data. The function returns a
    pandas Series containing the predicted volatility values for the test data.

    Example usage:
    >>> df = pd.read_csv('data.csv')
    >>> y_test_wfv = walk_forward_valid(df, p=2, q=2, name="AAPL")
    """
    # Create empty list to hold predictions
    predictions = []

    # Calculate size of test data (20%)
    test_size = int(len(df) * 0.2)

    # Walk forward
    for i in range(test_size):
        # Create test data
        df_test = df.iloc[: -(test_size - i)]

        # Train model
        # disp=0, avoid model training spit on screen
        model = arch_model(df_test, p=p, q=q, rescale=False).fit(disp=0)

        # Generate next prediction (volatility, not variance)
        next_pred = model.forecast(horizon=1, reindex=False).variance.iloc[0,0]**0.5

        # Append prediction to list
        predictions.append(next_pred)

    # Create Series from predictions list
    y_test_wfv = pd.Series(predictions, index=df.tail(test_size).index)

    print("y_test_wfv type:", type(y_test_wfv))
    print("y_test_wfv shape:", y_test_wfv.shape)
    y_test_wfv.head()

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot returns for test data
    df.tail(test_size).plot(ax=ax, label=f"{name} Return")

    # Plot volatility predictions * 2
    (2 * y_test_wfv).plot(ax=ax, c="C1", label="2 SD Predicted Volatility")

    # Plot volatility predictions * -2
    (-2 * y_test_wfv).plot(ax=ax, c="C1")

    # Label axes
    plt.xlabel("Date")
    plt.ylabel("Return")

    # Add legend
    plt.legend();
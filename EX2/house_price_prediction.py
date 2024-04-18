import math

import IMLearn.utils
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"
from IMLearn.utils import split_train_test


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    # load data and drop unnecessary data
    raw_data = pd.read_csv(filename).drop(["id", "date", "lat", "long", "sqft_lot15", "sqft_living15"], axis=1)

    # adds a McCollum of mean price by zip
    add_mean_price_by_zip(raw_data)

    #replace zip code col to dummies
    return pd.get_dummies(raw_data, prefix='zipcode_', columns=['zipcode'], dtype=int)


def add_mean_price_by_zip(raw_data):
    """
    this function adds to the raw data the mean price of all transaction in a certian zipcode, in order to evaluate the
    location factor of the property better
    """
    mean_price_by_zipcode = raw_data.groupby("zipcode")["price"].mean()
    mean_price_by_zipcode = mean_price_by_zipcode.sort_values(ascending=False)
    raw_data['zipcode_mean_price'] = raw_data['zipcode'].replace(mean_price_by_zipcode)


def preprocess(raw_database):
    """
    this function drop unprobable values in the data.
    to be used on the train data only!!!
    """
    pured_database = raw_database.drop_duplicates().dropna()
    for title in ["price", "sqft_living", "floors", "sqft_above", "yr_built"]:
        pured_database = pured_database[pured_database[title] > 0]
    for title in ["sqft_lot", "sqft_basement", "yr_renovated", "bedrooms", "bathrooms"]:
        pured_database = pured_database[pured_database[title] >= 0]
    pured_database["decade_built"] = (pured_database["yr_built"] / 10).astype(int)
    pured_database = pured_database.drop("yr_built", axis=1)
    pured_database = pured_database[pured_database["waterfront"].isin([0, 1]) &
                                    pured_database["view"].isin(range(5)) &
                                    pured_database["condition"].isin(range(1, 6)) &
                                    pured_database["grade"].isin(range(1, 14))]
    return pured_database


def feature_evaluation(X: pd.DataFrame, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    y, X = X['price'], X.drop('price', axis=1)
    for feature in ["bedrooms", "sqft_living", "sqft_lot", "floors", "waterfront", "view",
                    "condition", "grade", "sqft_basement", "sqft_above", "yr_renovated", 'zipcode_mean_price'
                    ]:
        pearsons_correlation = round(np.cov(X[feature], y)[0, 1] / (math.sqrt((np.var(X[feature]) * np.var(y)))), 2)
        pearsons_correlation_dataframe = pd.DataFrame({f'{feature}': X[feature], "price": y})
        plot = px.scatter(pearsons_correlation_dataframe, x=f"{feature}", y="price",
                          color_discrete_sequence=["black"], title=f"correlation for {feature} and price"
                                                                   f"(pearsons correlation = {pearsons_correlation})",
                          labels={f'{feature}': f"{feature}", "price": "price"})
        pio.write_image(plot, output_path + f"/pearsons.{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    raw_data = load_data("../datasets/house_prices.csv")
    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(preprocess(raw_data))

    # Question 3 - Split samples into training- and testing sets.
    train = raw_data.sample(frac=0.75)
    test = raw_data.loc[raw_data.index.difference(train.index)]
    train = preprocess(train)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    # Define a list of training percentages
    training_percentages = list(range(10, 101))

    # Initialize an array to store the results
    num_trials = 10
    results = np.zeros((len(training_percentages), num_trials))

    clean_test = test.fillna(test.median())

    # Loop over the training percentages and number of trials
    for i, percentage in enumerate(training_percentages):
        for j in range(num_trials):
            # Sample a fraction of the training data

            X_train = train.sample(frac=percentage / 100.0)
            y_train, X_train = X_train['price'], X_train.drop('price', axis=1)

            # Fit a linear regression model and compute the test MSE
            model = LinearRegression(include_intercept=True).fit(X_train, y_train)
            test_MSE = model.loss(clean_test.drop('price', axis=1), test['price'])

            # Store the result
            results[i, j] = test_MSE

    # Compute the mean and standard deviation of the results
    means = np.mean(results, axis=1)
    std = np.std(results, axis=1)

    # Create a plot of test MSE as a function of training percentage
    fig = go.Figure(
        [go.Scatter(x=training_percentages, y=means - 2 * std, fill=None, mode="lines", line=dict(color="blue")),
         go.Scatter(x=training_percentages, y=means + 2 * std, fill='tonexty', mode="lines", line=dict(color="blue")),
         go.Scatter(x=training_percentages, y=means, mode="markers+lines", marker=dict(color="black"))],
        layout=go.Layout(title="Correlation between percentage of training samples and loss value",
                         xaxis=dict(title="percentage of samples used as training"),
                         yaxis=dict(title="Loss Over Test Set (MSE)"),
                         showlegend=False))
    vline_annotation = go.layout.Annotation(  # y-coordinate of the annotation
        text='Minimum percentage sample (10)',  # text of the annotation
        font=dict(size=16)  # the font size of the annotation text
    )
    fig.add_vline(x=10, line_width=3, line_dash="dash", line_color="red", annotation=vline_annotation)

    fig.write_image("./plots_images/mse.over.training.percentage.png")

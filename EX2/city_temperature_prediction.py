from typing import Tuple

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import random
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    raw_data = pd.read_csv(filename)

    # create a new column called "Date" containing datetime values based on "day", "month", and "year"
    raw_data['Date'] = pd.to_datetime(raw_data[['Year', 'Month', 'Day']], errors='coerce')

    # fill in missing values in the "Date" column with the corresponding values from the "day", "month", and "year" columns
    raw_data['Date'] = raw_data['Date'].fillna(pd.to_datetime(raw_data['Year'] * 10000 + raw_data['Month'] * 100 + raw_data['Day'], format='%Y%m%d'))

    # create a column of the day of the year (from 1 to 365) and drop the Date column
    raw_data['Day_of_the_year'] = raw_data['Date'].dt.dayofyear
    return raw_data.drop(['Date'], axis=1)


def preprocess(raw_database):
    """preprocess thedata"""
    pured_database = raw_database.drop_duplicates().dropna()
    pured_database = pured_database[pured_database.Temp > -30]
    pured_database = pured_database[pured_database.Temp < 60]
    return pured_database


if __name__ == '__main__':
    np.random.seed(42)
    # Question 1 - Load and preprocessing of city temperature dataset
    full_data = load_data("../datasets/City_Temperature.csv")
    preprocessed_full_data = preprocess(full_data)

    # Question 2 - Exploring data for specific country

    # keeps only samples taken in israel

    israel_samples = full_data[(full_data.Country == 'Israel')]
    israel_samples = israel_samples [ israel_samples.Temp > -30]

    israel_fig = px.scatter(israel_samples, x="Day_of_the_year", y=israel_samples['Temp'], color="Year",
                            title="temperature by day of the year in Israel") \
        .update_layout(xaxis_title="Day of the year")
    israel_fig.write_image("./plots_images/temp_israel.png")

    israel_samples_month_std = israel_samples.groupby(['Month'], as_index=False).agg(std=('Temp', 'std'))
    israel_fig_by_month_std = px.bar(israel_samples_month_std, x='Month', y=israel_samples_month_std['std'],
                                     title="std of temperature as function of month in israel")
    israel_fig_by_month_std.write_image("./plots_images/std_by_month_israel.png")

    # Question 3 - Exploring differences between countries

    #groups columns by country and month (each country has 12 rows for every month) and give its temprature
    #and adds 2 columns- one for the mean temprature and one for the std of the specific month
    all_month_std_mean = preprocessed_full_data.groupby(['Country', 'Month'], as_index=False)\
        .agg(mean_temperature=('Temp', 'mean'),
                                                                                     std=('Temp', 'std'))

    average_monthly_temp = px.line(all_month_std_mean, x='Month', y='mean_temperature', error_y='std',
                                   color='Country',
                                   title="average temperature in different countries by month").update_layout(
        yaxis_title='average temperature'
    )
    average_monthly_temp.write_image("./plots_images/average_temperature_4_countries.png")


    #question 4

    #split data into train and test (didnt want to use the other function we wrote)
    israel_train = israel_samples.sample(frac=0.75)
    israel_test = israel_samples.loc[israel_samples.index.difference(israel_train.index)]

    #this array will contain the loss for every polynomial degree from 1 to 10
    k_loss = []

    for k in range(1, 11):
        #fits a new polynomial estimator for degree k
        k_estimator = PolynomialFitting(k).fit(X = israel_train['Day_of_the_year'].
                                               to_numpy(),y= israel_train['Temp'].to_numpy())

        #adds the lost of the estimatior over the test
        k_loss.append(k_estimator.loss(israel_test['Day_of_the_year'], israel_test['Temp']
                                       .to_numpy()))
    print(k_loss)
    k_loss = enumerate(k_loss)
    loss_dataframe = pd.DataFrame(k_loss, columns=['degree', 'loss'])
    loss_dataframe['degree'] = loss_dataframe['degree'] + 1
    loss_by_degree = px.bar(loss_dataframe, x='degree', y='loss',
                            title="loss as function of polynomial degree over the test samples").update_layout(
        xaxis_title = "polynomial degree of the estimator", yaxis_title = 'mean squared loss'
    )

    loss_by_degree.write_image("./plots_images/loss_by_degree.png")

    # Question 5 - Evaluating fitted model on different countries
    #fits an estimator for degree 4 over all the israeli samples
    best_estimator = PolynomialFitting(4).fit(X = israel_samples['Day_of_the_year'].
                                               to_numpy(), y = israel_samples['Temp'].to_numpy())

    #this array will contain the MSL of the israeli model over 3 diffrent countries
    other_countries_loss = []

    # divide samples by country
    holland_samples = full_data[full_data.Country == 'The Netherlands']
    jordan_samples = full_data[full_data.Country == 'Jordan']
    s_africa_samples = full_data[full_data.Country == 'South Africa']

    #adds the MSE of the israeli model oer every country
    other_countries_loss.append(best_estimator.loss(holland_samples['Day_of_the_year'], holland_samples['Temp']))
    other_countries_loss.append(best_estimator.loss(jordan_samples['Day_of_the_year'], jordan_samples['Temp']))
    other_countries_loss.append(best_estimator.loss(s_africa_samples['Day_of_the_year'], s_africa_samples['Temp']))


    other_countries_loss =pd.DataFrame(other_countries_loss, columns= ['loss']).round(2)
    other_countries_loss['country'] = ['holland', 'jordan', 'south africa']
    best_estimator_plot = px.bar(other_countries_loss, x='country', y='loss',
                            title="loss of the israeli model over different countries", text = 'loss',
                                 color= 'country',  color_discrete_sequence = ['orange', 'green', 'red'])
    best_estimator_plot.write_image("./plots_images/country_color.png")

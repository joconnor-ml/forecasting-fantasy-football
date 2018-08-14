# Forecasting Fantasy Football

A web app for scraping, transforming and modelling Fantasy Premier League data.
Built on Apache Airflow, Scikit-Learn and Pandas.

## Requirements

`Docker`

## Installation

`git clone https://github.com/nuebar/forecasting-fantasy-football.git`

`cd forecasting-fantasy-football`

`docker-compose -f docker-compose-fff.yml up -d` or `bash install.sh`.

## Airflow

To access the Airflow UI, navigate to localhost:8080.

To access the Airflow CLI, use `bash airflow.sh`.

## Predictions

Predictions for the next week are served by a simple webapp found at localhost:5000. Try the following:
 - localhost:5000/top/<n>
 - localhost:5000/club/<club>
 - localhost:5000/position/<position>, 
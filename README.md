# Forecasting Fantasy Football

A web app for scraping, transforming and modelling Fantasy Premier League data.
Built on Apache Airflow, Scikit-Learn and Pandas.

## Requirements

`Docker`

## Installation

`git clone ...`
`cd ...`
`docker-compose -f docker-compose-fff.yml up -d`

## Airflow

To access the UI, navigate to localhost:8080.

To access the CLI, use `sudo docker exec -ti forecastingfantasyfootball_webserver_1 bash`.

## Predictions

Daily predictions are saved as CSV files to forecasting-fantasy-football/predictions.
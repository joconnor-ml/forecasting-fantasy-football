git pull && \
 docker stop forecastingfantasyfootball_postgres_1 forecastingfantasyfootball_mongo_1 forecastingfantasyfootball_webserver_1 forecastingfantasyfootball_webapp_1 && \
 docker rm forecastingfantasyfootball_postgres_1 forecastingfantasyfootball_mongo_1 forecastingfantasyfootball_webserver_1 forecastingfantasyfootball_webapp_1 && \
 docker-compose -f docker-compose-fff.yml up -d

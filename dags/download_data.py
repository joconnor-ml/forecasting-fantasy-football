"""download fpl datafrom web api to file.
Could stick them in a database? Might be a nice use of MongoDB.
Or is this somewhere to use Kafka?"""
from datetime import datetime
import requests
import time
import logging
import os

from pymongo import MongoClient

PLAYER_URL = "https://fantasy.premierleague.com/drf/element-summary/"
PLAYER_DETAIL_URL = "https://fantasy.premierleague.com/drf/bootstrap-static"
MAX_PLAYER_ID = 650

client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]


def download_data(execution_date, **kwargs):
    if True: return  # hack to skip downloading for now
    response = requests.get(PLAYER_DETAIL_URL)
    data = response.json()

    for key, val in data.items():
        collection = db[key]
        if type(val) is not list:
            continue
        for row in val:
            collection.update({"id": row["id"]}, row, upsert=True)

    logging.info("got player details")
    collection = db["player_data"]
    for player_id in range(1, MAX_PLAYER_ID + 1):
        if player_id % 100 == 0:
            logging.info(player_id)
        response = requests.get("{}/{}".format(PLAYER_URL, player_id))
        data = response.json()
        data["id"] = player_id
        data["retrieval_date"] = datetime.now().isoformat()
        collection.update({"id": data["id"]}, data, upsert=True)
        time.sleep(1)


if __name__ == "__main__":
    from datetime import datetime

    download_data(datetime.today(), None)

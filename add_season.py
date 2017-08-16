from pymongo import MongoClient

client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]

if __name__ == "__main__":
    cols = db.collection_names()

    for col in cols:
        collection = db[col]
        for doc in collection.find():
            if "season" in doc:
                continue
            else:  # if season missing, set it to 2016
                doc["season"] = 2016
                collection.update({"id": doc["id"]}, doc, upsert=True)

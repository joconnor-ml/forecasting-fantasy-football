from flask import Flask, render_template, request, redirect, url_for  # For flask implementation
from pymongo import MongoClient  # Database connector

client = MongoClient()  # Configure the connection to the database
db = client["fantasy_football"]  # Select the database
pred_db = db["predictions"]  # Select the collection

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Joe.'


@app.route("/list")
def lists():
    # Display the all Tasks
    preds = pred_db.find()
    titles = preds[0].keys()
    return render_template('index.html', preds=preds, titles=titles)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

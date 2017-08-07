from flask import Flask, render_template, request, redirect, url_for  # For flask implementation
from pymongo import MongoClient  # Database connector

client = MongoClient()  # Configure the connection to the database
db = client["fantasy_football"]  # Select the database
todos = db["predictions"]  # Select the collection

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Joe.'


@app.route("/list")
def lists():
    # Display the all Tasks
    todos_l = todos.find()
    a1 = "active"
    return render_template('index.html', a1=a1, todos=todos_l, t="Predictions", h="Week 1")


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

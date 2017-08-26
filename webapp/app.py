import os
from flask import Flask, render_template, request, redirect, url_for
from pymongo import MongoClient, DESCENDING  # Database connector
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show


client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]  # Select the database
pred_db = db["predictions"]  # Select the collection

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Joe.'


@app.route("/list")
def lists():
    # Display the all Tasks
    preds = list(pred_db.find().sort("mean_model", DESCENDING))[:-200]
    titles = preds[0].keys()
    return render_template('index.html', preds=preds, titles=titles)


@app.route("/performance")
def performance():
    # Create the plot
    p = figure(plot_width=800, plot_height=400)

    preds = pred_db.find()
    mean_preds = [pred["mean_model"] for pred in preds]

    # add a line renderer
    p.line(list(range(len(mean_preds))), mean_preds, line_width=2)

    # Set the x axis label
    p.xaxis.axis_label = "Week"

    # Set the y axis label
    p.yaxis.axis_label = "RMSE"

    # Embed plot into HTML via Flask Render
    script, div = components(p)
    return render_template("validation_plot.html", script=script, div=div)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

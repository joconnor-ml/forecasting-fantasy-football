import os
from flask import Flask, render_template, request, redirect, url_for, abort
from pymongo import MongoClient, DESCENDING  # Database connector
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show


client = MongoClient(os.environ['MONGO_URL'])
db = client["fantasy_football"]  # Select the database
pred_db = db["predictions"]  # Select the collection
score_db = db["scores"]  # Select the collection

app = Flask(__name__)


@app.route('/')
def hello_world():
    return 'Hello, Joe.'


@app.route("/top/<int:n>")
def top(n):
    # Display top n players
    preds = list(pred_db.aggregate([
        {"$sort": {"mean_model": 1}},
        {"$limit": n}
    ]))
    titles = preds[0].keys()
    return render_template('index.html', preds=preds, titles=titles)


@app.route('/position/<position>')
def position(position):
    position = position.upper()
    if position not in ["GK", "DF", "MF", "FW"]:
        return abort(404)
    preds = list(pred_db.find({"position": position}).sort("mean_model",
                                                           DESCENDING))
    titles = preds[0].keys()
    return render_template('index.html', preds=preds, titles=titles)


@app.route('/club/<club>')
def club(club):
    club = club.title()
    try:
        preds = list(pred_db.find({"name_x": club}).sort("mean_model",
                                                         DESCENDING))
        titles = preds[0].keys()
        return render_template('index.html', preds=preds, titles=titles)
    except:
        return abort(404)

@app.route("/performance")
def performance():
    # Create the plot
    p = figure(plot_width=800, plot_height=400)

    scores = score_db.find()
    mean_preds = [score["mean_model"] for score in scores]

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

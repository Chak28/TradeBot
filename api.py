from flask import jsonify, request, render_template, Response, make_response
from gevent.pywsgi import WSGIServer
import flask
import inspect
import sys
from flask_cors import CORS
from Stock_training import main

app = flask.Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": "*"
    }
})
app.config["DEBUG"] = True

@app.route("/get_stocks")
def get_selected_stocks():
    try:
        args = request.args
        ticker = args.get("ticker").split(",")
        main(tickers=ticker)
        return {"Status":"Successful"}
    except Exception as e:
        error = f"Error: {sys.exc_info()}"
        message = f"error in line number: {sys.exc_info()[-1].tb_lineno} in function: {inspect.trace()[0].function}"
        print(message,e)
        stat = {"Error": e}
        return Response(status=400, response=stat)

@app.route("/")
def home():
    return render_template("index.html")
if __name__ == '__main__':
    http_server = WSGIServer(('0.0.0.0', 3000), app)
    http_server.serve_forever()
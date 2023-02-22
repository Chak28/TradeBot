### Importing Required Libraries
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash import dcc 
from dash import html
import dash_daq as daq
import plotly.graph_objs as go
from dash.dependencies import Input, Output, State
from datetime import datetime, date
import pickle
import plotly.express as px
import flask

server = flask.Flask(__name__) # define flask app.server

## Setting up the Dashboard
app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    external_stylesheets=[dbc.themes.CYBORG],
server=server)
app.title = "Stock Trading Dashboard"

## Pulling Lookup Information for Displaying Stocks
def Lookup():
    lkup = pd.read_csv('./Data/Stock_Lookup.csv')
    return lkup

lkup = Lookup()
lkup = list(lkup['Stock'])

def logo(app):
    title = html.H5(
        "Stock Information",
        style={"marginTop": 5, "marginLeft": "10px"},
    )

    info_about_app = html.H6(
        "This Dashboard is focused on Showing the Historical Stock prices and Forecasts made by Machine Learning. "
        "The algorithm used is an ARIMA Model",
        style={"marginLeft": "10px"},
    )

    logo_image = html.Img(
        src=app.get_asset_url("yf.png"), style={"float": "right", "height": 50}
    )
    link = html.A(logo_image, href="https://plotly.com/dash/")

    return dbc.Row(
        [dbc.Col([dbc.Row([title]), dbc.Row([info_about_app])]), dbc.Col(link)]
    )

## Graphs Styling Component
graphs = dbc.Card(
    children=[
        dbc.CardBody(
            [
                html.Div(
                    [
                        dcc.Graph(id ="Training-Graph"),
                        html.Pre(id="update-on-click-data"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Graph(id ="Forecast-Graph"),
                        html.Pre(id="update-on-click-data"),
                    ],
                    style={"width": "49%", "display": "inline-block"},
                ),
                html.Div(
                    [
                        dcc.Dropdown(
                            id="stock-dropdown",
                            options=[
                               {"label": lkup, "value": lkup} for lkup in lkup
                            ],
                            value="",
                            multi=False,
                            searchable=False,
                        )
                    ],
                    style={
                        "width": "33%",
                        "display": "inline-block",
                        "color": "black",
                    },
                ),

            ],
            style={
                "backgroundColor": "black",
                "border-radius": "1px",
                "border-width": "5px",
                "border-top": "1px solid rgb(216, 216, 216)",
            },
        )
    ]
)
gauge_size = "auto"
sidebar_size = 12
graph_size = 10
app.layout = dbc.Container(
    fluid=True,
    children=[
        logo(app),
            dbc.Col(graphs,
                        xs=graph_size,
                        md=graph_size,
                        lg=graph_size,
                        width=graph_size)

    ],
)



## Training Data Callbacks Definition
@app.callback(
    [
        Output('Training-Graph', 'figure')
    ], 
    [
        Input('stock-dropdown', 'value')
    ],
)
              
def update__training_graph(value):
    
    train = pd.read_csv(f'./Data/{value}_historical_actuals.csv')
    fig = go.Figure()
    
    # Create and style traces
    fig.add_trace(go.Scatter(x=train['Date'], y=train['Close'], name='Stock Prices',
                            line=dict(color='firebrick', width=4)))
    
    # Edit the layout
    fig.update_layout(title=f'Historical Stock Prices Prediction for {value}',
                    xaxis_title='Date',
                    yaxis_title='Closing Price')

    
    return [fig]

## Forecasts Data Callbacks Definition
@app.callback(
    [
        Output('Forecast-Graph','figure')
    ], 
    [
        Input('stock-dropdown', 'value')
    ],
)
def update_forecast_graph(value):
    
    fig2 = go.Figure()
    Pred = pd.read_csv(f'./Data/{value}_Predictions.csv')
    fig2.add_trace(go.Scatter(x=Pred['Date'], y=Pred['Forecasts'], name='Forecast Stock Prices',
                            line=dict(color='green', width=4)))
    
    # Edit the layout
    fig2.update_layout(title=f'30 day Forecasted Stock Prices Prediction for {value}',
                    xaxis_title='Date',
                    yaxis_title='Forecasted Closing Price')

    return [fig2]
if __name__ == "__main__":
    app.run_server(debug=False)

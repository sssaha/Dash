import time
import uuid

import dash_ag_grid as dag
import dash_mantine_components as dmc
import snowflake.connector
from dash import (
    Dash,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    html,
    no_update,
)

app = Dash(__name__, external_stylesheets=dmc.styles.ALL)
ctx = snowflake.connector.connect(
    user="SSSAHA1989",
    password="1989S!hammyas!N",
    account="LKBZCQP-WHA95916",
    warehouse="COMPUTE_WH",
    database="LOADDEMAND",
    schema="PUBLIC",
    insecure_mode=True,
    # session_parameters={
    #     'QUERY_TAG': 'EndOfMonthFinancials',
    # }
)

cur = ctx.cursor()
sql = "Select TOP 100 * from PIDATA"
cur.execute(sql)
df = cur.fetch_pandas_all()
columnDefs = [{"field": x, "sortable": False} for x in df.columns]

app.layout = dmc.MantineProvider(
    dmc.Container(
        [
            html.Div(
                [
                    dag.AgGrid(
                        id="column-definitions-basic",
                        rowData=df.to_dict("records"),
                        defaultColDef={"filter": True},
                        columnDefs=columnDefs,
                        columnSize="sizeToFit",
                        dashGridOptions={"animateRows": False},
                    ),
                ]
            )
        ],
        p="1rem 2rem",
    ),
)


server = app.server


# if __name__ == "__main__":
#     app.run(debug=True)

# from typing import Union

# from fastapi import FastAPI
# import redis
# redis_host = "mdct.redis.cache.windows.net"
# redis_port = 6380
# redis_password = "KMlFJyj2BPCYPHgMmsd9JR6mFboCCOYRSAzCaJV9olY="

# r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
# app = FastAPI()


# @app.get("/")
# def read_root():
# return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
# # Connect to Redis (replace with your actual connection details)


# # Push data to Redis
# data = {"item_id": item_id, "q": q}
# r.set(f"item:{item_id}", str(data))
# return {"item_id": item_id, "q": q}


# # For data manipulation, visualization, app
# from dash import Dash, dcc, html, callback, Input, Output, dash_table
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import pandas as pd
# import os
# import pyarrow
# import  uuid
# import numpy as np

# # For modeling
# from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # loading Datasets
# base_path = os.path.dirname(__file__)
# file_name = 'heart_failure_clinical_records_dataset,predictions.csv'
# feature_name = 'feature_importance.csv'
# total_path = base_path + '//Data//'
# df1 = pd.read_csv(total_path + file_name)
# feature_importance = pd.read_csv(total_path + feature_name).sort_values(by=['Importance'],
# ascending=False)

# program_data_folder = os.path.join(base_path,'programdata')

# def filter_dataframe(input_df, var1, var2, var3):
# bp_list, sex_list, anaemia_list = [], [], []
# # Filtering for blood pressure
# if var1 == "all_values":
# bp_list = input_df['high_blood_pressure'].drop_duplicates()
# else:
# bp_list = [var1]
# # Filtering for sex
# if var2 == "all_values":
# sex_list = input_df['sex'].drop_duplicates()
# else:
# sex_list = [var2]
# # Filtering for Anaemia
# if var3 == "all_values":
# anaemia_list = input_df['anaemia'].drop_duplicates()
# else:
# anaemia_list = [var3]
# # Applying filters to dataframe
# input_df = input_df[(input_df['high_blood_pressure'].isin(bp_list)) &
# (input_df['sex'].isin(sex_list)) &
# (input_df['anaemia'].isin(anaemia_list))]

# return input_df


# def draw_Text(input_text):
# return html.Div([
# dbc.Card(
# dbc.CardBody([
# html.Div([
# html.H2(input_text),
# ], style={'textAlign': 'center'})
# ])
# ),
# ])


# def draw_Image(input_figure):
# return html.Div([
# dbc.Card(
# dbc.CardBody([
# dcc.Graph(figure=input_figure.update_layout(
# template='plotly_dark',
# plot_bgcolor='rgba(0, 0, 0, 0)',
# paper_bgcolor='rgba(0, 0, 0, 0)',
# )
# )
# ])
# ),
# ])


# # Returning model performance
# cmatrix = confusion_matrix(df1['DEATH_EVENT'], df1['Prediction'])

# # Building and Initializing the app
# app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE],)


# # Defining component styles
# SIDEBAR_STYLE = {
# "position": "fixed",
# "top": 0,
# "left": 0,
# "bottom": 0,
# "width": "18rem",
# "padding": "2rem 1rem",
# "background-color": "#f8f9fa",
# "display": "inline-block"
# }

# CONTENT_STYLE = {
# "margin-left": "18rem",
# "margin-right": "2rem",
# "padding": "2rem 1rem",
# "display": "inline-block",
# "width": "100%"
# }
# FILTER_STYLE = {"width": "30%"}

# # Defining components
# sidebar = html.Div(children=[
# html.H2("Description", className="display-4"),
# html.Hr(),
# html.P(
# "Tutorial project detailing how to develop a basic front end application exploring the factors influencing heart failure", className="lead"
# ),
# html.H3("Model"
# ),
# html.P(
# "This project uses a Random Forest Classifier to predict heart failure based on 12 independent variables.", className="lead"
# ),

# html.H3("Code"
# ),
# html.P(
# "The complete code for this project is available on github.", className="lead"
# ),
# html.A(
# href="https://github.com/pinstripezebra/Dash-Tutorial",
# children=[
# html.Img(
# alt="Link to Github",
# src="github_logo.png",
# )
# ],
# style={'color': 'black'}
# )

# ], style=SIDEBAR_STYLE
# )

# filters = html.Div([
# dbc.Row([
# html.Div(children=[
# html.H1('Heart Failure Prediction'),
# dcc.Markdown('A comprehensive tool for examining factors impacting heart failure'),

# html.Label('Blood Pressure'),
# dcc.Dropdown(
# id='BP-Filter',
# options=[{"label": i, "value": i} for i in df1['high_blood_pressure'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values"),

# html.Label('Sex'),
# dcc.Dropdown(
# id='Sex-Filter',
# options=[{"label": i, "value": i} for i in df1['sex'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values"),

# html.Label('Anaemia'),
# dcc.Dropdown(
# id='Anaemia-Filter',
# options=[{"label": i, "value": i} for i in df1['anaemia'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values")])
# ])
# ], style=FILTER_STYLE)

# sources = html.Div([
# html.H3('Data Sources:'),
# html.Div([
# html.Div(children=[
# html.Div([
# dcc.Markdown("""Data Description: This dataset contains 12 features that
# can be used to predict mortality by heart failure with each row representing
# a separate patient, the response variable is DEATH_EVENT.""")
# ]),
# html.Div([
# html.A("Dataset available on Kaggle",
# href='https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv', target="_blank")
# ], style={'display': 'inline-block'})
# ]),

# html.H3('Citation'),
# dcc.Markdown(
# """Davide Chicco, Giuseppe Jurman: Machine learning can predict survival
# of patients with heart failure from serum creatinine and ejection fraction alone.
# BMC Medical Informatics and Decision Making 20, 16 (2020)""")
# ])
# ])

# app.layout = html.Div(children=[
# sidebar,
# html.Div([
# filters,
# html.Div([
# dbc.Card(
# dbc.CardBody([
# dbc.Row(id='kpi-Row'),
# html.Br(),
# dbc.Row(id='EDA-Row'),
# html.Br(),
# dbc.Row(id='ML-Row'),
# sources
# ]), color='dark'
# )
# ])
# ], style=CONTENT_STYLE)
# ])


# # callback for top row
# @callback(
# Output(component_id='EDA-Row', component_property='children'),
# [Input('BP-Filter', 'value'),
# Input('Sex-Filter', 'value'),
# Input('Anaemia-Filter', 'value')]
# )
# def update_output_div(bp, sex, anaemia):
# #Making copy of DF and filtering
# filtered_df = df1
# filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)

# #Creating figures
# factor_fig = px.histogram(filtered_df, x='age', facet_col="diabetes", color='DEATH_EVENT',
# title="Age and Diabetes vs. Death")
# age_fig = px.scatter(filtered_df, x="ejection_fraction", y="serum_creatinine", facet_col="high_blood_pressure",
# color="DEATH_EVENT",
# title="Ejection Fraction and Creatinine vs. Death")
# time_fig = px.scatter(filtered_df, x='time', y='platelets', color='DEATH_EVENT',
# title='Time and Platelets vs Death')

# return dbc.Row([
# dbc.Col([
# draw_Image(factor_fig)
# ], width={"size": 3, "offset": 0}),
# dbc.Col([
# draw_Image(age_fig)
# ], width={"size": 3}),
# dbc.Col([
# draw_Image(time_fig)
# ], width={"size": 3}),
# ])


# # callback for second row
# @callback(
# Output(component_id='ML-Row', component_property='children'),
# Input('Sex-Filter', 'value')
# )
# def update_model(value):
# # Making copy of df
# confusion = cmatrix
# #x_copy = X_cols
# f_importance = feature_importance

# # Aggregating confusion dataframe and plotting
# confusion_fig = px.imshow(confusion,
# labels=dict(x="Predicted Value",
# y="True Value", color="Prediction"),
# aspect="auto",
# text_auto=True,
# title="Confusion Matrix - Predicted vs Actual Values")

# # Graphing feature importance
# feature_fig = px.bar(f_importance, x='Feature Name', y='Importance',
# title='Feature Importance')

# return dbc.Row([
# dbc.Col([
# draw_Image(feature_fig)
# ], width={"size": 5}),
# dbc.Col([
# draw_Image(confusion_fig)
# ], width={"size": 3})
# ])


# # callback for kpi row
# @callback(
# Output(component_id='kpi-Row', component_property='children'),
# [Input('BP-Filter', 'value'),
# Input('Sex-Filter', 'value'),
# Input('Anaemia-Filter', 'value')]
# )
# def update_kpi(bp, sex, anaemia):
# # Copying and filtering dataframe
# filtered_df = df1
# filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)
# file_name_pd = f'{uuid.uuid4()}.feather'
# filtered_df.to_feather(os.path.join(program_data_folder, file_name_pd))
# app.logger.info(f"Filtered dataframe size: {filtered_df.shape} - saved to file -- {file_name_pd}")
# observation_count = filtered_df.shape[0]
# death_count = filtered_df[filtered_df['DEATH_EVENT'] == 1].shape[0]
# no_death_count = filtered_df[filtered_df['DEATH_EVENT'] == 0].shape[0]

# return dbc.Row([
# dbc.Col([
# draw_Text("Observations: " + str(observation_count))
# ], width=3),
# dbc.Col([
# draw_Text("Death Count: " + str(death_count))
# ], width=3),
# dbc.Col([
# draw_Text("Survival Count: " + str(no_death_count))
# ], width=3),
# ])


# server = app.server
# # Runing the app
# if __name__ == '__main__':
# app.run(debug=False)


# gunicorn --bind=0.0.0.0 --timeout 600 app:server
# =======
# import eventlet
# eventlet.monkey_patch()
# import time
# import uuid
# from dash_socketio import DashSocketIO
# import dash_mantine_components as dmc
# from dash import Dash, Input, Output, State, callback, clientside_callback, html, no_update
# from flask_socketio import SocketIO, emit


# app = Dash(__name__, external_stylesheets=dmc.styles.ALL)
# app.server.secret_key = "Test!"

# socketio = SocketIO(app.server)

# app.layout = dmc.MantineProvider(
#     dmc.Container(
#         [
#             dmc.NotificationProvider(position="top-right"),
#             dmc.Title("Hello Socket.IO!", mb="xl"),
#             dmc.Stack(
#                 [
#                     dmc.Textarea(id="dummy", minRows=5, placeholder="Ask LoremLM something..."),
#                     html.Div(dmc.Button("Ask LoremLM", id="btn", mb="md", disabled=True)),
#                 ]
#             ),
#             dmc.Text(id="results", style={"maxWidth": "60ch"}),
#             html.Div(id="notification_wrapper"),
#             DashSocketIO(id='socketio', eventNames=["notification", "stream"]),
#         ],
#         p="1rem 2rem",
#     ),
# )


# @socketio.on("connect")
# def on_connect():
#     print("Client connected")

# @socketio.on("disconnect")
# def on_disconnect():
#     print("Client disconnected")

# def notify(socket_id, message, color=None):
#     emit(
#         "notification",
#         dmc.Notification(
#             message=message,
#             action="show",
#             id=uuid.uuid4().hex,
#             color=color,
#         ).to_plotly_json(),
#         namespace="/",
#         to=socket_id,
#     )

# paragraph = """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
# Integer augue eros, tincidunt vitae eros eu, faucibus tempus risus.
# Donec ullamcorper velit in arcu fermentum faucibus.
# Etiam finibus tortor ac vestibulum dictum. Vestibulum ultricies risus eu lacus luctus pretium.
# Duis congue et nisl eu fringilla. Mauris lorem metus, varius eget ex eget, ultrices suscipit est.
# Integer nunc risus, auctor posuere vehicula id, rutrum et urna.
# Pellentesque gravida, orci id pharetra tempus, nulla neque sagittis elit, condimentum tempor mi velit et urna.
# Fusce faucibus ac libero facilisis commodo. Quisque condimentum suscipit mi.
# Vivamus augue neque, commodo sagittis mollis sed, mollis in sapien.
# Integer cursus et magna nec cursus.
# Class aptent taciti sociosqu ad litora torquent per conubia nostra, per inceptos himenaeos.
# """

# @callback(
#     Output("results", "children"),
#     Output("notification_wrapper", "children", allow_duplicate=True),
#     Input("btn", "n_clicks"),
#     State("socketio", "socketId"),
#     running=[[Output("results", "children"), "", None]],
#     prevent_initial_call=True,
# )
# def display_status(n_clicks, socket_id):
#     if not n_clicks or not socket_id:
#         return no_update, []
#     notify(socket_id, "Sending the question to LoremLM...")
#     time.sleep(1)
#     notify(socket_id, "Streaming answer...")

#     for i, word in enumerate(paragraph.replace("\n", " ").split(" ")):
#         emit("stream", " " * bool(i) + word, namespace="/", to=socket_id)
#         time.sleep(0.05)

#     notify(socket_id, "Done!", color="green")

#     return paragraph, []

# clientside_callback(
#     """connected => !connected""",
#     Output("btn", "disabled"),
#     Input("socketio", "connected"),
# )

# clientside_callback(
#     """(notification) => {
#         if (!notification) return dash_clientside.no_update
#         return notification
#     }""",
#     Output("notification_wrapper", "children", allow_duplicate=True),
#     Input("socketio", "data-notification"),
#     prevent_initial_call=True,
# )

# clientside_callback(
#     """(word, text) => text + word""",
#     Output("results", "children", allow_duplicate=True),
#     Input("socketio", "data-stream"),
#     State("results", "children"),
#     prevent_initial_call=True,
# )

# server = app.server


# from typing import Union

# from fastapi import FastAPI
# import redis
# redis_host = "mdct.redis.cache.windows.net"
# redis_port = 6380
# redis_password = "KMlFJyj2BPCYPHgMmsd9JR6mFboCCOYRSAzCaJV9olY="

# r = redis.Redis(host=redis_host, port=redis_port, password=redis_password, decode_responses=True)
# app = FastAPI()


# @app.get("/")
# def read_root():
# return {"Hello": "World"}


# @app.get("/items/{item_id}")
# def read_item(item_id: int, q: Union[str, None] = None):
# # Connect to Redis (replace with your actual connection details)


# # Push data to Redis
# data = {"item_id": item_id, "q": q}
# r.set(f"item:{item_id}", str(data))
# return {"item_id": item_id, "q": q}


# # For data manipulation, visualization, app
# from dash import Dash, dcc, html, callback, Input, Output, dash_table
# import dash_bootstrap_components as dbc
# import plotly.express as px
# import pandas as pd
# import os
# import pyarrow
# import  uuid
# import numpy as np

# # For modeling
# from sklearn.metrics import confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# # loading Datasets
# base_path = os.path.dirname(__file__)
# file_name = 'heart_failure_clinical_records_dataset,predictions.csv'
# feature_name = 'feature_importance.csv'
# total_path = base_path + '//Data//'
# df1 = pd.read_csv(total_path + file_name)
# feature_importance = pd.read_csv(total_path + feature_name).sort_values(by=['Importance'],
# ascending=False)

# program_data_folder = os.path.join(base_path,'programdata')

# def filter_dataframe(input_df, var1, var2, var3):
# bp_list, sex_list, anaemia_list = [], [], []
# # Filtering for blood pressure
# if var1 == "all_values":
# bp_list = input_df['high_blood_pressure'].drop_duplicates()
# else:
# bp_list = [var1]
# # Filtering for sex
# if var2 == "all_values":
# sex_list = input_df['sex'].drop_duplicates()
# else:
# sex_list = [var2]
# # Filtering for Anaemia
# if var3 == "all_values":
# anaemia_list = input_df['anaemia'].drop_duplicates()
# else:
# anaemia_list = [var3]
# # Applying filters to dataframe
# input_df = input_df[(input_df['high_blood_pressure'].isin(bp_list)) &
# (input_df['sex'].isin(sex_list)) &
# (input_df['anaemia'].isin(anaemia_list))]

# return input_df


# def draw_Text(input_text):
# return html.Div([
# dbc.Card(
# dbc.CardBody([
# html.Div([
# html.H2(input_text),
# ], style={'textAlign': 'center'})
# ])
# ),
# ])


# def draw_Image(input_figure):
# return html.Div([
# dbc.Card(
# dbc.CardBody([
# dcc.Graph(figure=input_figure.update_layout(
# template='plotly_dark',
# plot_bgcolor='rgba(0, 0, 0, 0)',
# paper_bgcolor='rgba(0, 0, 0, 0)',
# )
# )
# ])
# ),
# ])


# # Returning model performance
# cmatrix = confusion_matrix(df1['DEATH_EVENT'], df1['Prediction'])

# # Building and Initializing the app
# app = Dash(__name__,external_stylesheets=[dbc.themes.SLATE],)


# # Defining component styles
# SIDEBAR_STYLE = {
# "position": "fixed",
# "top": 0,
# "left": 0,
# "bottom": 0,
# "width": "18rem",
# "padding": "2rem 1rem",
# "background-color": "#f8f9fa",
# "display": "inline-block"
# }

# CONTENT_STYLE = {
# "margin-left": "18rem",
# "margin-right": "2rem",
# "padding": "2rem 1rem",
# "display": "inline-block",
# "width": "100%"
# }
# FILTER_STYLE = {"width": "30%"}

# # Defining components
# sidebar = html.Div(children=[
# html.H2("Description", className="display-4"),
# html.Hr(),
# html.P(
# "Tutorial project detailing how to develop a basic front end application exploring the factors influencing heart failure", className="lead"
# ),
# html.H3("Model"
# ),
# html.P(
# "This project uses a Random Forest Classifier to predict heart failure based on 12 independent variables.", className="lead"
# ),

# html.H3("Code"
# ),
# html.P(
# "The complete code for this project is available on github.", className="lead"
# ),
# html.A(
# href="https://github.com/pinstripezebra/Dash-Tutorial",
# children=[
# html.Img(
# alt="Link to Github",
# src="github_logo.png",
# )
# ],
# style={'color': 'black'}
# )

# ], style=SIDEBAR_STYLE
# )

# filters = html.Div([
# dbc.Row([
# html.Div(children=[
# html.H1('Heart Failure Prediction'),
# dcc.Markdown('A comprehensive tool for examining factors impacting heart failure'),

# html.Label('Blood Pressure'),
# dcc.Dropdown(
# id='BP-Filter',
# options=[{"label": i, "value": i} for i in df1['high_blood_pressure'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values"),

# html.Label('Sex'),
# dcc.Dropdown(
# id='Sex-Filter',
# options=[{"label": i, "value": i} for i in df1['sex'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values"),

# html.Label('Anaemia'),
# dcc.Dropdown(
# id='Anaemia-Filter',
# options=[{"label": i, "value": i} for i in df1['anaemia'].drop_duplicates()] +
# [{"label": "Select All", "value": "all_values"}],
# value="all_values")])
# ])
# ], style=FILTER_STYLE)

# sources = html.Div([
# html.H3('Data Sources:'),
# html.Div([
# html.Div(children=[
# html.Div([
# dcc.Markdown("""Data Description: This dataset contains 12 features that
# can be used to predict mortality by heart failure with each row representing
# a separate patient, the response variable is DEATH_EVENT.""")
# ]),
# html.Div([
# html.A("Dataset available on Kaggle",
# href='https://www.kaggle.com/datasets/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv', target="_blank")
# ], style={'display': 'inline-block'})
# ]),

# html.H3('Citation'),
# dcc.Markdown(
# """Davide Chicco, Giuseppe Jurman: Machine learning can predict survival
# of patients with heart failure from serum creatinine and ejection fraction alone.
# BMC Medical Informatics and Decision Making 20, 16 (2020)""")
# ])
# ])

# app.layout = html.Div(children=[
# sidebar,
# html.Div([
# filters,
# html.Div([
# dbc.Card(
# dbc.CardBody([
# dbc.Row(id='kpi-Row'),
# html.Br(),
# dbc.Row(id='EDA-Row'),
# html.Br(),
# dbc.Row(id='ML-Row'),
# sources
# ]), color='dark'
# )
# ])
# ], style=CONTENT_STYLE)
# ])


# # callback for top row
# @callback(
# Output(component_id='EDA-Row', component_property='children'),
# [Input('BP-Filter', 'value'),
# Input('Sex-Filter', 'value'),
# Input('Anaemia-Filter', 'value')]
# )
# def update_output_div(bp, sex, anaemia):
# #Making copy of DF and filtering
# filtered_df = df1
# filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)

# #Creating figures
# factor_fig = px.histogram(filtered_df, x='age', facet_col="diabetes", color='DEATH_EVENT',
# title="Age and Diabetes vs. Death")
# age_fig = px.scatter(filtered_df, x="ejection_fraction", y="serum_creatinine", facet_col="high_blood_pressure",
# color="DEATH_EVENT",
# title="Ejection Fraction and Creatinine vs. Death")
# time_fig = px.scatter(filtered_df, x='time', y='platelets', color='DEATH_EVENT',
# title='Time and Platelets vs Death')

# return dbc.Row([
# dbc.Col([
# draw_Image(factor_fig)
# ], width={"size": 3, "offset": 0}),
# dbc.Col([
# draw_Image(age_fig)
# ], width={"size": 3}),
# dbc.Col([
# draw_Image(time_fig)
# ], width={"size": 3}),
# ])


# # callback for second row
# @callback(
# Output(component_id='ML-Row', component_property='children'),
# Input('Sex-Filter', 'value')
# )
# def update_model(value):
# # Making copy of df
# confusion = cmatrix
# #x_copy = X_cols
# f_importance = feature_importance

# # Aggregating confusion dataframe and plotting
# confusion_fig = px.imshow(confusion,
# labels=dict(x="Predicted Value",
# y="True Value", color="Prediction"),
# aspect="auto",
# text_auto=True,
# title="Confusion Matrix - Predicted vs Actual Values")

# # Graphing feature importance
# feature_fig = px.bar(f_importance, x='Feature Name', y='Importance',
# title='Feature Importance')

# return dbc.Row([
# dbc.Col([
# draw_Image(feature_fig)
# ], width={"size": 5}),
# dbc.Col([
# draw_Image(confusion_fig)
# ], width={"size": 3})
# ])


# # callback for kpi row
# @callback(
# Output(component_id='kpi-Row', component_property='children'),
# [Input('BP-Filter', 'value'),
# Input('Sex-Filter', 'value'),
# Input('Anaemia-Filter', 'value')]
# )
# def update_kpi(bp, sex, anaemia):
# # Copying and filtering dataframe
# filtered_df = df1
# filtered_df = filter_dataframe(filtered_df, bp, sex, anaemia)
# file_name_pd = f'{uuid.uuid4()}.feather'
# filtered_df.to_feather(os.path.join(program_data_folder, file_name_pd))
# app.logger.info(f"Filtered dataframe size: {filtered_df.shape} - saved to file -- {file_name_pd}")
# observation_count = filtered_df.shape[0]
# death_count = filtered_df[filtered_df['DEATH_EVENT'] == 1].shape[0]
# no_death_count = filtered_df[filtered_df['DEATH_EVENT'] == 0].shape[0]

# return dbc.Row([
# dbc.Col([
# draw_Text("Observations: " + str(observation_count))
# ], width=3),
# dbc.Col([
# draw_Text("Death Count: " + str(death_count))
# ], width=3),
# dbc.Col([
# draw_Text("Survival Count: " + str(no_death_count))
# ], width=3),
# ])


# server = app.server
# # Runing the app
# if __name__ == '__main__':
# app.run(debug=False)


# gunicorn --bind=0.0.0.0 --timeout 600 app:server

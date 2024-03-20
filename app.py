import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
tqdm.pandas()
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.tsaplots import plot_acf, acf
from sklearn.ensemble import RandomForestRegressor
import statsmodels.api as sm
from functools import reduce
from sklearn.linear_model import LinearRegression

df = pd.read_csv('data/final_data_day.csv')
df_15 = pd.read_csv('data/final_data_15min.csv')
df_15['DATE'] = pd.to_datetime(df_15['DATE'])

# Filter rows where the year is greater than 2020
df_15_bis = df_15[df_15['DATE'].dt.year > 2020]
df_15_bis = df_15_bis[df_15_bis['ATTENDANCE'] > 0]
df_15_bis = df_15_bis[df_15_bis['GUEST CARRIED'] > 0]
df_15_bis.sort_values('DATE', inplace = True)
df_15_bis.drop(columns = ['START OF 1 RIDE FOR ATTRACTION','END OF 1 RIDE FOR ATTRACTION', 'ATTRACTION'], inplace = True)


# Assuming df_15_bis is already prepared
rush_hour_df = df_15_bis[(df_15_bis['HOUR START OF 1 RIDE'] > 9) & (df_15_bis['HOUR START OF 1 RIDE'] < 17)]
rush_hour_df['Ratio People/Capacity'] = rush_hour_df['ATTENDANCE'] / rush_hour_df['CAPACITY']
capacity_analysis = rush_hour_df.groupby('ATTRACTION NAME')['CAPACITY'].mean().reset_index()
ratio_analysis = rush_hour_df.groupby('ATTRACTION NAME')['Ratio People/Capacity'].mean().reset_index()
average_wait_time_by_attraction = rush_hour_df.groupby('ATTRACTION NAME')['WAIT TIME'].mean().reset_index()

# Combine data frames
data_frames = [ratio_analysis, capacity_analysis, average_wait_time_by_attraction]
combined_analysis = reduce(lambda left, right: pd.merge(left, right, on='ATTRACTION NAME'), data_frames)
combined_analysis.sort_values('WAIT TIME', ascending=False)
# Créez le DataFrame `combined_analysis` si ce n'est pas déjà fait
combined_analysis['Average Wait Time'] = rush_hour_df.groupby('ATTRACTION NAME')['WAIT TIME'].mean().values
combined_analysis = combined_analysis.dropna(subset=['Ratio People/Capacity'])



# Assuming df_15_bis is your DataFrame
available_attractions = df_15_bis['ATTRACTION NAME'].unique()

app = dash.Dash(__name__)

# Set a white background and use external CSS for better styling
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

card_style = {
    'textAlign': 'center',
    'color': 'black',  # Changer la couleur du texte en noir
    'padding': '20px',  # Augmenter le padding pour plus d'espace autour du texte
    'margin': '10px',
    'border-radius': '5px',  # Arrondir les coins
    'background-color': 'rgba(245, 245, 245, 0.9)',  # Gris clair avec une légère transparence
    'display': 'inline-block',
    'width': '20%',  # Ajustez en fonction de la largeur de votre layout
    'fontSize': '20px',  # Augmenter la taille du texte
    'boxShadow': '0 4px 8px 0 rgba(0,0,0,0.2)'  # Optionnel : ajouter une ombre pour un effet de profondeur
}




app.layout = html.Div([
    html.H1("Attraction Wait Time Prediction", style={'textAlign': 'center'}),
    html.Div([
    html.Div([
        html.H3('Avg Waiting Time'),
        html.P(id='kpi-average-waiting-time'),  # Ajout d'un id pour mise à jour dynamique
    ], style=card_style),
    
    html.Div([
        html.H3('Total Attendance'),
        html.P(id='kpi-global-attendance'),  # Ajout d'un id pour mise à jour dynamique
    ], style=card_style),
    
    html.Div([
        html.H3('Waiting Time / Attendance'),
        html.P(id='kpi-ratio-waiting-attendance'),  # Ajout d'un id pour mise à jour dynamique
    ], style=card_style),
], style={'textAlign': 'center'}),
    html.Div([
        dcc.Dropdown(
            id='attraction-dropdown',
            options=[{'label': i, 'value': i} for i in available_attractions],
            value='Dizzy Dropper',  # Set a default value
            style={
        'width': '60%',  # Largeur du dropdown
        'display': 'inline-block',
        'height': '50px',  # Augmenter la hauteur du dropdown
        'fontSize': '25px',  # Taille de la police plus grande pour le texte
    }
        ),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=df_15_bis['DATE'].min(),
            end_date=df_15_bis['DATE'].max(),
            display_format='YYYY-MM-DD',
            style={
        'width': '60%',  # Largeur du dropdown
        'display': 'inline-block',
        'height': '50px',  # Augmenter la hauteur du dropdown
        'fontSize': '25px',  # Taille de la police plus grande pour le texte
    }
        )
    ], style={'textAlign': 'center', 'backgroundColor': 'white'}),
    dcc.Graph(id='prediction-plot'),
    html.Div(id='error-output', style={'textAlign': 'center', 'fontSize': 20, 'backgroundColor': 'white'}),
    dcc.Graph(id='heatmap-plot'),  # New Graph component for the heatmap
    dcc.Graph(id='capacity-wait-time-plot')
], style={'width': '80%', 'margin-left': 'auto', 'margin-right': 'auto', 'backgroundColor': 'white'})

@app.callback(
    [Output('prediction-plot', 'figure'),
     Output('error-output', 'children'),
     Output('heatmap-plot', 'figure')],  # Adding Output for the heatmap
    [Input('attraction-dropdown', 'value'),  # Listening to changes in dropdown value
     Input('date-picker-range', 'start_date'),  # Listening to changes in start date
     Input('date-picker-range', 'end_date')]  # Listening to changes in end date
)
def update_output(selected_attraction, start_date, end_date):
    # Filter the dataset for the selected attraction and date range
    filtered_df = df[(df['ATTRACTION NAME'] == selected_attraction) & 
                     (df['DATE'] >= start_date) & 
                     (df['DATE'] <= end_date) ]

    filtered_df_15 = df_15_bis[(df_15_bis['ATTRACTION NAME'] == selected_attraction) &
                                 (df_15_bis['DATE'] >= start_date) &
                                    (df_15_bis['DATE'] <= end_date)]

    # Assuming 'DATE' is the index
    filtered_df = filtered_df.reset_index().rename(columns={'DATE': 'ds', 'WAIT TIME': 'y'})


    # Initialize and fit the Prophet model
    model = Prophet(weekly_seasonality=True, daily_seasonality=True, yearly_seasonality=True)
    model.fit(filtered_df)

    # Make future predictions
    future_dates = model.make_future_dataframe(periods=365)
    forecast = model.predict(future_dates)

    # Calculate MSE for the last 90 days as an example
    actuals = filtered_df['y'][-90:]
    predictions = forecast['yhat'][-90:]
    mse = mean_squared_error(actuals, predictions)
    mean_ = actuals = filtered_df['y'][-365:].mean()
    error = round(abs(( np.sqrt(mse) - mean_ ))/ mean_ *100, 1)

    # Prepare the plotly figure for predictions
    figure = {
        'data': [
            {'x': forecast['ds'], 'y': forecast['yhat'], 'type': 'line', 'name': 'Predicted'},
            {'x': filtered_df['ds'], 'y': filtered_df['y'], 'type': 'line', 'name': 'Actual'},
        ],
        'layout': {
            'title': 'Predictions vs Actuals',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'WAIT TIME'},
        }
    }

    # Heatmap Data Preparation
    filtered_df_15['DATE'] = pd.to_datetime(filtered_df_15['DATE'])  # Ensure 'ds' is datetime for dayofweek
    filtered_df_15['day_of_week'] = filtered_df_15['DATE'].dt.dayofweek
    filtered_df_15['day_of_week'] = filtered_df_15['day_of_week'].sort_values()
    filtered_df_15['day_of_week_name'] = filtered_df_15['DATE'].dt.day_name()

    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_df = filtered_df_15.pivot_table(index='day_of_week_name', columns='HOUR START OF 1 RIDE', values='WAIT TIME', aggfunc=np.mean)
    
    # Réordonnez l'index de heatmap_df selon l'ordre des jours de la semaine
    heatmap_df = heatmap_df.reindex(days_order)

    heatmap_fig = px.imshow(heatmap_df, labels=dict(x="Hour of Start", y="Day of Week", color="Average Wait Time"),
                            aspect="auto", color_continuous_scale='Viridis')
    heatmap_fig.update_layout(title='Average Wait Time Heatmap')

    # Return the figures and MSE text
    return figure, f'Relative deviation of the prediction from the mean : {error} %', heatmap_fig


@app.callback(
    Output('capacity-wait-time-plot', 'figure'),
    [Input('attraction-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_capacity_wait_time_plot(selected_attraction, start_date, end_date):
    # Filtrer `combined_analysis` basé sur `selected_attraction` et la plage de dates, si applicable
    
    # Création du graphique
    fig = px.scatter(combined_analysis, x='CAPACITY', y='Average Wait Time',
                     size='Ratio People/Capacity', color='ATTRACTION NAME',
                     hover_name='ATTRACTION NAME', size_max=60,
                     title="Link between Capacity and Wait Time")
    
    # Ajustements du layout
    fig.update_layout(xaxis_title="Average Capacity",
                      yaxis_title="Average Waiting Time",
                      legend_title="Attraction")
    
    return fig


@app.callback(
    [Output('kpi-average-waiting-time', 'children'),
     Output('kpi-global-attendance', 'children'),
     Output('kpi-ratio-waiting-attendance', 'children')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_kpis(start_date, end_date):
    # Filtrez votre DataFrame en fonction de la plage de dates sélectionnée
    filtered_df = df[(df['DATE'] >= start_date) & (df['DATE'] <= end_date)]
    
    # Calcul des KPIs
    average_waiting_time = filtered_df['WAIT TIME'].mean()
    global_attendance = filtered_df['ATTENDANCE'].sum()
    
    # Vérifiez si global_attendance est non nul pour éviter la division par zéro
    ratio = (filtered_df['WAIT TIME'].sum() / global_attendance * 1000000) if global_attendance else 0
    
    # Formatez les valeurs pour l'affichage
    avg_wait_formatted = "{:.2f} min".format(average_waiting_time)
    global_attendance_formatted = "{:,}".format(global_attendance)
    ratio_formatted = "{:.2f} min/1M visitors".format(ratio)
    
    return avg_wait_formatted, global_attendance_formatted, ratio_formatted


if __name__ == '__main__':
    app.run_server(debug=True, port=8051)

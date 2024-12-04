import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

import os
import subprocess
import time
import json
import requests
import threading
from datetime import datetime
from joblib import load
from scipy.spatial.distance import cosine

from flask import Flask
import dash
from dash import Dash, dcc, html, dash_table, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px

try:
    import xgboost
except ImportError:
    subprocess.check_call([os.sys.executable, "-m", "pip", "install", "xgboost"])
    import xgboost

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", 100)
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
update_in_progress = False
update_complete = False
data, latest_distribution, next_demands, next_renewables = None, None, None, None
next_objective_values, next_optimized_co2, next_optimal_combinations = None, None, None
fig1, fig2, fig3 = None, None, None


#I. PIPELINE

## accessing the open energy data from Energinet Denmark, through their public API
def get_PowerSystem_data():

    url = 'https://api.energidataservice.dk/dataset/PowerSystemRightNow?start=now-P1D&end=now&sort=Minutes1DK'
    response = requests.get(url)

    if response.status_code != 200:
        print("Request failed with status code:", response.status_code)

    selected_records = response.json()['records'][::5]
    df = pd.DataFrame(selected_records).fillna(0)

    total_generation = df.ProductionGe100MW + df.ProductionLt100MW + df.SolarPower + df.OffshoreWindPower + df.OnshoreWindPower

    net_imports = df.Exchange_DK1_DE + df.Exchange_DK1_NL + df.Exchange_DK1_GB + df.Exchange_DK1_NO + df.Exchange_DK1_SE + \
                    df.Exchange_DK1_DK2 + df.Exchange_DK2_DE + df.Exchange_DK2_SE + df.Exchange_Bornholm_SE

    imbalance = df.ImbalanceDK1 + df.ImbalanceDK2

    df['Demand'] = total_generation + net_imports - imbalance
    df['Renewables'] = df.SolarPower + df.OffshoreWindPower + df.OnshoreWindPower
    actionable = df['Demand'] - df['Renewables']

    return df

data = get_PowerSystem_data()


#II. LOAD MODELS

try:
    infer_co2_levels = load('decision_tree_regressor.joblib')
    demand_xgb_models = load('XGBoost_regressor_demand.joblib')
    renewables_xgb_models = load('XGBoost_regressor_renewables.joblib')
except Exception as e:
    print(f"Model loading failed: {e}")

def latest_and_nextDR():

    X = data.iloc[:, 3:-2]
    X1 = X.iloc[:, :2]
    X2 = X.iloc[:, 6:-6]
    X = pd.concat([X1,X2], axis=1).to_numpy()
    latest_distribution = X[-1]

    latest = data.Demand.iloc[-10:]
    next_demands = []

    for demand_model in demand_xgb_models:
        next_demands.append(demand_model.predict([latest])[0])

    next_demands = np.round(next_demands,2)
    next_demands

    latest_re = data.Renewables.iloc[-10:]
    next_renewables = []

    for renewables_model in renewables_xgb_models:
        next_renewables.append(renewables_model.predict([latest_re])[0])

    next_renewables = np.round(next_renewables,2)
    next_renewables

    return latest_distribution, next_demands, next_renewables

latest_distribution, next_demands, next_renewables = latest_and_nextDR()


#III. GENETIC ALGORITHM FOR OPTIMAL ENERGY DISTRIBUTION

##setting min and max for gene variables
def generate_chromosome():
    chromosomes = np.array([random.choice(range(70,2300)), #boundaries for ProductionGe100MW
                           random.choice(range(20,800)), #boundaries for ProductionLt100MW
                           #minimum values chosen to maintain spinning reserve necessary for frequency balance during days with high renewable energy production

                           random.choice(range(-3000,3000)), #interconnector limits for exchanges between DK1-DE
                           random.choice(range(-750,750)), #interconnector limits for exchanges between DK1-NL
                           random.choice(range(-1500,1500)), #DK1-GB
                           random.choice(range(-1800,1800)), #DK1-NO
                           random.choice(range(-800,800)), #DK1-SE
                           random.choice(range(-600,600)), #DK1-DK2
                           
                           random.choice(range(-1000,1000)), #DK2-DE
                           random.choice(range(-1900,1900)), #DK2-SE
                           random.choice(range(-60,40))]) #Bornholm-SE
    
    return chromosomes

def generate_population(size):
    pop = np.array([generate_chromosome() for _ in range(size)])
    return pop

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def cosine_similarity(matrix, vector):

    vector = vector / np.linalg.norm(vector)
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_normalized = matrix / matrix_norms

    return np.dot(matrix_normalized, vector)

def vectors_cosine_similarity(a, b):
    return 1 - cosine(a, b)

def fitness(pop, iteration):
    
    renewable_gene = np.full((pop.shape[0],1), next_renewables[iteration])
    pop_re = np.hstack((renewable_gene, pop))
    
    co2_evaluation = infer_co2_levels.predict(pop_re.tolist())

    penalty = 10**4 
    distributable_energy = next_demands[iteration] - next_renewables[iteration]
    distributable_broadcasted = np.full((pop.shape[0]), distributable_energy)

    pop_valued = np.hstack((pop, co2_evaluation.reshape(pop.shape[0],1), np.sum(pop, axis=1).reshape(pop.shape[0],1)))
    pop_valued[:,-2] = penalty * (np.abs(distributable_broadcasted - pop_valued[:,-1]) > 0.01*next_demands[iteration]) + (math.sqrt(penalty) * pop_valued[:,-2]) - \
                        penalty * (cosine_similarity(pop, latest_distribution))
    ##this is the function that needs to be minimized
    ##chromosomes that do not balance the power grid are penalized with a large number that will significantly reduce their coupling chances

    pop_valued = pop_valued[:,:-1]

    return pop_valued, co2_evaluation

def selection(pop_valued, co2_evaluation, beta):

    ascsort = np.argsort(pop_valued[:,-1])
    sorted_generation = pop_valued[ascsort]

    m,n = sorted_generation.shape 
    selected_genotypes = sorted_generation[:,:-1]
    selected_scores = sorted_generation[:,-1]
    
    choices = np.arange(selected_genotypes.shape[0])
    choices_norm = beta*((choices - np.mean(choices))/np.std(choices))  
                    #beta controls the trade-off between exploration and exploitation (lower beta means more exploration)
    coupling_probabilities = sorted(softmax(choices_norm), reverse=True)

    couples = np.array([np.random.choice(choices, 2, 
                replace=False, p=coupling_probabilities)
                for _ in range(int(m/2)+1)])
    
    return selected_genotypes, sorted_generation, couples

def crossover(selected_genotypes, couples):
    
    crossm = []
    
    for i in range(couples.shape[0]):
        x = selected_genotypes[couples[i][0]]
        y = selected_genotypes[couples[i][1]]
        xx = x.copy()
        yy = y.copy()
        
        mask = np.array(random.choices([0,1], weights=[3,1],
                            k=selected_genotypes.shape[1]))
        one = np.where(mask==1)
        
        xx[one] = y[one]
        yy[one] = x[one]
        
        crossm.append(np.array(xx))
        crossm.append(np.array(yy))
        
    crossover_matrix = np.array(crossm)
    
    return crossover_matrix

def mutation(crossover_matrix):
    
    mutation_matrix = crossover_matrix.copy()
    
    for i in range(mutation_matrix.shape[0]):
        mutation_decision = np.random.choice([0,1],1)
        mutation_locus = np.random.choice(mutation_matrix.shape[1],1)
    
        if mutation_decision == 1:
            mutation_matrix[i][mutation_locus] = generate_chromosome()[mutation_locus]
            
    return mutation_matrix

def new_generation(mutation_matrix, selected_genotypes):

    new_pop = np.vstack((mutation_matrix, 
                    selected_genotypes[0]))
    new_generation = np.unique(new_pop, axis=0)
    
    return new_generation

def evolution(size, epochs, beta, iteration):
    pop = generate_population(size)
        
    for k in range(epochs):
        pop_valued, co2_evaluation = fitness(pop, iteration)
        selected_genotypes, sorted_generation, couples = selection(pop_valued, co2_evaluation, beta)
        
        crossover_matrix = crossover(selected_genotypes, couples)
        mutation_matrix = mutation(crossover_matrix)        
        pop = new_generation(mutation_matrix, selected_genotypes)
               
    optimal_obj_value = min(sorted_generation[:,-1])
    optimal_combination = selected_genotypes[0]
    optimized_co2_level = infer_co2_levels.predict([[next_renewables[iteration]] + selected_genotypes[0].tolist()])
    
    return optimal_obj_value, optimal_combination, optimized_co2_level[0]

def genetic_ensemble(n_optimizers, size, epochs, beta):

    next_objective_values = []
    next_optimized_co2 =[]
    next_optimal_combinations = []

    for iteration in range(len(next_demands)):
        
        objective_values =[]
        combinations = []
        runs = []

        ##ensemble approach to escape local optima
        for i in range(n_optimizers):
            optimal_obj_value, optimal_combination, optimized_co2_level = evolution(size, epochs, beta, iteration)

            objective_values.append(optimal_obj_value)
            combinations.append(optimal_combination)
            runs.append(optimized_co2_level)

        optimized_obj = min(objective_values)
        optimized_co2 = runs[np.where(np.array(objective_values) == min(objective_values))[0][0]]
        optimal_distribution = list(combinations[np.where(np.array(objective_values) == min(objective_values))[0][0]])
    
        next_objective_values.append(optimized_obj)
        next_optimized_co2.append(optimized_co2)
        next_optimal_combinations.append(optimal_distribution)

    next_objective_values = np.array(next_objective_values)
    next_optimized_co2 = np.array(next_optimized_co2)
    next_optimal_combinations = np.array(next_optimal_combinations)

    return next_objective_values, next_optimized_co2, next_optimal_combinations

next_objective_values, next_optimized_co2, next_optimal_combinations = genetic_ensemble(25, 100, 150, 8)


#IV. VISUALIZATION

title = "REDUCING CO₂ LEVELS USING GENETIC ALGORITHMS"

main_text1 = """
            This project aims to provide a solution framework for minimizing CO₂ emission levels in a national power grid, through machine learning methods. 
            The data is collected via Energinet's public API which provides real-time data about Denmark's power system. 
            The dataset prepared for this analysis consists of data recorded every 5 minutes for the past year.

            The main elements of this project are: a regression tree for CO₂ levels, XGBoost regressors for predicting demand and renewable energy production
            (trained on the past year data) and genetic algorithms for finding the optimal energy distribution with respect to CO₂ emission minimization. 
            The goal is to infer the optimal resource combination that minimizes CO₂ emissions, depending on future demand and
            renewables production. Fig 1 shows a comparison between emissions in the past hour in Denmark's power system and the minimized CO₂ levels 
            in the next 5 time points with optimal energy distribution:

            
            """
main_text2 = """
            The methodology is described below:

            I. A decision tree regressor infers the CO₂ levels taking into account 12 features: renewable energy production (solar and wind), 
            energy produced by power plants with installed capacity greater or equal to 100 MW, 
            energy produced by power plants with installed capacity less than 100 MW, and energy exchanges between Denmark and other countries or areas (DK1-DE,
            DK1-NL,
            DK1-GB,
            DK1-NO,
            DK1-SE,
            DK1-DK2,
            DK2-DE,
            DK2-SE,
            Bornholm-SE); positive exchange values are imports, while the negative ones are exports. The hyperparameters of the regressor are tuned using grid search, 
            and the model is evaluated through 5-fold cross-validation.

            II. The XGBoost regressors provide multi-step forecasting for energy demand and renewable production for the next 5 time points. 
            The hyperparameters are tuned using grid search. These forecasts are important because the optimal power distribution must match demand constantly 
            to keep the grid frequency at 50 Hz 
            (within a margin of 0.1 Hz, or $\pm1\%$ of the total demand). Considering that renewable energy depends on weather conditions, 
            we will need a forecasted amount of distributable energy whose allocation will be optimized among the mentioned sources (except for renewables, of course); 
            this quantity will be represented by the difference between the forecasted demand and forecasted renewable energy.

            III. For each of the 5 forecasted steps, the optimal combination of sources is inferred using genetic algorithms. 
            In order to escape eventual local optima and increase the success of the method, this is done by employing a genetic ensemble with 25 optimizers. 
            Populations of solutions are initialized stochastically and transformed by evolutionary processes to arrive at the "fittest" distribution, 
            exploring a search space of approximately $2 \cdot 10^{32}$ combinations. The restrictions are the ranges of power plant production and the interconnector limits for exchanges (as inferred from past year's data). 
            These limits are considered at initialization time and respected throughout the process, during random mutation events. 
            Another condition, as we have seen, is that the distributable energy matches the difference between forecasted demand and forecasted renewables; 
            this is addressed directly in the fitness function.

            The first generation is of the form:

            $$
            G = 
            \\begin{bmatrix}
            - & x^{(1)} & - \\\\
            - & x^{(2)} & - \\\\
            \\phantom{0} & . & \\phantom{0} \\\\
            \\phantom{0} & . & \\phantom{0} \\\\
            \\phantom{0} & . & \\phantom{0} \\\\
            - & x^{(m)} & -
            \\end{bmatrix} \\in \\mathbb{R}^{m \\times d}
            $$

            where $m$ is the size if the population, $d$ is the number of genes within each chromosome $x^{(j)}$ and each $x^{(j)}$ represents a vector 
            comprising representations of genes, whose values have to be optimized through selection, crossover and mutation. Each chromosome is evaluated 
            using the regression tree to infer the CO₂ emission level implied by the solution it represents, while solutions that do not balance 
            the power grid are penalized by an augmentation that will significantly reduce their coupling chances. Some simplifying approaches are in place, 
            as the model does not take into consideration other factors that might influence the quantities, like exchange contracts with neighboring countries 
            or energy prices in the area. However, the fitness function also includes a term whose role is to maximize the similarity between the proposed solutions 
            and the latest distribution in the dataset. This makes the optimal solutions for the next 5 time points more realistic and also makes sure there is a 
            certain coherence between them, as much as possible, avoiding crazy swings of production or exchange values, 5 minutes apart. Thus, each optimal solution 
            minimizes the fitness function composed of these elements: grid imbalance penalty, negative resemblance and CO₂ level.

            $$
            x^* = \\underset{x^{(j)}}{\\operatorname{argmin}} \\; \\left[ \\xi \\left( \\mathbf{I}\\left(\\left| D_{XG} - R_{XG} - \\sum_{i=1}^{d} x_i^{(j)} \\right| > 0.01 D_{XG}\\right) - \\cos\\left(x^{(j)}, \\lambda\\right) \\right) + \\zeta \\Delta_{CO_2} \\right]
            $$

            where:
            - $\\xi$ is the penalty hyperparameter,
            - $\\mathbf{I}$ is the indicator function,
            - $D_{XG}$ is the demand forecast provided by the XGBoost regressor,
            - $R_{XG}$ is the renewable energy forecast provided by the XGBoost regressor,
            - $\\cos(x^{(j)}, \\lambda)$ measures the cosine similarity between the chromosome and the latest energy distribution in the dataset,
            - $\\Delta_{CO_2}$ is the CO₂ emission level inferred by the decision tree regressor.

            The indicator function $\\mathbf{I}$ checks whether the solution balances the grid within a margin of 1%. 
            If not, the chromosome is penalized with $\\xi$, so that the probability of coupling for crossover is reduced considerably. 
            The genetic heuristic encourages this term to be 0.

            The second term, $\\cos(x^{(j)}, \\lambda)$, computes the cosine similarity between $x^{(j)}$ and the latest distribution $\\lambda$, 
            defined as their dot product divided by their norm: $\\frac{x^{(j)} \\cdot \\lambda}{\\|x^{(j)}\\| \\phantom{|}\\|\\lambda\\|}$, giving results between -1 
            (completely opposite vectors) and 1 (identical vectors). This term is taken with a negative sign in the fitness function, so that the similarity is 
            maximized. In this case, the role of $\\xi$ is to augment the cosine result to make it detectable in the optimization process.

            The third term minimizes the CO₂ level inferred by the regression tree, augmented by $\\zeta$.

            Ultimately, every 5 minutes, this model produces 5 forecasted time points presenting values for power plant production and exchanges that minimize 
            CO₂ levels, balance the grid and, at the same time, maximize the resemblance with the latest distribution in order to increase realism and 
            coherence (Fig 2). This model has provided solutions that represented potential 50%-75% CO₂ emission reductions.

            """

text_fig1 = """
Fig 1: The purple area represents the CO₂ emission levels in Denmark's power grid in the past hour. 
The light blue area shows how the CO₂ levels would be in the next 5 timepoints if the resources in the dataset were allocated according 
to the genetic optimization.
"""

text_fig2 = """
Fig 3: Relationship between power plant energy production, renewable energy, total net exchanges and CO₂ emissions, for the past 24 hours. 
CO₂ levels are represented as continuous color gradient from light blue (low emission levels) to intense red (high emission levels).
Drag the chart with the left click to rotate the figure.
"""

table_title = "Optimal Resource Allocation in the Next 5 Time Points"

text_fig3 = """
Fig 2: The blue cells display the values for the next 5 timepoints in terms of power plant production and energy exchanges, optimized for minimizing the CO₂ 
emission levels, as resulted from the evolutionary process of the genetic heuristic. The green cells show the forecasted demand and renewables which condition 
the forecasted distributable energy for each of the future timepoints, important in the evaluation of chromosomes.
Finally, the last column shows the minimized CO₂ levels 
that would result from the optimal resource allocation displayed in the blue area of the chart.
"""

row_colors_T = ['#666666', '#808080', '#B3B3B3', '#CCCCCC', '#E6E6E6']

row_colors1 = ['#506A7A', '#618092', '#6E93AA', '#86B3CF', '#94C5E4']

row_colors2 = ['#697E77', '#748E86', '#809B92', '#97B9AD', '#AAD1C4']


server = Flask(__name__)

@server.route("/")
def home():
    return "Welcome to the CO2 Reduction Dashboard! Go to '/desktop-view' to view the Dash app."

print("Initializing Dash...")
app = Dash(__name__, server=server, url_base_pathname="/desktop-view/")
print("Dash initialized successfully.")

viewport_width = app.config['suppress_callback_exceptions']
base_font_size = 10
font_scaling_factor = 0.02

responsive_font_size = base_font_size + (font_scaling_factor * viewport_width)
responsive_tick_font_size = base_font_size - 1 + (font_scaling_factor * viewport_width)


def visuals():
    data_viz1 = data.iloc[:, 3:]
    data_viz1_1 = data_viz1.iloc[:, :2]
    data_viz1_2 = data_viz1.iloc[:, 6:-8]
    data_viz2 = data.iloc[:,-2:]

    data_viz = pd.concat([data.iloc[:, 1], data_viz1_1, data_viz1_2, data_viz2, data.iloc[:, 2]], axis=1)
    data_viz.Minutes1DK = pd.to_datetime(data_viz.Minutes1DK)
    data_viz = data_viz.reset_index(drop=True)

    blind_point = pd.concat([pd.DataFrame([data_viz.iloc[-1,0] + pd.Timedelta(minutes=5)]), data_viz.iloc[-1,1:]]).T
    blind_point.columns = data_viz.columns

    next_timestamps = [data_viz.iloc[-1,0] + pd.Timedelta(minutes=10), data_viz.iloc[-1,0] + pd.Timedelta(minutes=15),
                   data_viz.iloc[-1,0] + pd.Timedelta(minutes=20), data_viz.iloc[-1,0] + pd.Timedelta(minutes=25),
                   data_viz.iloc[-1,0] + pd.Timedelta(minutes=30)]

    next_df = pd.concat([pd.DataFrame(next_timestamps), pd.DataFrame(next_optimal_combinations), pd.DataFrame(next_demands), 
                        pd.DataFrame(next_renewables), pd.DataFrame(next_optimized_co2)], axis=1).reset_index(drop=True)

    next_df.columns = data_viz.columns

    data_and_next = pd.concat([data_viz, blind_point, next_df], ignore_index=True)


    table_visual = next_df.copy()
    table_visual.columns = ['Date&Time', 'Power Plants Ge100MW', 'Power Plants Lt100MW',
        'DK1-DE', 'DK1-NL', 'DK1-GB',
        'DK1-NO', 'DK1-SE', 'DK1-DK2',
        'DK2-DE', 'DK2-SE', 'Bornholm-SE', 'Demand Forecast',
        'Renewables Forecast', 'CO₂ Optimized']

    #table_visual['Demand Forecast'] = table_visual['Demand Forecast'].round()
    table_visual.iloc[:,1:] = table_visual.iloc[:,1:].round()


    ## Fig 1: CO2 levels chart - history and next optimization

    start_point = len(data_and_next) - 18
    split_point = len(data_and_next) - 5

    tooltip_main = [f"{t}<br>CO<sub>2</sub> Level: {v}" for t, v in zip(data_and_next['Minutes1DK'][start_point:split_point], 
                                                                        data_and_next['CO2Emission'][start_point:split_point])]
    tooltip_last5 = [f"{t}<br>CO<sub>2</sub> Level: {v}" for t, v in zip(data_and_next['Minutes1DK'][split_point:], 
                                                                        data_and_next['CO2Emission'][split_point:])]


    ### Trace for the main area (first n-5 timepoints)
    trace_main = go.Scatter(
        x=data_and_next['Minutes1DK'][start_point:split_point],
        y=data_and_next['CO2Emission'][start_point:split_point],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#836c8c', width=2),
        name='past hour',
        text=tooltip_main,
        hovertemplate='%{text}'
    )

    ### Trace for the last 5 timepoints
    trace_last5 = go.Scatter(
        x=data_and_next['Minutes1DK'][split_point:],
        y=data_and_next['CO2Emission'][split_point:],
        fill='tozeroy',
        mode='lines',
        line=dict(color='#2EF6F9', width=2, dash='dash'),
        fillcolor='#aad1c4',
        name='with optimal distribution',
        text=tooltip_last5,
        hovertemplate='%{text}'
    )

    y_min = min(data_and_next['CO2Emission'])
    y_max = max(data_and_next['CO2Emission'])

    ### Create the figure and add traces
    fig1 = go.Figure(data=[trace_main, trace_last5])

    fig1.update_layout(
        autosize=True,
        xaxis=dict(
            showgrid=False,
            tickfont=dict(
                color='#f7deb2',
                size=responsive_tick_font_size
            )
        ),
        yaxis=dict(
            title=dict(
                text='CO<sub>2</sub> Levels (g/kWh)',
                font=dict(color='#f7deb2', size=responsive_font_size),
            ),
            showgrid=False,
            tickfont=dict(
                color='#f7deb2',
                size=responsive_tick_font_size
            )
        ),
        legend=dict(
            x=0,                  
            y=1.2,                  
            xanchor='center',
            yanchor='bottom',
            font=dict(color='white', size=responsive_font_size - 2),  # Slightly smaller legend font
        ),
        showlegend=True,
        plot_bgcolor='rgb(47,55,79)',
        paper_bgcolor='rgb(47,55,79)',
        margin=dict(l=20, r=20, t=50, b=20)
    )


    ## Fig 2: Relationship between renewables, power plant production and CO2 levels

    custom_color_scale = ['#2EF6F9', 'orange', '#9D4F54']

    fig2 = px.scatter_3d(data, x='ProductionGe100MW', y='Renewables', z='Exchange_Sum', color='CO2Emission', color_continuous_scale=custom_color_scale,
                    range_color=[30, 100], title = 'Relationship Between Main Variables',
                    labels = {'CO2Emission': 'CO₂ Level'})

    fig2.update_layout(
        autosize=True,
        paper_bgcolor='rgb(47,55,79)',
        margin=dict(l=0, r=0, t=50, b=0),
        scene_camera=dict(eye=dict(x=1, y=1, z=1)),
        title=dict(
            text='Relationship Between Main Variables',
            font=dict(
                family="Century Gothic",
                color='white',
                size=responsive_font_size + 5
            ),
            x=0.47,
            y=0.95,
        )
    )

    fig2.update_traces(
        hovertemplate=(
            "Power Plants Energy: %{x}<br>"
            "Renewables: %{y}<br>"
            "Total Exchanges: %{z}<br>"
            "CO₂ Level: %{marker.color}"
        ),
        marker_size=3
    )

    fig2.update_layout(
        scene=dict(
            xaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title=dict(
                    text='Power Plants Energy',
                    font=dict(
                        color='white',
                        size=responsive_font_size + 2
                    )
                ),
                tickfont=dict(
                    color='white',
                    size=responsive_tick_font_size + 2
                )
            ),
            yaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title=dict(
                    text='Renewables',
                    font=dict(
                        color='white',
                        size=responsive_font_size + 2
                    )
                ),
                tickfont=dict(
                    color='white',
                    size=responsive_tick_font_size + 2
                )
            ),
            zaxis=dict(
                backgroundcolor="rgba(0, 0, 0,0)",
                gridcolor="white",
                showbackground=True,
                zerolinecolor="white",
                title=dict(
                    text='Total Exchanges',
                    font=dict(
                        color='white',
                        size=responsive_font_size + 2
                    )
                ),
                tickfont=dict(
                    color='white',
                    size=responsive_tick_font_size + 2
                )
            ),
            aspectratio=dict(x=0.5, y=0.5, z=0.5)
        )
    )

    fig2.update_layout(legend=dict(font=dict(family="Arial", size=responsive_font_size-5, color="white")),
                font=dict(family='Calibri', color='white'))


    ## Fig 3: the optimal resource allocation in the next 5 time points, minimizing CO2 levels

    fig3  = dash_table.DataTable(
            data = table_visual.to_dict('records'),
            columns=[{"name": col, "id": col} for col in table_visual.columns],
            style_header={
                'backgroundColor': 'rgb(47,55,79)',
                'color': '#f0c671',
                'textAlign': 'center',
                'fontFamily': 'Arial, sans-serif',
                'fontSize': '13px'
            },
            style_cell={
                'textAlign': 'center',
                'padding': '10px',
                'fontSize': '11.5px',
                'border': '1px solid lightgray',
                'backgroundColor': 'rgb(47,55,79)',
                'color': 'black',
                'whiteSpace': 'normal',  
                'height': 'auto',
                'lineHeight': '15px',
                'width': '100%',     
                'maxWidth': '100%',  
                'overflow': 'hidden',
                'fontFamily': 'Century Gothic, sans-serif'
            },
            style_data_conditional = [
                {
                    'if': {'row_index': i, 'column_id': 'Date&Time'},
                    'backgroundColor': row_colors_T[i]
                } for i in range(5)
            ] +
            [
                {
                    'if': {'row_index': i, 'column_id': col},
                    'backgroundColor': row_colors1[i]
                } for i in range(5) for col in table_visual.columns[1:-3]] + 
                [{
                    'if': {'row_index': i, 'column_id': col},
                    'backgroundColor': row_colors2[i]
                } for i in range(5) for col in table_visual.columns[-3:]],
                
            style_data = {
                'hover': {
                'backgroundColor': '#d3d3d3'}},
            style_table = {
                'position': 'relative',
                'marginLeft': '2.3%', 'marginRight': '2.3%',
                'width': '100%',
                'overflowX': 'auto',
                'maxHeight': '400px',            
                'maxWidth': '100%',
                'overflowY': 'auto'})
    
    return fig1, fig2, fig3


def update_data():
    global data, latest_distribution, next_demands, next_renewables
    global next_objective_values, next_optimized_co2, next_optimal_combinations
    global fig1, fig2, fig3, update_in_progress, update_complete

    if update_in_progress:
        return

    update_in_progress = True
    print("Background update started...")

    try:
        # Perform updates
        data = get_PowerSystem_data()
        latest_distribution, next_demands, next_renewables = latest_and_nextDR()
        next_objective_values, next_optimized_co2, next_optimal_combinations = genetic_ensemble(25, 100, 150, 8)
        fig1, fig2, fig3 = visuals()
    finally:
        update_in_progress = False
        update_complete = True
        print("Background update completed.")



#V. BUILDING THE WEB APP

app.layout = html.Div(
    style={
        "backgroundColor": "#1E1E1E",
        "color": "#D4D4D4",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "flex-start",
        "padding": "10px",
        "width": "100%",
    },

    children=[
        dcc.Interval(
            id="update-interval",
            interval=300 * 1000,  #trigger every 5 minutes
            n_intervals=0,
        ),
        html.Br(),
        html.Br(),
        dcc.Markdown(
            title,
            style={
                "color": "#D4D4D4",
                "fontSize": "1.5vw",
                "fontFamily": "Segoe UI, sans-serif",
                "textAlign": "center",
                "width": "100%"
            },
        ),
        html.Br(),
        html.Br(),

        dcc.Markdown(main_text1, mathjax=True,
                style={
                "color": "#D4D4D4",
                "fontSize": "1vw",
                "fontFamily": "Segoe UI, sans-serif",
                "textAlign": "justify",
                "width": "100%",
                "maxWidth": "900px",
                "margin": "0 auto"}
                ),

        html.Br(),

        dcc.Graph(
            id="fig1",
            figure=fig1 if fig1 else go.Figure(),
            style={
                "width": "40vw",
                "maxWidth": "800px",
                "height": "45vh",
                "margin": "10px 0",
            }
        ),

        dcc.Markdown(
            text_fig1,
            style={
                "color": "#D4D4D4",
                "fontSize": "0.8vw",
                "fontFamily": "Century Gothic, sans-serif",
                "textAlign": "center",
                "width": "40%",
                "margin": "0 auto",
            }
        ),

        dcc.Markdown(main_text2, mathjax=True,
                style={
                "color": "#D4D4D4",
                "fontSize": "1vw",
                "fontFamily": "Segoe UI, sans-serif",
                "textAlign": "justify",
                "width": "100%",
                "maxWidth": "900px",
                "margin": "0 auto"}
                ),

        html.Br(),

        dcc.Markdown(
            table_title,
            style={
                "color": "#D4D4D4",
                "fontSize": "0.95vw",
                "fontFamily": "Century Gothic, sans-serif",
                "textAlign": "center",
                "margin": "20px 0",
            }
        ),

        html.Div(id="fig3-container", children=fig3 if fig3 else html.Div("Loading..."),
                 style={
                "width": "80vw",
                "maxWidth": "1000px",
                "margin": "0 auto",
             #   "overflowX": "auto"
            }),

        dcc.Markdown(
            text_fig3,
            style={
                "color": "#D4D4D4",
                "fontSize": "0.8vw",
                "fontFamily": "Century Gothic, sans-serif",
                "textAlign": "center",
                "width": "50%",
                "margin": "20px auto",
            }
        ),

        html.Br(),

        dcc.Graph(
            id="fig2",
            figure=fig2 if fig2 else go.Figure(),
            style={
                "width": "40vw",
                "maxWidth": "700px",
                "height": "70vh",
                "margin": "10px 0",
            }
        ),

        dcc.Markdown(
            text_fig2,
            style={
                "color": "#D4D4D4",
                "fontSize": "0.8vw",
                "fontFamily": "Century Gothic, sans-serif",
                "textAlign": "center",
                "width": "40%",
                "margin": "0 auto",
            }
        ),

        html.P(
            "Last updated: Never",
            id="last-updated",
            style={"color": "white", "textAlign": "center",
                   "width": "100%", "margin": "10px 0"}
        )
    ]
)

## trigger background updates
@app.callback(
    Output("last-updated", "children"),
    [Input("update-interval", "n_intervals")],
)
def trigger_update(n_intervals):
    global update_in_progress

    if not update_in_progress:
        # Start a new thread for background updates
        threading.Thread(target=update_data).start()

## update visuals
@app.callback(
    [Output("fig1", "figure"), Output("fig2", "figure"), Output("fig3-container", "children")],
    [Input("update-interval", "n_intervals")],
)
def refresh_visuals(n_intervals):
    global fig1, fig2, fig3, update_complete

    if update_complete:
        ## reset the update_complete flag and return updated visuals
        update_complete = False
        return fig1, fig2, fig3

    ## return placeholders if updates are still in progress
    return (
        go.Figure(),  # Empty figure
        go.Figure(),  # Empty figure
        html.Div("Loading... Please wait."),
    )

This project aims to provide a solution framework for minimizing CO2 emission levels in a national power grid, through machine learning
methods. The data is collected via Energinet's public API which provides real-time data about Denmark's power system. 

Visit the Flask web app deployed on Heroku (desktop view only), which includes visuals showcasing a dynamic comparison between past hour's emissions and the minimized CO2 levels in the next 5 time points (inferred by the model presented herein), as well as the optimal energy resource distributions leading to these minimized emission levels.

The main elements of this project are: a regression tree for CO2 levels, XGBoost regressors for predicting demand and renewable energy production, genetic algorithms for finding the optimal energy distribution with respect to CO2 emission minimization. Continue reading the methodology description and check out the Python scripts for more details.

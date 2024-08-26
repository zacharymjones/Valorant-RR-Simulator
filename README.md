# Valorant RR Simulator

This is a Streamlit web application to simulate RR (Rank Rating) changes in Valorant, a popular competitive shooter game. The simulation is based on customizable parameters for winrate, RR gain, RR loss, and the number of games to be played in each simulation. The application allows users to explore possible RR outcomes, including the best and worst cases.
## Features

- **Input Parameters**: The sidebar allows you to adjust simulation parameters, such as winrate, RR gain and loss mean values, RR gain and loss minimum and maximum values, standard deviation, the number of games per simulation, and the number of simulations to run.
- **Simulation**: The app simulates RR changes for the number of games and simulations specified by the user. It uses a truncated normal distribution to model RR gain and loss.
## Visualizations

- **RR Gain and Loss Distribution**: The app displays histograms that show the probability distribution for RR gains and losses based on the provided parameters.
- **RR Simulation Results**: A plot that shows how RR fluctuates over the course of games for all simulations. Each line represents a single simulation.
- **Outcome Summaries**: The app provides a summary of the best and worst RR outcomes, the average RR gained at the end of all simulations, and the percentage of simulations that ended with negative RR.
## Parameters

Winrate: The probability of winning a game.
- **RR Gain/Loss Mean**: The average RR points gained/lost in a game.
- **RR Gain/Loss Min/Max**: The minimum and maximum possible RR gain/loss in a game.
- **RR Standard Deviation**: The standard deviation of the truncated normal distribution used for RR gain/loss.
- **Number of Games**: The number of games to simulate for each simulation.
- **Number of Simulations**: The total number of simulations to run.
## Simulation Output

- **Histograms for RR Gain and Loss**: Visual representation of the distributions for RR gains and losses based on user inputs.
- **RR Simulation Graph**: Line plot showing the progression of RR over the number of games played across all simulations.
- **Best and Worst Outcomes**: Displays the highest and lowest RR achieved in the simulations.
- **Average RR Gained**: Shows the average RR at the end of all simulations.
- **Simulations Below 0 RR**: Percentage of simulations that ended with a negative RR.

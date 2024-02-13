import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

# Set Streamlit page configuration
st.set_page_config(
    page_title="Valorant RR Simulator",
    page_icon="🎮",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Function to simulate Valorant RR
def simulate_rr(winrate, rr_gain_mean, rr_gain_min, rr_gain_max, rr_loss_mean, rr_loss_min, rr_loss_max, rr_std, num_simulations, num_games):

    rr_data = pd.DataFrame()
    best_case_rr = float('-inf')
    worst_case_rr = float('inf')
    best_case_simulation = 0
    worst_case_simulation = 0

    for i in range(num_simulations):
        rr = 0
        rr_values = []

        for j in range(num_games):
            outcome = np.random.choice(["win", "loss"], p=[winrate, 1 - winrate])

            if outcome == "win":
                # Use a truncated normal distribution for RR gain
                a, b = (rr_gain_min - rr_gain_mean) / rr_std, (rr_gain_max - rr_gain_mean) / rr_std
                rr_gain = max(0, truncnorm.rvs(a, b, loc=rr_gain_mean, scale=rr_std))
                rr += rr_gain
            else:
                # Use a truncated normal distribution for RR loss
                a, b = (rr_loss_min - rr_loss_mean) / rr_std, (rr_loss_max - rr_loss_mean) / rr_std
                rr_loss = max(0, truncnorm.rvs(a, b, loc=rr_loss_mean, scale=rr_std))
                rr -= rr_loss

            rr_values.append(rr)

        rr_data = pd.concat([rr_data, pd.DataFrame({"Game": range(1, num_games + 1),
                                                    "Simulation": [i + 1] * num_games,
                                                    "RR": rr_values})], ignore_index=True)

        # Update best and worst case RR outcomes
        if max(rr_values) > best_case_rr:
            best_case_rr = max(rr_values)
            best_case_simulation = i + 1

        if min(rr_values) < worst_case_rr:
            worst_case_rr = min(rr_values)
            worst_case_simulation = i + 1

    return rr_data, best_case_rr, worst_case_rr

# Function to add average line to the bell curve plot
def add_average_line(ax, x_values, y_values, average, color):
    ax.plot(x_values, y_values, label="Distribution", color=color, alpha=0.7)
    ax.axvline(average, color="black", linestyle="--", label="Average", linewidth=2)
    ax.legend()

# Streamlit app
st.title("Valorant RR Simulator")

# Sidebar for input parameters
st.sidebar.header("Input Parameters")

# Game Outcome Parameters
winrate = st.sidebar.slider("Winrate (%)", 10, 90, 55, 1, format="%d%%")

# RR Gain Parameters
st.sidebar.subheader("RR Gain Parameters")
rr_gain_mean = st.sidebar.slider("Mean (RR Gain)", 1, 50, 17, 1)
rr_gain_min = st.sidebar.slider("Min (RR Gain)", 1, 50, 1, 1)
rr_gain_max = st.sidebar.slider("Max (RR Gain)", 1, 50, 30, 1)

# RR Loss Parameters
st.sidebar.subheader("RR Loss Parameters")
rr_loss_mean = st.sidebar.slider("Mean (RR Loss)", 1, 50, 17, 1)
rr_loss_min = st.sidebar.slider("Min (RR Loss)", 1, 50, 1, 1)
rr_loss_max = st.sidebar.slider("Max (RR Loss)", 1, 50, 30, 1)

# RR Standard Deviation
st.sidebar.subheader("RR Standard Deviation")
rr_std = st.sidebar.slider("Standard Deviation", 1, 10, 3, 1)

# Number of Games and Simulations
st.sidebar.subheader("Simulation Settings")
num_games = st.sidebar.slider("Number of Games", 10, 500, 100, 10)
num_simulations = st.sidebar.slider("Number of Simulations", 10, 1000, 100, 10)

st.sidebar.image("gentlesquare.jpeg", use_column_width=True)
st.sidebar.write('Created by [Windfall]("https://www.youtube.com/@windfallval")')
# Simulate RR and get the best and worst outcomes
rr_simulations, best_case_rr, worst_case_rr = simulate_rr(winrate / 100, rr_gain_mean, rr_gain_min, rr_gain_max,
                                                           rr_loss_mean, rr_loss_min, rr_loss_max, rr_std, num_simulations, num_games)

# Plot histogram for RR gain distribution
fig_gain, ax_gain = plt.subplots(figsize=(10, 6))
ax_gain.set_title("RR Gain Distribution", color='black')
ax_gain.set_xlabel("RR Gain", color='black')
ax_gain.set_ylabel("Probability Density", color='black')
x_gain = np.linspace(rr_gain_min, rr_gain_max, 100)
a_gain, b_gain = (rr_gain_min - rr_gain_mean) / rr_std, (rr_gain_max - rr_gain_mean) / rr_std
rr_gain_curve = truncnorm.pdf(x_gain, a_gain, b_gain, loc=rr_gain_mean, scale=rr_std)
rr_gain_curve = rr_gain_curve[rr_gain_curve > 0]
ax_gain.hist(x_gain, bins=30, weights=rr_gain_curve, density=True, label="RR Gain Distribution", color="green", alpha=0.7)
ax_gain.legend()

# Plot histogram for RR loss distribution
fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
ax_loss.set_title("RR Loss Distribution", color='black')
ax_loss.set_xlabel("RR Loss", color='black')
ax_loss.set_ylabel("Probability Density", color='black')
x_loss = np.linspace(rr_loss_min, rr_loss_max, 100)
a_loss, b_loss = (rr_loss_min - rr_loss_mean) / rr_std, (rr_loss_max - rr_loss_mean) / rr_std
rr_loss_curve = truncnorm.pdf(x_loss, a_loss, b_loss, loc=rr_loss_mean, scale=rr_std)
rr_loss_curve = rr_loss_curve[rr_loss_curve > 0]
ax_loss.hist(x_loss, bins=30, weights=rr_loss_curve, density=True, label="RR Loss Distribution", color="red", alpha=0.7)
ax_loss.legend()

# Plot the results without legend
fig_simulator, ax_simulator = plt.subplots(figsize=(10, 6))
for i in range(num_simulations):
    ax_simulator.plot(rr_simulations[rr_simulations["Simulation"] == i + 1]["Game"],
                      rr_simulations[rr_simulations["Simulation"] == i + 1]["RR"],
                      label=f"Simulation {i + 1}")

ax_simulator.set_title("Valorant RR Simulator", color='black')
ax_simulator.set_xlabel("Game", color='black')
ax_simulator.set_ylabel("RR Points", color='black')

# Show the plots
st.pyplot(fig_gain)
st.pyplot(fig_loss)
st.pyplot(fig_simulator)


# Count the number of lines below 0 at the end of the simulation
lines_below_zero = np.sum(rr_simulations[rr_simulations["Game"] == num_games]["RR"] < 0)

# Calculate and display the average RR gained
average_rr_gained = rr_simulations[rr_simulations["Game"] == num_games]["RR"].mean()
st.subheader(f"Number of simulations below 0 RR: {int(round(lines_below_zero/num_simulations*100, 0))}%")
st.subheader(f"Average RR gained at the end of the simulation: {int(round(average_rr_gained))}")


# Display best and worst RR outcomes
st.subheader(f"Best RR outcome: {int(round(best_case_rr))}")
st.subheader(f"Worst RR outcome: {int(round(worst_case_rr))}")
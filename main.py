from pyautogui import press, typewrite, hotkey 
import pyautogui as agu
import time
import matplotlib.pyplot as plt
import numpy as np

import functions
initial_state = functions.Get_current_state()

actions = ["up", "down", "left", "right", "space"]
number_of_actions = len(actions)

# Hyperparameters: 
arcitecture = [initial_state.shape[0], 256, 128, 64, number_of_actions]


# Initializing:
L = len(arcitecture)
params = functions.Get_initial_params(arcitecture)


# Training:
timer = 0
time_limit = 120    # seconds

while timer < time_limit:
    state = functions.Get_current_state()
    output = functions.Forward(state, params, L)
    action_commands = functions.Get_action_commands(output, actions)

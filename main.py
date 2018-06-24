from pyautogui import press, typewrite, hotkey 
import pyautogui as agu
import time
import matplotlib.pyplot as plt
import numpy as np

import functions
from sys import exit as ex
#time.sleep(3)

pos = agu.locateOnScreen("active.png")
if pos == None:
    pos = agu.locateOnScreen("unactive.png")
    if pos == None:
        pos = agu.locateOnScreen("gameover.png")

    if pos == None:
        ex("Can't find Tetris window, try moving window away from edge. Aborting.")
    agu.moveTo(pos[0]-30,pos[1]+5)
    agu.click()
pos = agu.locateOnScreen("active.png")
game_region =(pos[0]-128,pos[1]+30, 365,365)

# initial_state = agu.screenshot(region=game_region)

# img = np.array(initial_state)
# plt.imshow(img)
# plt.show()



initial_state = functions.Get_current_state(game_region)

#ex("asd")

# history path /home/kodemannen/snap/quadrapassel/44/.local/share/quadrapassel/history



actions = ["nothing", "up", "down", "left", "right", "space"]
#actions = ["nothing", "up", "down", "left", "right"]
number_of_actions = len(actions)

# Hyperparameters: 
arcitecture = [initial_state.shape[0], 1024, 64, number_of_actions]
change_limit = 5    # if the state is unchanged in 5 frames, we assume the game is over

# Initializing:
print("Initializing..")
L = len(arcitecture)
params = functions.Get_initial_params(arcitecture)
print("Initializing complete.")
print(" ")


# Training:
timer = 0
time_limit = 120    # seconds
time.sleep(2)
print("Starting new game")
hotkey("ctrl", "n")
time.sleep(1.5)
print("Playing..")

change_counter = 0
old_state = initial_state
game_over = False
while game_over == False:
    state = functions.Get_current_state(game_region)
    output = functions.Forward(state, params, L)
    #action_commands = functions.Get_multiple_action_commands(output, actions)
    action_commands, index = functions.Get_single_action_command(output, actions)

    press(action_commands)
    print(action_commands)
    #print(action_commands)
    #time.sleep(0.5)
    
    change_counter += 1
    if change_counter == 5:
        if np.linalg.norm(state-old_state) <= 1e-12:
            game_over = True
            print("Game over")
        else:
            change_counter = 0
            old_state = state
    
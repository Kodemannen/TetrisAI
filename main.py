from pyautogui import press, typewrite, hotkey 
import pyautogui as agu
import time
import matplotlib.pyplot as plt
import numpy as np
import functions
from sys import exit as ex


#########################
# Resetting score file: #
#########################
score_file_path = "/home/kodemannen/snap/quadrapassel/44/.local/share/quadrapassel/history"
score_file = open(score_file_path, "w")
score_file.truncate()
score_file.close()



#############################
# LOCATING GAME WINDOW ETC: #
#############################
#pos = agu.locateOnScreen("Images/active.png")
#if pos == None:
pos = agu.locateOnScreen("Images/unactive.png")
if pos == None:
    pos = agu.locateOnScreen("Images/gameover.png")
if pos == None:
    pos = agu.locateOnScreen("Images/closetoedge.png")
if pos == None:
    ex("Can't find Tetris window, try moving window away from edge. Aborting.")
agu.moveTo(pos[0]-30,pos[1]+5)
agu.click()
pos = agu.locateOnScreen("Images/active.png")
game_region =(pos[0]-118,pos[1]+55, 340,318)
initial_state = functions.Get_current_state(game_region)


# img = agu.screenshot(region=game_region)
# plt.imshow(np.array(img))
# plt.show()
# ex("saddf")


#############################
# Hyperparameters and shit: #
#############################
actions = ["nothing", "up", "down", "left", "right", "space"]

number_of_actions = len(actions)
architecture = [initial_state.shape[0], 256, number_of_actions]
L = len(architecture)
change_limit = 3    # if the state is unchanged in 5 frames, we assume the game is over
loss_score = -10    # for when the score is 0
games_per_episode = 1
learning_rate = 0.01


#################
# Initializing: #
#################
print("Initializing neural network..")
params = functions.Get_initial_params(architecture, L)
print("Initialization complete.")
print(" ")

episode_count = 0
training = True
game_count = 0
best_score = 0
while training:
    max_score = 0
    #####################
    # Starting episode: #
    #####################
    episode_count += 1

    for game_index in range(games_per_episode):
        game_count += 1
        print("Training game number " + str(game_count), ", episode " + str(episode_count))
        ###################
        # Playing a game: #
        ###################
        print("Starting new game")
        time.sleep(1)
        hotkey("ctrl", "n")
        time.sleep(1.5)
        print("Playing..")

        activations_list = []
        actions_list = []

        change_counter = 0
        old_state = initial_state
        game_over = False
        while game_over == False:
            state = functions.Get_current_state(game_region)
            output, activations = functions.Forward(state, params, L)
            #action_commands = functions.Get_multiple_action_commands(output, actions)
            action_commands, action_index = functions.Get_single_action_command(output, actions)
            press(action_commands)

            
            #################################################################
            # Checking if game has stopped by seeing if state is unchanged: #
            #################################################################
            # This method might possibly cause false positives, but not sure how likely it is
            change_counter += 1
            if change_counter == 5:
                
                if np.linalg.norm(state-old_state) <= 1e-12:
                    game_over = True
                    print("Game over")
                else:
                    change_counter = 0
                    old_state = state
                
            activations_list.append(activations)
            actions_list.append(action_index)

            time.sleep(0.1)

        ##################
        # Getting score: #
        ##################
        score_file_path = "/home/kodemannen/snap/quadrapassel/44/.local/share/quadrapassel/history"
        score_file = open(score_file_path, "r")
        line = score_file.readline().split()
        if len(line) == 0:
            score = loss_score
        else:
            score = np.log(float(line[-1]))

        print(score)
        score_file.close()
        # Emptying score file:
        score_file = open(score_file_path, "w")
        score_file.truncate()
        score_file.close()
        print("Score: " + str(score))

        #########################################
        # Saving config if score is new record: #
        #########################################
        if score > best_score:
            np.save("/home/kodemannen/TetrisGameStates/Best_config%s.npy"%game_count, params)

        ############################################
        # Saving state/actions pairs for the game: #
        ############################################

        actions_list.append(score)
        
        activations_filename = "/home/kodemannen/TetrisGameStates/activations_game%s" % game_index
        actions_filename = "/home/kodemannen/TetrisGameStates/actions_game%s" % game_index

        np.save(activations_filename, activations_list)
        np.save(actions_filename, actions_list)


    ####################
    # Backpropagation: #
    ####################
    for game_index in range(games_per_episode):

        activations_filename = "/home/kodemannen/TetrisGameStates/activations_game%s.npy" % game_index
        actions_filename = "/home/kodemannen/TetrisGameStates/actions_game%s.npy" % game_index

        activations_list = np.load(activations_filename)
        actions_list = np.load(actions_filename)

        # print(activations_list.shape)
        # print(activations_list[0].shape)
        # print(activations_list[0][0].shape)

        Ttime = len(actions_list) - 1   # -1 since last element is the score for the game
        score = actions_list[-1]
        R = np.log(score) if score > 0 else score
        R = learning_rate*R

        gradients = functions.Get_zero_gradient(architecture, L)

        for t in range(Ttime):
            #state_t = activations_list[t][0]
            activations = activations_list[t]
            action_index = actions_list[t]

            functions.Update_gradients(gradients, params, activations, L, action_index)

        functions.Update_params(gradients, params, L, R)

        ##############
        # VECTORIZE: #
        ##############


    print("Successful backprop")


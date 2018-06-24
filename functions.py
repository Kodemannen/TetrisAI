#from pyautogui import press, typewrite, hotkey 
import pyautogui as agu
import time
import matplotlib.pyplot as plt
import numpy as np
#img = agu.screenshot(region=(65,62,367,365)) #region(x,y,width,height)
#npimg = np.array(img)
#print(npimg.shape)
#plt.imshow(npimg)
#plt.show()

def Get_current_state(region):
    # Gets image of tetris game, returns as column vector
    state = agu.screenshot(region=region) #region(x,y,width,height)
    numpystate = np.array(state)
    shape = numpystate.shape

    columnstate = numpystate.reshape(shape[0]*shape[1]*shape[2]) / 255.
    return columnstate

def Policy_function(state):
    # Function that returns an action based on the current state 
    action = None
    return action

def ReLu(vec):
    a = vec*(vec>0)
    return a

def Softmax(vec):
    exped = np.exp(vec)
    softmaxed = exped / np.sum(exped)
    return softmaxed
    
def Get_multiple_action_commands(vec, actions):
    # Samples multiple action commands
    ########## NOT PROPERLY IMPLEMENTED FOR BACKPROP ############
    commands = []
    for i in range(len(vec)):
        #if vec[i] < threshold:
        prob = vec[i]
        dice = np.random.randn()
        if dice <= prob:
            commands.append(actions[i])
    return commands

def Get_single_action_command(vec, actions): 
    # Samples a single action command
    index = np.random.choice(len(vec), p=vec)
    command = actions[index]
    return command, index

def Get_initial_params(arcitecture):
    params = {}
    for l in range(len(arcitecture)-1):
        n = arcitecture[l]
        params["w%s" % l] = np.random.randn(arcitecture[l+1],n) / np.sqrt(n)    # weight matrix of layer l
        params["b%s" % l] = np.zeros(shape=(arcitecture[l+1]))               # bias vector of layer l
    return params

def Forward(state, params, L):
    a = state
    for l in range(L-1):
        w = params["w%s" % l]
        b = params["b%s" % l]
        y = np.dot(w, a) + b
        a = ReLu(y) if  l != L-2 else Softmax(y)
    return a


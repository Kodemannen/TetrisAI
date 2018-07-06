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
    #a = vec*(vec>0)
    #return a
    vec[vec<0] = 0
    return vec

def Softmax(vec):
    exped = np.exp(vec-np.max(vec))
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

def Get_initial_params(architecture, L):
    params = {}
    for l in range(1, L):
        n = architecture[l-1]
        params["w%s" % l] = np.random.randn(n, architecture[l]) / np.sqrt(n)    # weight matrix of layer l
        params["b%s" % l] = np.zeros(shape=(architecture[l]))               # bias vector of layer l
    return params


def Forward(state, params, L):
    
    a = state
    activations = [a]
    for l in range(1, L):
        w = params["w%s" % l]
        b = params["b%s" % l]
        z = np.dot(w.T, a) + b
        a = ReLu(z) if  l != L-1 else Softmax(z)
        activations.append(a)
    return a, activations


def Get_zero_gradient(architecture, L):
    grads = {}
    for l in range(1, L):
        n = architecture[l-1]
        grads["w%s" % l] = np.random.randn(n, architecture[l]) / np.sqrt(n)    # weight matrix of layer l
        grads["b%s" % l] = np.zeros(shape=(architecture[l]))               # bias vector of layer l
    return grads

def Update_gradients(gradients, params, activations, L, action_index):
    
    y = activations[-1]
    y_targ = np.zeros(len(y))
    y_targ[action_index] = 1

    for l in reversed(range(1,L)):

        if l==(L-1):
            grad_z = y_targ - y
        else:
            w_lp1 = params["w%s"%(l+1)]      
            grad_z = (activations[l]>0)*np.dot(w_lp1, grad_z)
            
        gradients["b%s"%l] += grad_z
        gradients["w%s"%l] += np.outer(activations[l-1], grad_z.T)
    return None

def Update_multiple_gradients(gradients, params, activations, L, action_indices):
    Ttot = len(activations)

    y = np.array(activations[:,-1])
    print(y)
    n = len(y[0])
    #print(n, len(y))
    y_targ = np.zeros((len(y), n))
    #print(np.shape(y_targ))

    y_targ[range(Ttot), action_indices] = 1
    print("yes")
    for l in reversed(range(1,L)):

        if l==(L-1):
            grad_z = y_targ - y
        else:
            w_lp1 = params["w%s"%(l+1)]      
            grad_z = (activations[l]>0)*np.dot(w_lp1, grad_z)
            
        gradients["b%s"%l] += grad_z
        gradients["w%s"%l] += np.outer(activations[l-1], grad_z.T)
    return None


def Update_params(gradients, params, L, R):

    for l in range(1,L):
        params["w%s"%l] -= R*gradients["w%s"%l]
        params["b%s"%l] -= R*gradients["b%s"%l] 
    return None
import numpy as np
import pandas as pd

from pygmo.core import hypervolume
import matplotlib.pyplot as pltorc

import torch
import torch.nn as nn

def reset_model(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
            
def weights_init(m):
    if isinstance(m, nn.Conv3d or nn.Conv1d): #nn.Conv3d
        torch.nn.init.xavier_uniform_(m.weight.data, init.calculate_gain('relu'))
        m.bias.data.fill_(0)
        # torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

def to_np(scores):
    """Convert the scores as output by the pareto manager to numpy
    to be able to use for finding the pareto front and plotting
    """
    scores = [i[0] for i in scores]
    scores = np.array(scores)
    return(scores)

def identify_pareto(scores):
    """For n pareto points in d dimensions, 'scores' is a numpy array
    with shape (n, d). Returns the indices of the points which are part of 
    the pareto front
    """
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]

def get_pareto_points(scores):
    """Get pareto points in a form convenient for plotting
    """
    x = [0 for i in range(scores.shape[1]) ]
    for i in range(scores.shape[1]):
        x[i] = scores[:,i]
    
    pareto = identify_pareto(scores)
    pareto_front = scores[pareto]
    
    pareto_front_df = pd.DataFrame(pareto_front)
    pareto_front_df.sort_values(0, inplace=True)
    pareto_front = pareto_front_df.values
    
    x_pareto = [0 for i in range(pareto_front.shape[1])]
    for i in range(pareto_front.shape[1]):
        x_pareto[i] = pareto_front[:,i]
    
    return(x_pareto[0], x_pareto[1])



my_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
def plot_paretos(score_list, names, axes_labels, colors=my_colors):
    pareto = []
    for score in score_list:
        pareto.append(get_pareto_points(score))

    for i, p in enumerate(pareto):
        rgb = np.random.rand(3,)
        plt.plot(p[0], p[1], c=colors[i], label=names[i])
        plt.scatter(p[0], p[1], c=colors[i])


    plt.xlabel(axes_labels[0])
    plt.ylabel(axes_labels[1])
    plt.legend()
    plt.show()
    
def plot_2d_pareto(scores):
    pareto = get_pareto_points(scores)
    plt.plot(pareto[0], pareto[1])
    plt.scatter(pareto[0], pareto[1])
    
    
    
    
def get_hypervolume(scores):
    scores = -scores
    hv = hypervolume(scores)
    d = scores.shape[1]
    return hv.compute([0.0]*d)

def is_dominated(p, S):
    for s in S:
        if(np.all(s>=p)):
            return True
    return False

def coverage(S1, S2):
    count = 0
    for s in S2:
        if(is_dominated(s, S1)):
            count = count + 1
    return(count/len(S2))

def distance_to_closest_neighbor(s, scores):
    min_distance = 2*scores.shape[1]
    for s_j in scores:
        distance = np.sum(abs(s_j - s))
        if(distance < min_distance):
            min_distance = distance
    return(min_distance)
            
    
def spacing(scores):
    distances = []
    for i, s in enumerate(scores):
        d_i = distance_to_closest_neighbor(s, np.delete(scores, i, axis=0))
        distances.append(d_i)
    distances = np.array(distances)
    d_mean = np.mean(distances)
    
    total=0
    for d in distances:
        total+=((d-d_mean)*(d-d_mean))
    return np.sqrt(total/float(len(scores) - 1))

def get_solution(scores, norm='l2', ideal_point_id='zenith'):
    d = scores.shape[1]
    if(ideal_point_id == 'zenith'):        
        ip = np.ones(d)
    elif(isinstance(ideal_point_id, int)):
        ip = np.zeros(d)
        ip[ideal_point_id] = 1
        
    d_min = 10*d
    s_min = np.zeros(d)
    for i, s in enumerate(scores):
        if(norm=='l2'):
            d_s = np.sum((ip - s)**2)
        elif(norm=='l1'):
            d_s = np.sum(np.abs(ip - s))
        if(d_s<d_min):
            d_min = d_s
            s_min = s
            i_min = i
    return s_min, i_min
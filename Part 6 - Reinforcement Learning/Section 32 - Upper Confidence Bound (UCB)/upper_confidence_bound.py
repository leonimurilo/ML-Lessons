# Upper Confidence Bound

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset]
# the dataset contains boolean cells that indicate whether an ad was clicked or not
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Implementing UCB
import math

N = 10000 # get number of rows
d = dataset.shape[1] # get the number of columns

ads_selected = [] # vector used by the histogram
numbers_of_selections = [0] * d # stores how many times each item was chosen by the algorithm (ex: how many times a specific lever was pulled or how many time an ad was picked)
sums_of_rewards = [0] * d # stores how many times the selection was sucessful (tip: remember of on-line learning)
total_reward = 0 # counter of obtained rewards

for n in range(0, N):
    selected_ad = 0
    max_upper_bound = 0

    for i in range(0, d):
        # if ad "i" was chosen at least one time
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            # this forces the ad to be chosen in the first rounds
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            selected_ad = i

    ads_selected.append(selected_ad) # add selected ad to histogram vector
    numbers_of_selections[selected_ad] = numbers_of_selections[selected_ad] + 1 # compute algorithm selection for the specific ad
    reward = dataset.values[n, selected_ad] # "check" if the ad was really selected in the "real life" dataset
    sums_of_rewards[selected_ad] = sums_of_rewards[selected_ad] + reward # compute the reward for the selected ad (0 or 1)
    total_reward = total_reward + reward

# Visualising the results
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()

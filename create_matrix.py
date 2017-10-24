########################################################################
# Copyright (C) 2017  Seyed Mehran Kazemi, Licensed under the GPL V3;  #
# see: <https://www.gnu.org/licenses/gpl-3.0.en.html>                  #
########################################################################

import numpy as np
import pickle

user_ratings = {}
gender = {} 
rating_cutoff, test_cutoff = 974687810, 967587781

def extract_cols(lst,indexes):
    return (lst[i] for i in indexes)

with open("ml-1m/ratings.dat",'r') as ratingsfile:
    all_ratings = (tuple(int(e) for e in extract_cols(line.strip().split(':'),[0,2,4,6])) for line in ratingsfile)
    ratings = [eg for eg in all_ratings if eg[3] <= rating_cutoff]
    all_users = {u for (u,i,r,d) in ratings}
    all_items = {i for (u,i,r,d) in ratings}
    training_users = {u for (u,i,r,d) in ratings if d <= test_cutoff}
    test_users = all_users - training_users

    # extract the training and test dictionaries
    with open("ml-1m/users.dat",'r') as usersfile: user_info = (extract_cols(line.strip().split(':'),[0,4,2,6,8]) for line in usersfile)

    for (u,a,g,o,p) in user_info:
        user_ratings[int(u)] = []
        if int(u) in training_users or int(u) in test_users:
            gender[int(u)] = g

    for (u,i,r,d) in ratings:
        user_ratings[u].append(i)

training_users = list(training_users)
test_users = list(test_users)

train_x = np.zeros((len(training_users), len(all_items)))
train_y = np.zeros((len(training_users), 1))

for i, user in enumerate(training_users):
    if gender[user] == "M":
        train_y[i][0] = 1.0
    for j, item in enumerate(all_items):
        if item in user_ratings[user]:
            train_x[i][j] = 1

test_x = np.zeros((len(test_users), len(all_items)))
test_y = np.zeros((len(test_users), 1))
for i, user in enumerate(test_users):
    if(gender[user] == "M"):
        test_y[i][0] = 1.0
    for j, item in enumerate(all_items):
        if item in user_ratings[user]:
            test_x[i][j] = 1

with open('ml{}_matrices.pickle'.format(datasetname),'wb') as f:
    pickle.dump([train_x, train_y, test_x, test_y], f)


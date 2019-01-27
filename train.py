from keras.models import load_model
import os
import utils
import random
import numpy as np

def train(model, X, Y, test_split = 0.1, epochs = 20, learning_rate = 10**(-5), batch_size = 8, savepath= None, save_only_best = True):
    if type(model)==str and os.path.isfile(model):
        model = load_model(model)
    if type(X)==str and os.path.isdir(X) and type(Y)==str and os.path.isdir(Y):
        X, Y = utils.get_training_set(X, Y)
    
    random_idxes = [np.random.choice(np.array(range(len(X))), replcae = False) for _ in range(int(len(X)*test_split))]
    X_test, Y_test = X[random_idxes], Y[random_idxes]
    X_train = np.delete(X, random_idxes, axis = 0)
    Y_train = np.delete(X, random_idxes, axis = 0)

    
    model.fit(X_train, Y_train, validation_data = (X_test, Y_test), shuffle = True, epochs = epochs, batch_size = batch_size)

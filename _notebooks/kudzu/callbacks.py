class Callback():
    def __init__(self, learner):
        self.learner = learner
    def fit_start(self):
        return True
    def fit_end(self):
        return True
    def epoch_start(self, epoch):
        return True
    def batch_start(self, batch):
        return True
    def after_loss(self, loss):
        return True
    def batch_end(self):
        return True
    def epoch_end(self):
        return True
    
from collections import defaultdict
import numpy as np

def take_mean(data, bpe, afrac):
    if afrac== 0.:
        return np.mean(data)
    else:
        mean_but_last = np.mean(data[:-1])
        #return (1./bpe)*np.mean(data[-1]) + (1. - 1./bpe)*mean_but_last
        return (afrac*data[-1] + (bpe -1)*mean_but_last)/(bpe - 1 + afrac)
        
    
class AccCallback(Callback):
    def __init__(self, learner, bs):
        super().__init__(learner)
        self.bs = bs
        self.losses = []
        self.batch_losses = []
        self.paramhist = defaultdict(list)
        self.gradhist = defaultdict(list)
        self.bpe = 0
        self.afrac=0.
        
    def get_weights(self, layer, index):
        return np.array([wmat[index][0] for wmat in self.paramhist[layer+'_w']])
    def get_weightgrads(self, layer, index):
        return np.array([wmat[index][0] for wmat in self.gradhist[layer+'_w']])
    def get_biases(self, layer):
        return np.array([e[0] for e in self.paramhist[layer+'_b']])
    def get_biasgrads(self, layer):
        return np.array([e[0] for e in self.gradhist[layer+'_b']])
    def fit_start(self):
        self.bpe = self.learner.bpe
        self.afrac = self.learner.afrac
        return True
    def fit_end(self):
        return True
    def epoch_start(self, epoch):
        self.epoch = epoch
        #print("EPOCH {}".format(self.epoch))
        return True
    def batch_start(self, batch):
        self.batch = batch
    def after_loss(self, loss):
        self.loss = loss
        #print("loss", self.epoch, self.loss)
        return True
    def batch_end(self):
        self.batch_losses.append(self.loss)
    def epoch_end(self):
        for layer, name, fnval, grval in self.learner.model.params_and_grads():
            self.paramhist[layer.name+'_'+name].append(fnval)
            self.gradhist[layer.name+'_'+name].append(grval)
        eloss = take_mean(self.batch_losses[-self.bpe:], self.bpe, self.afrac)
        self.losses.append(eloss)
        if self.epoch % 10 ==0:
            print(f"Epoch {self.epoch} Loss {eloss}")

        return True
    
class ClfCallback(Callback):
    def __init__(self, learner, bs, xtrain, xtest,ytrain,ytest):
        super().__init__(learner)
        self.accuracies = []
        self.test_accuracies = []
        self.bs = bs
        self.losses = []
        self.batch_losses = []
        self.paramhist = defaultdict(list)
        self.gradhist = defaultdict(list)
        self.bpe = 0
        self.afrac=0.
        self.xtrain = xtrain
        self.xtest = xtest
        self.ytrain = ytrain
        self.ytest = ytest
        
        
    def get_weights(self, layer, index):
        return np.array([wmat[index][0] for wmat in self.paramhist[layer+'_w']])
    def get_weightgrads(self, layer, index):
        return np.array([wmat[index][0] for wmat in self.gradhist[layer+'_w']])
    def get_biases(self, layer):
        return np.array([e[0] for e in self.paramhist[layer+'_b']])
    def get_biasgrads(self, layer):
        return np.array([e[0] for e in self.gradhist[layer+'_b']])
    def fit_start(self):
        self.bpe = self.learner.bpe
        self.afrac = self.learner.afrac
        return True
    def fit_end(self):
        return True
    def epoch_start(self, epoch):
        self.epoch = epoch
        #print("EPOCH {}".format(self.epoch))
        return True
    def batch_start(self, batch):
        self.batch = batch
    def after_loss(self, loss):
        self.loss = loss
        #print("loss", self.epoch, self.loss)
        return True
    def batch_end(self):
        self.batch_losses.append(self.loss)
    def epoch_end(self):
        for layer, name, fnval, grval in self.learner.model.params_and_grads():
            self.paramhist[layer.name+'_'+name].append(fnval)
            self.gradhist[layer.name+'_'+name].append(grval)
        eloss = take_mean(self.batch_losses[-self.bpe:], self.bpe, self.afrac)
        self.losses.append(eloss)
        
        self.prob_ytrain = self.learner.model(self.xtrain)
        self.pred_ytrain = 1*(self.prob_ytrain > 0.5)
        
        train_accuracy = np.mean(self.pred_ytrain == self.ytrain)
        self.accuracies.append(train_accuracy)
        
        self.prob_ytest = self.learner.model(self.xtest)
        self.pred_ytest = 1*(self.prob_ytest > 0.5)
        
        test_accuracy = np.mean(self.pred_ytest == self.ytest)
        self.test_accuracies.append(test_accuracy)
        
        
        if self.epoch % 10 ==0:
            print(f"Epoch {self.epoch} Loss {eloss}")
            print(f"train accuracy {train_accuracy}, test accuracy {test_accuracy}")
     

        return True
    
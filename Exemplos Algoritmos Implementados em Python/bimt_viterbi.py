# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 21:41:27 2020

@author: Geraldo
"""
import numpy as np

'''
N: number of hidden states
'''

A =[
[0.3777, 0.0110, 0.0009, 0.0084, 0.0584, 0.0090, 0.0025],
[0.0008, 0.0002, 0.7968, 0.0005, 0.0008, 0.1698, 0.0041],
[0.0322, 0.0005, 0.0050, 0.0837, 0.0615, 0.0514, 0.2231],
[0.0366, 0.0004, 0.0001, 0.0733, 0.4509, 0.0036, 0.0036],
[0.0096, 0.0176, 0.0014, 0.0086, 0.1216, 0.0177, 0.0068],
[0.0068, 0.0102, 0.1011, 0.1012, 0.0120, 0.0728, 0.0479],
[0.1147, 0.0021, 0.0002, 0.2157, 0.4744, 0.0102, 0.0017]
]

pi = [[ .2767,.006,.0031,.0453,.0449,.0510,.2026]]

B = [
 [ 0.000032, 0, 0, 0.000048, 0],
[ 0, 0.308431, 0, 0, 0],
[ 0, 0.000028, 0.000672, 0, 0.000028],
[ 0, 0, 0.000340, 0, 0],
[ 0, 0.000200, 0.000223, 0, 0.002337],
[ 0, 0, 0.010446, 0, 0],
[ 0, 0, 0, 0.506099, 0]
]

Anp = np.array(A)
pinp = np.array(pi).T
Bnp = np.array(B)

data = [0,1,2,3,4]

class Decoder(object):
    def __init__(self, initialProb, transProb, obsProb):
        self.N = initialProb.shape[0]
        self.initialProb = initialProb
        self.transProb = transProb
        self.obsProb = obsProb
        assert self.initialProb.shape == (self.N, 1)
        assert self.transProb.shape == (self.N, self.N)
        assert self.obsProb.shape[0] == self.N

    def Obs(self, obs):
        return self.obsProb[:, obs, None]

    def Decode(self, obs):
        trellis = np.zeros((self.N, len(obs)))
        backpt = np.ones((self.N, len(obs)), 'int32') * -1

        # initialization
        trellis[:, 0] = np.squeeze(self.initialProb * self.Obs(obs[0]))

        for t in range(1, len(obs)):
            trellis[:, t] = (trellis[:, t-1, None].dot(self.Obs(obs[t]).T) * self.transProb).max(0)
            backpt[:, t] = (np.tile(trellis[:, t-1, None], [1, self.N]) * self.transProb).argmax(0)
        # termination
        tokens = [trellis[:, -1].argmax()]
        for i in range(len(obs)-1, 0, -1):
            tokens.append(backpt[tokens[-1], i])
        return tokens[::-1]

d = Decoder(pinp,Anp,Bnp)
x = d.Decode(data)
print(x)

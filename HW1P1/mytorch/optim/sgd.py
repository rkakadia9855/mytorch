import numpy as np


class SGD:

    def __init__(self, model, lr=0.1, momentum=0):

        self.l = model.layers
        self.L = len(model.layers)
        self.lr = lr
        self.mu = momentum
        self.v_W = [np.zeros(self.l[i].W.shape, dtype="f")
                    for i in range(self.L)]
        self.v_b = [np.zeros(self.l[i].b.shape, dtype="f")
                    for i in range(self.L)]

    def step(self):
        
        """
        step: Updates W and b of each of the model layers:
        ∗ Because parameter gradients tell us which direction makes the model worse, we move opposite
        the direction of the gradient to update parameters.
        ∗ When momentum is non-zero, update velocities v W and v b, which are changes in the gradient
        to get to the global minima. The velocity of the previous update is scaled by hyperparameter
        µ, refer to lecture slides for more details.
        """

        for i in range(self.L):

            if self.mu == 0:

                self.l[i].W = self.l[i].W - (self.lr * self.l[i].dLdW) # TODO
                self.l[i].b = self.l[i].b - (self.lr * self.l[i].dLdb)  # TODO

            else:
                self.v_W[i] = [[x*self.mu for x in y] for y in self.v_W[0]] + (self.l[i].dLdW)  # TODO
                self.v_b[i] = [[x*self.mu for x in y] for y in self.v_b[0]] + (self.l[i].dLdb)  # TODO
                self.l[i].W = self.l[i].W -  ([[x*self.lr for x in y] for y in self.v_W[0]]) # TODO
                self.l[i].b = self.l[i].b -  ([[x*self.lr for x in y] for y in self.v_b[0]])  # TODO

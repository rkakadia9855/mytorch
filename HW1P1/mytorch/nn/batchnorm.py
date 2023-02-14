import numpy as np


class BatchNorm1d:

    """
– forward: It takes in a batch of data Z computes the batch normalized data Zˆ, and returns the
scaled and shifted data Z˜. In addition:
∗ During training, forward calculates the mean and standard-deviation of each feature over the
mini-batches and uses them to update the running M E[Z] and running V V ar[Z], which are
learnable parameter vectors trained during forward propagation. By default, the elements of
E[Z] are set to 1 and the elements of V ar[Z] are set to 0.
∗ During inference, the learnt mean running_M E[Z] and variance running_V Var[Z] over the
entire training dataset are used to normalize Z.
– backward: takes input dLdBZ, how changes in BN layer output affects loss, computes and stores
the necessary gradients dLdBW, dLdBb to train learnable parameters BW and Bb. Returns
dLdZ, how the changes in BN layer input Z affect loss L for downstream computation.

    """

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0] # TODO DONE
        self.M = np.sum(Z, axis = 0)/self.N  # TODO DONE
        self.V = np.sum(np.multiply((Z-self.M), (Z-self.M)), 0)/self.N  # TODO DONE

        if eval == False:
            # training mode
            self.NZ = (self.Z - self.M)/np.sqrt(self.V+self.eps)  # TODO
            self.BZ = np.multiply(self.BW, self.NZ) + self.Bb  # TODO

            self.running_M = (self.alpha * self.running_M) + ((1-self.alpha) * self.M) # TODO
            self.running_V = (self.alpha * self.running_V) + ((1-self.alpha) * self.V) # TODO
        else:
            # inference mode
            self.NZ =  (self.Z - self.running_M)/np.sqrt((self.running_V + self.eps))  # TODO
            self.BZ = np.multiply(self.BW, self.NZ) + self.Bb  # TODO

        return self.BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(np.multiply(dLdBZ, self.NZ) , axis=0) # TODO
        self.dLdBb = np.sum(dLdBZ , axis = 0) # TODO

        dLdNZ = np.multiply(dLdBZ, self.BW)  # TODO
        dLdV = -0.5 * (np.sum( np.multiply(np.multiply(dLdNZ, (self.Z - self.M)), np.power( (self.V + self.eps) , -1.5) ) , axis=0)) # TODO
        dNZdM = (-1*np.power((self.V + self.eps), -0.5)) + (-0.5 * np.multiply( (self.Z - self.M) , (np.power(self.V + self.eps, -1.5) * (-2 * (np.sum(self.Z - self.M, axis=0)/self.N ))) ) ) 
        dLdM = np.sum( dLdNZ * dNZdM , axis=0)  # TODO

        dLdZ = (dLdNZ * np.power(self.V + self.eps, -0.5)) + (dLdV * (2 * (1/self.N) * (self.Z-self.M))) + ((1/self.N) * dLdM)  # TODO

        return dLdZ

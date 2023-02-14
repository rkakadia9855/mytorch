import numpy as np

"""
FORWARD: forward method takes in a batch of data Z of shape N ⇥ C (representing N samples
where each sample has C features), and applies the activation function to each element of Z to
compute output A of shape N ⇥ C.

"""

"""
BACKWARD: backward method calculates and returns dAdZ, how changes in pre-activation features
Z affect post-activation values A. It is used to enable downstream computation, as seen in
subsequent sections.

"""


class Identity:

    def forward(self, Z):

        self.A = Z

        return self.A

    def backward(self):

        dAdZ = np.ones(self.A.shape, dtype="f")

        return dAdZ


class Sigmoid:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Sigmoid.
    """

    def forward(self, Z):

        self.A = 1/(1 + np.exp(-1*Z))

        return self.A

    def backward(self):

        dAdZ = self.A - (np.multiply(self.A, self.A))

        return dAdZ


class Tanh:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on Tanh.
    """
    def forward(self, Z):

        self.A = (np.exp(Z) - np.exp(-1*Z))/(np.exp(Z) + np.exp(-1*Z))

        return self.A

    def backward(self):

        # A = (e^z - e^-z)/(e^z + e^-z)
        # dA/dZ = 1-tanh^2 = 1 - A^2
        dAdZ = 1 - np.multiply(self.A, self.A)

        return dAdZ


class ReLU:
    """
    On same lines as above:
    Define 'forward' function
    Define 'backward' function
    Read the writeup for further details on ReLU.
    """

    def forward(self, Z):

        self.A = np.maximum(0, Z)

        return self.A

    def backward(self):
        
        dAdZ = np.where(self.A <= 0, 0, 1)

        return dAdZ

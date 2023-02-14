import numpy as np

from mytorch.nn.linear import Linear
from mytorch.nn.activation import ReLU

"""

forward: forward method takes input data A0 and applies the linear transformation 
self.layers[i].forward and activation function self.f[i].forward for i = 0, ..., l % 19 
where l is the number of layers, to compute output Al

backward: backward method takes in dLdAl, how changes in loss L affect model output Al, and
performs back-propagation from the last layer to the first layer by calling self.f[i].backward
and self.layers[i].backward for i = l % 1, ..., 0. It does not return anything

"""


class MLP0:

    def __init__(self, debug=False):

        self.layers = [Linear(2, 3)]
        self.f = [ReLU()]

        self.debug = debug

    def forward(self, A0):

        Z0 = self.layers[0].forward(A0)  # TODO DONE
        A1 = self.f[0].forward(Z0)  # TODO DONE

        if self.debug:

            self.Z0 = Z0
            self.A1 = A1

        return A1

    def backward(self, dLdA1):

        dA1dZ0 = self.f[0].backward()  # TODO DONE
        dLdZ0 = np.multiply(dLdA1, dA1dZ0)  # TODO DONE
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO DONE

        if self.debug:

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0


class MLP1:

    def __init__(self, debug=False):
        """
        Initialize 2 linear layers. Layer 1 of shape (2,3) and Layer 2 of shape (3, 2).
        Use Relu activations for both the layers.
        Implement it on the same lines(in a list) as MLP0
        """

        self.layers = [Linear(2, 3), Linear(3, 2)]  # TODO DONE
        self.f = [ReLU(), ReLU()]  # TODO DONE

        self.debug = debug

    def forward(self, A0):

        Z0 = self.layers[0].forward(A0)  # TODO DONE
        A1 = self.f[0].forward(Z0)  # TODO DONE

        Z1 = self.layers[1].forward(A1)  # TODO DONE
        A2 = self.f[1].forward(Z1)  # TODO DONE

        if self.debug:
            self.Z0 = Z0
            self.A1 = A1
            self.Z1 = Z1
            self.A2 = A2

        return A2

    def backward(self, dLdA2):

        dA2dZ1 = self.f[1].backward()  # TODO DONE
        dLdZ1 = np.multiply(dLdA2, dA2dZ1)  # TODO DONE
        dLdA1 = self.layers[1].backward(dLdZ1)  # TODO DONE

        dA1dZ0 = self.f[0].backward()  # TODO DONE
        dLdZ0 = np.multiply(dLdA1, dA1dZ0)  # TODO DONE
        dLdA0 = self.layers[0].backward(dLdZ0)  # TODO DONE

        if self.debug:

            self.dA2dZ1 = dA2dZ1
            self.dLdZ1 = dLdZ1
            self.dLdA1 = dLdA1

            self.dA1dZ0 = dA1dZ0
            self.dLdZ0 = dLdZ0
            self.dLdA0 = dLdA0

        return dLdA0


class MLP4:
    def __init__(self, debug=False):
        """
        Initialize 4 hidden layers and an output layer of shape below:
        Layer1 (2, 4),
        Layer2 (4, 8),
        Layer3 (8, 8),
        Layer4 (8, 4),
        Output Layer (4, 2)

        Refer the diagrmatic view in the writeup for better understanding.
        Use ReLU activation function for all the layers.)
        """
        # List of Hidden Layers
        self.layers = [Linear(2,4), Linear(4,8), Linear(8,8), Linear(8,4), Linear(4,2)]  # TODO DONE

        # List of Activations
        self.f = [ReLU(), ReLU(), ReLU(), ReLU(), ReLU()]  # TODO DONE

        self.debug = debug

    def forward(self, A):

        if self.debug:

            self.Z = []
            self.A = [A]

        L = len(self.layers)

        for i in range(L):

            Z = self.layers[i].forward(A)  # TODO DONE
            A = self.f[i].forward(Z)  # TODO DONE

            if self.debug:

                self.Z.append(Z)
                self.A.append(A)

        return A

    def backward(self, dLdA):

        if self.debug:

            self.dAdZ = []
            self.dLdZ = []
            self.dLdA = [dLdA]

        L = len(self.layers)

        for i in reversed(range(L)):

            dAdZ = self.f[i].backward()  # TODO DONE
            dLdZ = np.multiply(dLdA, dAdZ)  # TODO DONE
            dLdA = self.layers[i].backward(dLdZ)  # TODO DONE

            if self.debug:

                self.dAdZ = [dAdZ] + self.dAdZ
                self.dLdZ = [dLdZ] + self.dLdZ
                self.dLdA = [dLdA] + self.dLdA

        return dLdA

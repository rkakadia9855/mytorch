import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        """
        writeup: Two parameters define a linear layer: in feature (Cin) and out feature (Cout). 
        Zero initialize weight W and bias b based on the inputs. Refer to Table 5.1 to see how 
        the shapes of W and b are related to the inputs.

        Shape of W: Cout * Cin, Shape of b: Cout * 1
        """
        self.W = np.zeros((out_features, in_features))  # TODO DONE
        self.b = np.zeros((out_features, 1))  # TODO DONE

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        """
        writeup: forward method takes in a batch of data A of shape N ⇥ Cin (representing N samples
        where each sample has Cin features), and computes output Z of shape N ⇥ Cout – each data
        sample is now represented by Cout features.
        """
        self.A = A  # TODO DONE
        num_rows, num_cols = A.shape
        self.N = num_rows  # TODO store the batch size of input DONE
        # Think how will self.Ones helps in the calculations and uncomment below
        self.Ones = np.ones((self.N,1))
        Z = np.matmul(self.A, np.transpose(self.W)) + np.matmul(self.Ones, np.transpose(self.b))  # TODO

        return Z

    def backward(self, dLdZ):

        """
        writeup: backward method takes in input dLdZ, how changes in its output Z affect loss L. It
        calculates and stores dLdW, dLdb – how changes in the layer weights and bias affect
        loss, which are used to improve the model. It returns dLdA, how changes in the layer 
        inputs affect loss to enable downstream computation.
        """

        # Recall Z = A*W + b

        dZdA = np.transpose(self.W)  # TODO DONE Cin*Cout
        dZdW = self.A  # TODO DONE  N*Cin
        dZdb = self.Ones  # TODO DONE N*1

        dLdA = np.matmul(dLdZ, np.transpose(dZdA))  # TODO DONE
        dLdW = np.matmul(np.transpose(dLdZ), dZdW)  # TODO DONE
        dLdb = np.matmul(np.transpose(dLdZ), dZdb)  # TODO DONE
        self.dLdW = dLdW / self.N
        self.dLdb = dLdb / self.N

        if self.debug:

            self.dZdA = dZdA
            self.dZdW = dZdW
            self.dZdb = dZdb
            self.dLdA = dLdA

        return dLdA

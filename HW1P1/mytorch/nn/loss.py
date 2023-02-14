import numpy as np

"""
forward: forward method takes in model prediction A and desired output Y of the same shape
to calculate and return a loss value L. The loss value is a scalar quantity used to quantify the
mismatch between the network output and the desired output.

backward: backward method calculates and returns dLdA, how changes in model outputs A a↵ect
loss L. It is used to enable downstream computation, as seen in previous sections.
"""

class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y
        self.N, self.C = A.shape  # TODO DONE
        se = np.multiply((A-Y), (A-Y))  # TODO sqaured error DONE
        nOnes = np.ones((self.N, 1))
        cOnes = np.ones((self.C, 1))
        sse = np.matmul(np.matmul(np.transpose(nOnes), se), cOnes)  # TODO DONE
        mse = sse/(2 * self.N * self.C)  # TODO DONE

        return mse

    def backward(self):

        dLdA = (self.A-self.Y)/(self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        N, C = A.shape  # TODO DONE

        Ones_N = np.ones((N, 1))  # TODO  DONE
        Ones_C = np.ones((C, 1))  # TODO DONE

        z = A
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        self.softmax = e_x / div
        
        crossentropy = np.matmul( np.multiply((-1 * Y), np.log(self.softmax)), Ones_C)  # TODO DONE
        sum_crossentropy = np.matmul(np.transpose(Ones_N), crossentropy)  # TODO DONE
        L = sum_crossentropy / N

        return L

    def backward(self):

        z = self.A
        s = np.max(z, axis=1)
        s = s[:, np.newaxis] # necessary step to do broadcasting
        e_x = np.exp(z - s)
        div = np.sum(e_x, axis=1)
        div = div[:, np.newaxis] # dito
        softmax = e_x / div

        dLdA = softmax - self.Y  # TODO DONE

        return dLdA

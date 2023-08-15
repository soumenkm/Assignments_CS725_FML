import numpy as np
import math

class LogisticRegression:
    def __init__(self):
        """
        Initialize `self.weights` properly.
        Recall that for binary classification we only need 1 set of weights (hence `num_classes=1`).
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 1 # single set of weights needed
        self.d = 2 # input space is 2D. easier to visualize
        self.weights = np.random.rand(self.d+1, self.num_classes)
        self.v = np.zeros((self.d+1, self.num_classes))
        #self.weights = np.zeros((self.d+1, self.num_classes)) # w0 is bias
        #self.weights = np.array([0, 1, 0]) # w0 is bias

    def preprocess(self, input_x):
        """
        Preprocess the input any way you seem fit.
        """
        return input_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        x -- NumPy array with shape (self.d, 1)
        """
        z = np.dot(self.weights[1:].reshape((len(x),)),x) + self.weights[0]
        f = math.exp(z) / (1 + math.exp(z))
        return f

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        num_train = len(input_x)
        total_loss = 0
        for i in range(num_train):
            f = self.sigmoid(input_x[i])
            fpos = math.log(f) * input_y[i]
            fneg = math.log(1-f) * (1 - input_y[i])

            loss = - (fpos + fneg)
            total_loss += loss

        return total_loss/num_train

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        num_train = len(input_y)
        grad_vec = np.zeros_like(self.weights)
        for j in range(len(grad_vec)):
            bias_error = 0
            weight_error = 0
            for i in range(num_train):
                f = self.sigmoid(input_x[i])
                error = f - input_y[i]
                bias_error += error
                if j > 0:
                    weight_error += error * input_x[i][j-1]

            if j == 0:
                grad_vec[j] = bias_error/num_train
            else:
                grad_vec[j] = weight_error/num_train

        return grad_vec

    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        w_old = self.weights
        v_old = self.v
        v_new = v_old * momentum - learning_rate * grad
        self.weights = w_old + v_new


    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,)
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        num_train = len(input_x)
        output_y = np.zeros((num_train,))
        for i in range(num_train):
            f = self.sigmoid(input_x[i])
            if f >= 0.5:
                output_y[i] = 1
            else:
                output_y[i] = 0

        return output_y


class LinearClassifier:
    def __init__(self):
        """
        Initialize `self.weights` properly.
        We have given the default zero intialization with bias term (hence the `d+1`).
        You are free to experiment with various other initializations including random initialization.
        Make sure to mention your initialization strategy in your report for this task.
        """
        self.num_classes = 3 # 3 classes
        self.d = 4 # 4 dimensional features
        #self.weights = np.zeros((self.d+1, self.num_classes)) # w0 is bias
        self.v = np.zeros((self.d+1, self.num_classes))
        self.weights = np.random.rand(self.d+1, self.num_classes) # w0 is bias

    def preprocess(self, train_x):
        """
        Preprocess the input any way you seem fit.
        """
        return train_x

    def sigmoid(self, x):
        """
        Implement a sigmoid function if you need it. Ignore otherwise.
        """
        score_vec = np.zeros((self.num_classes,))
        for k in range(self.num_classes):
            score_vec[k] = np.dot(self.weights[1:,k].reshape((len(x)),),x) + self.weights[0][k]

        softmax_vec = np.zeros_like(score_vec)
        exp_score_vec = np.exp(score_vec)
        for k in range(self.num_classes):
            softmax_vec[k] = exp_score_vec[k]/np.sum(exp_score_vec)

        return softmax_vec

    def onehot_encode(self, input_y):
        """
        input_y -- NumPy array with shape (N,)
        Returns: a onehot encoding of input_y as NumPy array with shape (N, self.num_classes)
        """
        onehot_y = np.zeros((len(input_y),self.num_classes))
        for i in range(len(input_y)):
            onehot_y[i][input_y[i]] = 1

        return onehot_y

    def calculate_loss(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: a single scalar value corresponding to the loss.
        """
        input_y_oh = self.onehot_encode(input_y)
        loss_vec = np.zeros((len(input_x),))
        for i in range(len(loss_vec)):
            loss_vec[i] = -1 * np.dot(np.log(self.sigmoid(input_x[i])), input_y_oh[i])

        return np.mean(loss_vec)

    def calculate_gradient(self, input_x, input_y):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        input_y -- NumPy array with shape (N,)
        Returns: the gradient of loss function wrt weights.
        Ensure that gradient.shape == self.weights.shape.
        """
        num_train = len(input_y)
        input_y_oh = self.onehot_encode(input_y)
        grad_matrix = np.zeros_like(self.weights) # (d+1) * K

        error_matrix = np.zeros((num_train, self.num_classes))
        for i in range(num_train):
            error_matrix[i] = np.transpose(self.sigmoid(input_x[i]) - input_y_oh[i])

        for k in range(grad_matrix.shape[1]):
            for d in range(grad_matrix.shape[0]):
                if d == 0:
                    grad_matrix[d][k] = np.sum(error_matrix[:,k])/num_train
                else:
                    grad_matrix[d][k] = np.dot(error_matrix[:,k],input_x[:,d-1])/num_train

        return grad_matrix


    def update_weights(self, grad, learning_rate, momentum):
        """
        Arguments:
        grad -- NumPy array with same shape as `self.weights`
        learning_rate -- scalar
        momentum -- scalar
        Returns: nothing
        The function should update `self.weights` with the help of `grad`, `learning_rate` and `momentum`
        """
        w_old = self.weights
        v_old = self.v
        v_new = v_old * momentum - learning_rate * grad
        self.weights = w_old + v_new

    def get_prediction(self, input_x):
        """
        Arguments:
        input_x -- NumPy array with shape (N, self.d) where N = total number of samples
        Returns: a NumPy array with shape (N,)
        The returned array must be the list of predicted class labels for every input in `input_x`
        """
        num_train = len(input_x)
        output_y = np.zeros((num_train,))
        for i in range(num_train):
            softmax_vec = self.sigmoid(input_x[i])
            output_y[i] = np.argmax(softmax_vec)

        return output_y

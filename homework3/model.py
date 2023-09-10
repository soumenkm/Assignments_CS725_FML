import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

class Bernoulli:

    def __init__(self, random_sample):
        """random_sample : np.array of shape (N,)"""
        self.sample = random_sample
        self.par_p = 0
        self.N = self.sample.shape[0]

    def estimate_parameter(self):
        num_of_success = self.sample[self.sample == 1].shape[0]
        self.par_p = num_of_success/self.N

    @staticmethod
    def estimate_likelihood(x, p):
        """x : int [0 or 1]"""
        pmf_x =  (p ** x) * ((1-p) ** (1-x))
        return pmf_x

class Exponential:

    def __init__(self, random_sample):
        """random_sample : np.array of shape (N,)"""
        self.sample = random_sample
        self.par_lambda = 0

    def estimate_parameter(self):
        self.par_lambda = 1/np.mean(self.sample)

    @staticmethod
    def estimate_likelihood(x, _lambda):
        """x : float"""
        pdf_x = _lambda * np.exp(-1 * _lambda * x)
        return pdf_x

class Guassian:

    def __init__(self, random_sample):
        """random_sample : np.array of shape (N,)"""
        self.sample = random_sample
        self.par_mu = 0
        self.par_sigma_square = 0

    def estimate_parameter(self):
        self.par_mu = np.mean(self.sample)
        square_dist = (self.sample - self.par_mu)**2
        self.par_sigma_square = np.mean(square_dist)

    @staticmethod
    def estimate_likelihood(x, mean, var):
        """x : float"""
        pdf_x = (1/(np.sqrt(2 * np.pi * var))) * np.exp(-0.5 * ((x - mean)**2) / var)
        return pdf_x

class Laplace:

    def __init__(self, random_sample):
        """random_sample : np.array of shape (N,)"""
        self.sample = random_sample
        self.par_mu = 0
        self.par_b = 0

    def estimate_parameter(self):
        self.par_mu = np.median(self.sample)
        abs_dist = np.abs(self.sample - self.par_mu)
        self.par_b = np.mean(abs_dist)

    @staticmethod
    def estimate_likelihood(x, mu, b):
        """x : float"""
        pdf_x = 1/(2*b) * np.exp(-1 * np.abs(x - mu) / b)
        return pdf_x

class Multinomial:

    def __init__(self, random_sample):
        """random_sample : np.array of shape (N,)"""
        self.sample = random_sample
        self.category = np.unique(self.sample)
        self.par_pk = [0] * self.category.shape[0]
        self.N = self.sample.shape[0]

    def estimate_parameter(self):
        for k,i in enumerate(self.category):
            count_success_ith_cat = self.sample[self.sample == i].shape[0]
            self.par_pk[k] = count_success_ith_cat/self.N

    @staticmethod
    def estimate_likelihood(x, pk_vec):
        """x : int [0, 1, ... , k-1]
        pk_vec : list of p of size [k,] where k is number of class"""

        pmf_x = pk_vec[int(x)] ** x
        return pmf_x

class NaiveBayes:

    def __init__(self):

        self.priors = {"0":0,"1":0,"2":0}
        self.guassian = {"0":[],"1":[],"2":[]}
        self.bernoulli = {"0":[],"1":[],"2":[]}
        self.laplace = {"0":[],"1":[],"2":[]}
        self.exponential = {"0":[],"1":[],"2":[]}
        self.multinomial = {"0":[],"1":[],"2":[]}
        self.num_of_class = 0

    def fit(self, X, y):

        """Start of your code."""
        """
        X : np.array of shape (n,10)
        y : np.array of shape (n,)
        Create a variable to store number of unique classes in the dataset.
        Assume Prior for each class to be ratio of number of data points in that class to total number of data points.
        Fit a distribution for each feature for each class.
        Store the parameters of the distribution in suitable data structure, for example you could create a class for each distribution and store the parameters in the class object.
        You can create a separate function for fitting each distribution in its and call it here.
        """

        unique_class = np.unique(y)
        self.num_of_class = unique_class.shape[0]
        num_of_datapoint = y.shape[0]

        # Prior probability of y
        for k,i in enumerate(unique_class):
            count_of_label = y[y == i].shape[0]
            self.priors[str(k)] = count_of_label/num_of_datapoint

        # MLE of parameters for all the distribution x1, x2, ... , x10
        for i in unique_class:

            x1 = X[:,0]
            x2 = X[:,1]

            random_sample_x1 = x1[y == i]
            normal_obj_x1 = Guassian(random_sample_x1)
            normal_obj_x1.estimate_parameter()

            random_sample_x2 = x2[y == i]
            normal_obj_x2 = Guassian(random_sample_x2)
            normal_obj_x2.estimate_parameter()

            self.guassian[f"{int(i)}"] = [normal_obj_x1.par_mu,
                                     normal_obj_x2.par_mu,
                                     normal_obj_x1.par_sigma_square,
                                     normal_obj_x2.par_sigma_square]

            x3 = X[:,2]
            x4 = X[:,3]

            random_sample_x3 = x3[y == i]
            normal_obj_x3 = Bernoulli(random_sample_x3)
            normal_obj_x3.estimate_parameter()

            random_sample_x4 = x4[y == i]
            normal_obj_x4 = Bernoulli(random_sample_x4)
            normal_obj_x4.estimate_parameter()

            self.bernoulli[f"{int(i)}"] = [normal_obj_x3.par_p,
                                     normal_obj_x4.par_p]

            x5 = X[:,4]
            x6 = X[:,5]

            random_sample_x5 = x5[y == i]
            normal_obj_x5 = Laplace(random_sample_x5)
            normal_obj_x5.estimate_parameter()

            random_sample_x6 = x6[y == i]
            normal_obj_x6 = Laplace(random_sample_x6)
            normal_obj_x6.estimate_parameter()

            self.laplace[f"{int(i)}"] = [normal_obj_x5.par_mu,
                                    normal_obj_x6.par_mu,
                                    normal_obj_x5.par_b,
                                    normal_obj_x6.par_b]

            x7 = X[:,6]
            x8 = X[:,7]

            random_sample_x7 = x7[y == i]
            normal_obj_x7 = Exponential(random_sample_x7)
            normal_obj_x7.estimate_parameter()

            random_sample_x8 = x8[y == i]
            normal_obj_x8 = Exponential(random_sample_x8)
            normal_obj_x8.estimate_parameter()

            self.exponential[f"{int(i)}"] = [normal_obj_x7.par_lambda,
                                        normal_obj_x8.par_lambda]

            x9 = X[:,8]
            x10 = X[:,9]

            random_sample_x9 = x9[y == i]
            normal_obj_x9 = Multinomial(random_sample_x9)
            normal_obj_x9.estimate_parameter()

            random_sample_x10 = x10[y == i]
            normal_obj_x10 = Multinomial(random_sample_x10)
            normal_obj_x10.estimate_parameter()

            self.multinomial[f"{int(i)}"] = [normal_obj_x9.par_pk,
                                        normal_obj_x10.par_pk]

        """End of your code."""

    def predict(self, X):
        """Start of your code."""
        """
        X : np.array of shape (n,10)

        Calculate the posterior probability using the parameters of the distribution calculated in fit function.
        Take care of underflow errors suitably (Hint: Take log of probabilities)
        Return an np.array() of predictions where predictions[i] is the predicted class for ith data point in X.
        It is implied that prediction[i] is the class that maximizes posterior probability for ith data point in X.
        You can create a separate function for calculating posterior probability and call it here.
        """

        predictions = np.zeros((X.shape[0],))
        for j in range(X.shape[0]):

            score = np.zeros((self.num_of_class,))
            for i in range(self.num_of_class):

                likelihood = np.zeros((10,))

                x1 = X[j,0]
                x2 = X[j,1]

                likelihood[0] = Guassian.estimate_likelihood(x1,
                                             self.guassian[str(i)][0],
                                             self.guassian[str(i)][2])
                likelihood[1] = Guassian.estimate_likelihood(x2,
                                             self.guassian[str(i)][1],
                                             self.guassian[str(i)][3])

                x3 = X[j,2]
                x4 = X[j,3]

                likelihood[2] = Bernoulli.estimate_likelihood(x3,
                                              self.bernoulli[str(i)][0])
                likelihood[3] = Bernoulli.estimate_likelihood(x4,
                                              self.bernoulli[str(i)][1])

                x5 = X[j,4]
                x6 = X[j,5]

                likelihood[4] = Laplace.estimate_likelihood(x5,
                                            self.laplace[str(i)][0],
                                            self.laplace[str(i)][2])
                likelihood[5] = Laplace.estimate_likelihood(x6,
                                            self.laplace[str(i)][1],
                                            self.laplace[str(i)][3])

                x7 = X[j,6]
                x8 = X[j,7]

                likelihood[6] = Exponential.estimate_likelihood(x7,
                                                self.exponential[str(i)][0])
                likelihood[7] = Exponential.estimate_likelihood(x8,
                                                self.exponential[str(i)][1])

                x9 = X[j,8]
                x10 = X[j,9]

                likelihood[8] = Multinomial.estimate_likelihood(x9,
                                                self.multinomial[str(i)][0])
                likelihood[9] = Multinomial.estimate_likelihood(x10,
                                                self.multinomial[str(i)][1])

                score[i] = np.sum(np.log(likelihood)) + np.log(self.priors[str(i)])

            predictions[j] = np.argmax(score)

        return predictions

        """End of your code."""

    def getParams(self):
        """
        Return your calculated priors and parameters for all the classes in the form of dictionary that will be used for evaluation
        Please don't change the dictionary names
        Here is what the output would look like:
        priors = {"0":0.2,"1":0.3,"2":0.5}
        gaussian = {"0":[mean_x1,mean_x2,var_x1,var_x2],"1":[mean_x1,mean_x2,var_x1,var_x2],"2":[mean_x1,mean_x2,var_x1,var_x2]}
        bernoulli = {"0":[p_x3,p_x4],"1":[p_x3,p_x4],"2":[p_x3,p_x4]}
        laplace = {"0":[mu_x5,mu_x6,b_x5,b_x6],"1":[mu_x5,mu_x6,b_x5,b_x6],"2":[mu_x5,mu_x6,b_x5,b_x6]}
        exponential = {"0":[lambda_x7,lambda_x8],"1":[lambda_x7,lambda_x8],"2":[lambda_x7,lambda_x8]}
        multinomial = {"0":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"1":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]],"2":[[p0_x9,...,p4_x9],[p0_x10,...,p7_x10]]}
        """
        priors = self.priors
        guassian = self.guassian
        bernoulli = self.bernoulli
        laplace = self.laplace
        exponential = self.laplace
        multinomial = self.multinomial

        """Start your code"""





        """End your code"""
        return (priors, guassian, bernoulli, laplace, exponential, multinomial)


def save_model(model,filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open("model.pkl","wb")
    pkl.dump(model,file)
    file.close()

def load_model(filename="model.pkl"):
    """

    You are not required to modify this part of the code.

    """
    file = open(filename,"rb")
    model = pkl.load(file)
    file.close()
    return model

def visualise(data_points,labels):
    """
    datapoints: np.array of shape (n,2)
    labels: np.array of shape (n,)
    """

    plt.figure(figsize=(8, 6))
    plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, cmap='viridis')
    plt.colorbar()
    plt.title('Generated 2D Data from 5 Gaussian Distributions')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


def net_f1score(predictions, true_labels):
    """Calculate the multclass f1 score of the predictions.
    For this, we calculate the f1-score for each class

    Args:
        predictions (np.array): The predicted labels.
        true_labels (np.array): The true labels.

    Returns:
        float(list): The f1 score of the predictions for each class
    """

    def precision(predictions, true_labels, label):
        """Calculate the multclass precision of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The precision of the predictions.
        """
        """Start of your code."""

        tp = np.sum((predictions == label) & (true_labels == label))
        fp = np.sum((predictions == label) & (true_labels != label))

        prec = tp / (tp + fp)
        return prec

        """End of your code."""

    def recall(predictions, true_labels, label):
        """Calculate the multclass recall of the predictions.
        For this, we take the class with given label as the positive class and the rest as the negative class.
        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The recall of the predictions.
        """
        """Start of your code."""

        tp = np.sum((predictions == label) & (true_labels == label))
        fn = np.sum((predictions != label) & (true_labels == label))

        rec = tp / (tp + fn)
        return rec

        """End of your code."""


    def f1score(predictions, true_labels, label):
        """Calculate the f1 score using it's relation with precision and recall.

        Args:
            predictions (np.array): The predicted labels.
            true_labels (np.array): The true labels.

        Returns:
            float: The f1 score of the predictions.
        """

        """Start of your code."""

        prec = precision(predictions, true_labels, label)
        rec = recall(predictions, true_labels, label)

        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) != 0 else 0
        return f1

        """End of your code."""

    f1s = []
    for label in np.unique(true_labels):
        f1s.append(f1score(predictions, true_labels, label))
    return f1s

def accuracy(predictions,true_labels):
    """

    You are not required to modify this part of the code.

    """
    return np.sum(predictions==true_labels)/predictions.size

if __name__ == "__main__":
    """

    You are not required to modify this part of the code.

    """

    # Load the data
    train_dataset = pd.read_csv('./data/train_dataset.csv',index_col=0).to_numpy()
    validation_dataset = pd.read_csv('./data/validation_dataset.csv',index_col=0).to_numpy()

    # Extract the data
    train_datapoints = train_dataset[:,:-1]
    train_labels = train_dataset[:, -1]
    validation_datapoints = validation_dataset[:, 0:-1]
    validation_labels = validation_dataset[:, -1]

    # Visualize the data
    # visualise(train_datapoints, train_labels)

    # Train the model
    model = NaiveBayes()
    model.fit(train_datapoints, train_labels)

    # Make predictions
    train_predictions = model.predict(train_datapoints)
    validation_predictions = model.predict(validation_datapoints)

    # Calculate the accuracy
    train_accuracy = accuracy(train_predictions, train_labels)
    validation_accuracy = accuracy(validation_predictions, validation_labels)

    # Calculate the f1 score
    train_f1score = net_f1score(train_predictions, train_labels)
    validation_f1score = net_f1score(validation_predictions, validation_labels)

    # Print the results
    print('Training Accuracy: ', train_accuracy)
    print('Validation Accuracy: ', validation_accuracy)
    print('Training F1 Score: ', train_f1score)
    print('Validation F1 Score: ', validation_f1score)

    # Save the model
    save_model(model)

    # Visualize the predictions
    # visualise(validation_datapoints, validation_predictions)


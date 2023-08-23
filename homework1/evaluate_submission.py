import numpy as np
from utils import load_model
import os

def evaluate_model(dataset_name, model_name):
    print(f'Loading validation split of "{dataset_name}" dataset')
    valid_x = np.load(f'./data/{dataset_name}/val_x.npy')
    valid_y = np.load(f'./data/{dataset_name}/val_y.npy')

    print(f'Loading "{model_name}" model')
    model = load_model(model_name)()

    print(f'Loading weights')
    # file_name = [f'./checkpoints/{dataset_name}/{i}' for i in os.listdir(f'./checkpoints/{dataset_name}') if i.endswith('.npy')]
    model.weights = np.load(f'./submission/best_{dataset_name}.weights.npy')
    # model.weights = np.load(file_name[0])

    valid_x = model.preprocess(valid_x) # Must be (N, d)
    valid_loss = model.calculate_loss(valid_x, valid_y)
    valid_pred = model.get_prediction(valid_x) # Must be (N,)
    valid_acc = (valid_pred == valid_y).mean()

    print(f'Validation loss = {valid_loss}. Validation accuracy = {valid_acc * 100:.2f}%.')

def main():
    print(32*'-')
    evaluate_model('binary', 'logistic_regression')
    print(32*'-')
    evaluate_model('iris', 'linear_classifier')
    print(32*'-')

if __name__ == '__main__':
    main()

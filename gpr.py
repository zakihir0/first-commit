# -*- coding: utf-8 -*-
# %reset -f
"""
@author: Hiromasa Kaneko
"""

import matplotlib.figure as figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF, ConstantKernel, Matern, DotProduct
from sklearn.model_selection import train_test_split, GridSearchCV

number_of_test_samples = 200
fold_number = 5

class GPRegressor:
    def __init__(self, X, y, testsize, fold_number):
        self.X = X
        self.y = y
        self.testsize = testsize
        self.fold_number = fold_number
        
    def validation(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=self.testsize, random_state=0)
        X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0, ddof=1)
        y_train = (y_train - y_train.mean()) / y_train.std(ddof=1)
        X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0, ddof=1)
        
        return X_train, X_test, y_train, y_test
    
    def GPR(self):
        X_train, X_test, y_train, y_test = self.validation()
        
        kernels = [ConstantKernel() * DotProduct() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel(),
        ConstantKernel() * RBF() + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * RBF(np.ones(X_train.shape[1])) + WhiteKernel(),
        ConstantKernel() * RBF(np.ones(X_train.shape[1])) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=1.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=0.5) + WhiteKernel() + ConstantKernel() * DotProduct(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel(),
        ConstantKernel() * Matern(nu=2.5) + WhiteKernel() + ConstantKernel() * DotProduct()]
        
        cv_model = GridSearchCV(GaussianProcessRegressor(alpha=0), {'kernel': kernels}, cv=fold_number)
        cv_model.fit(X_train, y_train)
        optimal_kernel = cv_model.best_params_['kernel']
        model = GaussianProcessRegressor(optimal_kernel, alpha=0)
        model.fit(X_train, y_train)
        
        return model
    
    def calcurated(self):
        X_train, X_test, y_train, y_test = self.validation()
        model = self.GPR()
        calculated_y_train = model.predict(X_train) * y_train.std(ddof=1) + y_train.mean()
        
        # r2, RMSE, MAE
        print('r2: {0}'.format(float(1 - sum((y_train - calculated_y_train) ** 2) / sum((y_train - y_train.mean()) ** 2))))
        print('RMSE: {0}'.format(float((sum((y_train - calculated_y_train) ** 2) / len(y_train)) ** 0.5)))
        print('MAE: {0}'.format(float(sum(abs(y_train - calculated_y_train)) / len(y_train))))
        
        return calculated_y_train
        
    def calcplot(self):
        X_train, X_test, y_train, y_test = self.validation()
        calculated_y_train = self.calcurated()
        
        # yy-plot
        plt.rcParams['font.size'] = 18
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(y_train, calculated_y_train, c='blue')
        y_max = np.max(np.array([np.array(y_train), calculated_y_train]))
        y_min = np.min(np.array([np.array(y_train), calculated_y_train]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Calculated Y')
        plt.show()
        
    def prediction(self):
        X_train, X_test, y_train, y_test = self.validation()
        model = self.GPR()
        
        # prediction
        predicted_y_test, predicted_y_test_std = model.predict(X_test, return_std=True)
        predicted_y_test = predicted_y_test * y_train.std(ddof=1) + y_train.mean()
        predicted_y_test_std = predicted_y_test_std * y_train.std(ddof=1)
        
        # r2p, RMSEp, MAEp
        print('r2p: {0}'.format(float(1 - sum((y_test - predicted_y_test) ** 2) / sum((y_test - y_test.mean()) ** 2))))
        print('RMSEp: {0}'.format(float((sum((y_test - predicted_y_test) ** 2) / len(y_test)) ** 0.5)))
        print('MAEp: {0}'.format(float(sum(abs(y_test - predicted_y_test)) / len(y_test))))
        print('')
        return  predicted_y_test, predicted_y_test_std
        
    def pred_yyplot(self):
        predicted_y_test, predicted_y_test_std = self.prediction()
        X_train, X_test, y_train, y_test = self.validation()
        
        # yy-plot
        plt.figure(figsize=figure.figaspect(1))
        plt.scatter(y_test, predicted_y_test, c='blue')
        y_max = np.max(np.array([np.array(y_test), predicted_y_test]))
        y_min = np.min(np.array([np.array(y_test), predicted_y_test]))
        plt.plot([y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)],
                 [y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min)], 'k-')
        plt.ylim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlim(y_min - 0.05 * (y_max - y_min), y_max + 0.05 * (y_max - y_min))
        plt.xlabel('Actual Y')
        plt.ylabel('Predicted Y')
        plt.show()
        
    def sigmaplot(self):
        predicted_y_test, predicted_y_test_std = self.prediction()
        X_train, X_test, y_train, y_test = self.validation()
        
        plt.scatter(predicted_y_test_std, abs(y_test - predicted_y_test), c='blue')
        plt.xlabel('Std. of estimated Y')
        plt.ylabel('Error of Y')
        plt.show()
import pandas as pd
import numpy as np
import statsmodels.api as sm
import collections
import scipy.stats
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import time
import sys

class OrderLogit():

    def __init__(self, endog, exog, method, maxiter, theta_initial, descending=True):
        self.x = np.array(exog)
        self.y = np.array(endog)
        self.method = method
        self.maxiter = maxiter    
        self.descending = descending
        self.obs = len(self.y)
        self.cats = len(np.unique(self.y))
        self.columns = exog.columns.tolist()

        # Initialize parameters
        if len(theta_initial) < self.cats - 1:
            self.theta = np.concatenate([np.arange(-5, 2, 7/(self.cats - 1)), np.zeros(self.x.shape[1])])  
        else:
            self.theta = theta_initial
                
        # do one hot encoding for dependent variable
        if self.descending == False:
            self.integer_encoded = LabelEncoder().fit_transform(self.y)
        elif self.descending == True:    
            self.integer_encoded = LabelEncoder().fit_transform(self.y)
            self.integer_encoded = np.unique(self.integer_encoded).max() - self.integer_encoded 

        self.integer_encoded = self.integer_encoded.reshape(len(self.integer_encoded), 1)
        self.onehot_encoded = OneHotEncoder(sparse_output=False).fit_transform(self.integer_encoded) 

    # define the sigmoid function
    def sigmoid(self, t):
        """
        sigmoid function, returns 1 / (1 + exp(-t))
        """
        idx = t > 0
        out = np.empty(t.size, dtype='float')
        out[idx]  = 1. / (1 + np.exp(-t[idx]))
        exp_t = np.exp(t[~idx])
        out[~idx] = exp_t / (1. + exp_t)
        return out

    # define the derivative of sigmoid function
    def d_sigmoid(self,t):
        """
        derivative of sigmoid function, returns (1-sigmoid(t))*sigmoid(t)
        """
        out = self.sigmoid(t)*(1-self.sigmoid(t))
        return out


    # create common variables that will be used later in caculating loss_function, jacobian or hessian
    def dataprep(self):
        self.alpha = self.theta[0:self.cats-1]
        self.beta = self.theta[self.cats-1:]
        self.eta = self.x.dot(self.beta)
        self.y_expected = np.zeros([self.obs, self.cats], dtype=float)
        for k in range(self.cats):
            if k == 0:
                self.y_expected[:,k] = self.sigmoid(self.eta + self.alpha[k])
            elif k>0 and k< self.cats-1:
                self.y_expected[:,k] = self.sigmoid(self.eta + self.alpha[k])  - self.sigmoid(self.eta + self.alpha[k-1])
            elif k == self.cats-1:
                self.y_expected[:,k] = 1- self.sigmoid(self.eta + self.alpha[k-1])
        if self.method == 'Fisher':
            self.y_matrix = self.y_expected
        if self.method == 'Newton':
            self.y_matrix = self.onehot_encoded

    # define the negative log likelihood function: elements in intercepts array self.alpha should be increasing, if not, t
    def loss_function(self):
        if (all(self.alpha[i] < self.alpha[i + 1] for i in range(len(self.alpha)-1))):
            d = np.where(self.y_expected>0, np.log(self.y_expected, where= (self.y_expected>0)), float(sys.maxsize))
            log_likelihood = -(d*self.onehot_encoded).sum()
        else:
            log_likelihood = float(sys.maxsize)
        return log_likelihood
 

    def jacobian(self):
        ds = np.zeros([self.obs, self.cats], dtype=float)
        for k in range(self.cats):
            if k == 0:
                ds[:,k] = (1 - self.sigmoid(self.eta + self.alpha[k]))
            elif k>0 and k< self.cats-1:
                ds[:,k] = 1 - self.sigmoid(self.eta + self.alpha[k])  - self.sigmoid(self.eta + self.alpha[k-1])
            elif k == self.cats-1:
                ds[:,k] = -self.sigmoid(self.eta + self.alpha[k-1])
        w = (ds*self.onehot_encoded).sum(axis=1)
        d_beta = self.x.T.dot(w)
        del_alpha = []
        for k in range(self.cats-1):
            d_alpha = np.zeros([self.obs, self.cats], dtype=float)
            if k==0:
                d_alpha[:,k] = (1- self.sigmoid(self.eta + self.alpha[k]))
                d_alpha[:,k+1] = ((-self.d_sigmoid(self.eta + self.alpha[k]))/(self.sigmoid(self.eta + self.alpha[k+1]) - self.sigmoid(self.eta + self.alpha[k])))
            elif k>0 and k< self.cats-2:
                d_alpha[:,k] = ((self.d_sigmoid(self.eta + self.alpha[k]))/(self.sigmoid(self.eta + self.alpha[k])-self.sigmoid(self.eta + self.alpha[k-1])))
                d_alpha[:,k+1] = ((-self.d_sigmoid(self.eta + self.alpha[k]))/(self.sigmoid(self.eta + self.alpha[k+1]) - self.sigmoid(self.eta + self.alpha[k])))
            elif k == self.cats-2:
                d_alpha[:,k] = ((self.d_sigmoid(self.eta + self.alpha[k]))/(self.sigmoid(self.eta + self.alpha[k])-self.sigmoid(self.eta + self.alpha[k-1])))
                d_alpha[:,k+1] = ((-self.sigmoid(self.eta + self.alpha[k]))/(self.sigmoid(self.eta + self.alpha[k])))
            del_alpha.append((d_alpha*self.onehot_encoded).sum())
        self.jac = -np.concatenate([del_alpha, d_beta])
        return self.jac
       

    def hessian(self):
    #####################################
        # hessian with respect to beta's
    #####################################
        dds = np.zeros([self.obs, self.cats], dtype=float)
        for k in range(self.cats):
            if k==0:
                dds[:,k] = -(self.d_sigmoid(self.eta + self.alpha[k]))
            elif k>0 and k< self.cats-1:
                dds[:,k] = -(self.d_sigmoid(self.eta + self.alpha[k]) + self.d_sigmoid(self.eta + self.alpha[k-1]))
            elif k == self.cats-1:
                dds[:,k] = -(self.d_sigmoid(self.eta + self.alpha[k-1]))
        w = (dds*self.y_matrix).sum(axis=1).reshape(self.obs,1)
        dd_beta = self.x.T.dot((w)*self.x)
    #####################################
        # hessian with respect to self.alpha's
    #####################################
        dd_alpha = np.zeros([len(self.alpha)], dtype=float)
        for k in range(self.cats-1):
            dd_alpha_diagonal = np.zeros([self.obs, self.cats],dtype=float)
            dd_alpha_nondiagonal = np.zeros([self.obs, self.cats],dtype=float)             
            if k==0:
                dd_alpha_diagonal[:,k] = -self.d_sigmoid(self.eta + self.alpha[k])
                dd_alpha_diagonal[:,k+1] = -self.d_sigmoid(self.eta + self.alpha[k])*(1 + (self.d_sigmoid(self.eta + self.alpha[k+1]) - self.sigmoid(self.eta + self.alpha[k]))**2)
                dd_alpha_nondiagonal[:,k+1] = self.d_sigmoid(self.eta + self.alpha[k])*((self.d_sigmoid(self.eta + self.alpha[k+1]))/(self.sigmoid(self.eta + self.alpha[k]))**2)
            elif k>0 and k< self.cats-2:
                dd_alpha_diagonal[:,k]   = -self.d_sigmoid(self.eta + self.alpha[k])*(1 + (self.d_sigmoid(self.eta + self.alpha[k-1])) / (self.sigmoid(self.eta + self.alpha[k]) - self.sigmoid(self.eta + self.alpha[k-1]))**2)
                dd_alpha_diagonal[:,k+1] = -self.d_sigmoid(self.eta + self.alpha[k])*(1 + (self.d_sigmoid(self.eta + self.alpha[k+1])) / (self.sigmoid(self.eta + self.dd_alpha[k+1]) - self.sigmoid(self.eta + self.alpha[k]))**2)
                dd_alpha_nondiagonal[:,k+1] = self.d_sigmoid(self.eta + self.alpha[k])*((self.d_sigmoid(self.eta+self.alpha[k+1])) / (self.sigmoid(self.eta + self.alpha[k+1]) - self.sigmoid(self.eta+self.alpha[k]))**2)
            elif k == self.cats-2:
                dd_alpha_diagonal[:,k]   = -self.d_sigmoid(self.eta + self.alpha[k])*(1 + (self.d_sigmoid(self.eta + self.alpha[k-1])) / (self.sigmoid(self.eta + self.alpha[k]) - self.sigmoid(self.eta + self.alpha[k-1]))**2)
                dd_alpha_diagonal[:,k+1] = -self.d_sigmoid(self.eta + self.alpha[k])
  
            if k< self.cats-1:
                dd_alpha[k] = (dd_alpha_nondiagonal*self.y_matrix).sum()


        ########################################
            # hessian interaction terms b/w self.alpha and beta
        ########################################
        dd_beta_alpha = np.zeros([len(self.beta), len(self.alpha)], dtype=float)
        for k in range(self.cats-1):
            del_beta_alpha = np.zeros([self.obs, self.cats], dtype=float)
            del_beta_alpha[:,k] = -self.d_sigmoid(self.eta + self.alpha[k])
            del_beta_alpha[:,k+1] = -self.d_sigmoid(self.eta + self.alpha[k])
            for i in range(len(self.beta)):
                dd_beta_alpha[i, k] = (self.x[:,i].reshape(self.obs,1)).T.dot((del_beta_alpha*self.y_matrix).sum(axis=1))
                                        
        hessian_matrix = np.zeros([len(self.alpha) + len(self.beta), len(self.alpha)+len(self.beta)], dtype=float)
        hessian_matrix[0:len(self.alpha), 0:len(self.alpha)] = dd_alpha
        hessian_matrix[len(self.alpha):, 0:len(self.alpha)] = dd_beta_alpha
        hessian_matrix = hessian_matrix + hessian_matrix.T
        hessian_matrix.flat[::hessian_matrix.shape[1]+1] /= 2
        hessian_matrix[len(self.alpha):, len(self.alpha):] = dd_beta
        self.hessian_matrix = -hessian_matrix
        return self.hessian_matrix


 

    def fit(self):
        i = 0
         
        FisherInformation = np.ones((len(self.theta), len(self.theta)))

        dJ_Theta = np.ones(len(self.theta))

        while ((i < self.maxiter) & (((abs(np.dot(FisherInformation, dJ_Theta))).max() > 10e-6) | ((abs(dJ_Theta)).max()>10e-6))):

            self.dataprep()
            old_theta = self.theta
            J_old = self.loss_function()
            # calculate the Fisher Information
            c_3 = 1
            if np.all(np.linalg.eigvals(self.hessian()) > 0):
                c_2 = 0
            else:
                c_2 = 1
            H_tilde = self.hessian() + c_2 * (c_3 + np.linalg.eig(self.hessian())[0].min()) * np.identity(self.hessian().shape[0], dtype=float)
            FisherInformation = np.linalg.inv(H_tilde)
            
            dJ_Theta = self.jacobian()
            # get the updated coefficients
            self.theta = old_theta - np.dot(FisherInformation, dJ_Theta)
            # calculate the updated cost
            self.dataprep()
            J = self.loss_function()
            
            if (J > J_old) | (J == J_old):
                self.theta = old_theta
                for j in range(30):
                    self.theta = old_theta - np.dot(FisherInformation, dJ_Theta/2**j)
                    self.dataprep()
                    J = self.loss_function()
                    if J < J_old:
                        break
             
            i += 1
            # print(i)
            # print(J)
            # print((abs(np.dot(FisherInformation, dJ_Theta)) / (2 * J)))
            # print((abs(dJ_Theta)).max())

 
    def summary(self):


        self.results_summary = pd.DataFrame(self.theta)
        self.results_summary = self.results_summary.rename(columns = {0: 'params'})
        
        intercept_cat = np.unique(self.y)
        
        if self.descending == True:
            intercept_cat[: :-1].sort()
        elif self.descending == False:
            intercept_cat.sort()
        
        name = intercept_cat.tolist()
        intercept_name = ["Intercept_" + str(ele).strip('.0') for ele in name]
        intercept_name = intercept_name[:-1]

    
        predict_name = ["P_" + str(ele).strip('.0') for ele in name]
        self.results_summary.index = intercept_name + self.columns
        self.results_summary['Standard Error'] = np.sqrt(np.diagonal(np.linalg.inv(self.hessian())))
        self.results_summary['Wald Chi-Square'] = (self.theta/np.sqrt(np.diagonal(np.linalg.inv(self.hessian()))))**2            
        self.results_summary['Pr > Wald Chi-Square'] = 1 - stats.chi2.cdf(self.results_summary['Wald Chi-Square'], 1)
        self.aic_ = 2*self.loss_function() + 2*((self.cats-1) + len(self.beta))
        self.bic_ = 2*self.loss_function() + ((self.cats-1) + len(self.beta))*np.log(self.obs)
    
        if np.any(np.linalg.eigvals(self.hessian()) < 0):
            self.convergence_status = -2 # The Hessian has at least one negative eigenvalue.
        elif np.any(abs(self.jacobian()) > 10e-6):    
            self.convergence_status = -1 # Absolute convergence criterion (maximum absolute gradient) was not satisfied.
        elif np.linalg.det(self.hessian()) == 0:
            self.convergence_status = 1 # singular hessian matrix    
        else: 
            self.convergence_status = 0 # converged successfully    
        self.predict_proba = pd.DataFrame(self.y_expected)
        self.predict_proba.columns = predict_name
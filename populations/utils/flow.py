"""
Storm Colloms 21/6/23

Defines Class to instantiate and train noramlising flow for each channel used in inference.
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from scipy.stats import entropy
from scipy.stats import norm, gaussian_kde

import copy
import torch
from  glasflow import RealNVP, CouplingNSF
from torch import nn


class NFlow():

    #initialise flow with inputs, conditionals, including the type of network, real non-volume preserving,
    #or neural spline flow
    #spline flow increases the flexibility in the flow model
    def __init__(self, no_trans, no_neurons, training_inputs, cond_inputs,
                no_binaries, batch_size, total_hps, RNVP=True, num_bins=4):
        """
        Initialise Flow with inputed data, either RNVP or Spline flow.

        Parameters
        ----------
        no_trans : int
            number of transforms to give the flow
        no_neurons : int
            number of neurons to give the flow
        training_inputs : int
            number of parameters in dataspace (binary parameters)
        cond_inputs : int
            number of conditional population hyperparameters
        no_binaries : int
            number of binaries in each population
        batch_size : int
            number of training + validation samples to use in each batch
        total_hps : array
            [no populations x cond_inputs]
            all chi_b alpha hyperparameters
        RNVP : bool
            whether or not to use realNVP flow, if False use spline
        num_bins : int
            number of bins to use for a spline flow
        """
        self.no_params = training_inputs
        self.no_binaries = no_binaries
        self.batch_size = batch_size

        self.total_hps = total_hps

        self.cond_inputs = cond_inputs

        if RNVP:
            self.network = RealNVP(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                    n_neurons = no_neurons, n_transforms = no_trans, n_blocks_per_transform = 2,
                                    linear_transform = None, batch_norm_between_transforms=True)
        else:
            self.network = CouplingNSF(n_inputs = training_inputs, n_conditional_inputs= cond_inputs,
                                        n_neurons = no_neurons, n_transforms = no_trans,
                                        n_blocks_per_transform = 2, batch_norm_between_transforms=True,
                                        linear_transform = None, num_bins=num_bins)

    #training and validation loop for the flow
    def trainval(self, lr, epochs, batch_no, filename, training_data, val_data):

        print('initialise flow first')

        #set optimiser for flow, optimises flow parameters:
        #(affine - s and t that shift and scale the transforms)
        #(spline - nodes used to model the distribution of CDFs)
        optimiser_f = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=0)

        n_epochs = epochs #number of iterations to train  - 1 epoch goes through through entire dataset
        n_batches = batch_no #number of batches of data in one iteration

        #initialize best flow model
        best_epoch = 0
        best_val_loss = np.inf
        best_model_f = copy.deepcopy(self.network.state_dict())

        #record network values and outputs in dictionary as training
        self.history = {'train': [], 'val': [], 'kl': [], 'lr':[], 'traing':[], 'valg':[], 'trainf':[], 'valf':[],
                         'avemeans':[],'avelogstds':[]}

        #training loop
        for n in range(n_epochs): 
            train_loss = 0
            trainlossf=0
            trainlossg=0
            #set flow into training mode
            self.network.train()
            
            #Training
            for _ in range(n_batches):
                #split training data into - train: binary params; conditional: pop hyperparams
                x_train, x_conditional, xweights = self.get_training_data(training_data)

                #sets flow optimisers gradients to zero
                optimiser_f.zero_grad()
                #calculate the training loss function for flow as -log_prob
                loss_f = -self.network.log_prob(x_train, conditional=x_conditional).mean()
                #computes gradient of flow network parameters
                loss_f.backward()
                #steps optimtiser down gradient of loss surface
                optimiser_f.step()
                #track flow losses
                trainlossf += loss_f.item()
                train_loss += loss_f.item()

            #track and average losses
            train_loss /= n_batches
            trainlossf /= n_batches
            trainlossg /= n_batches
            self.history['train'].append(train_loss)
            self.history['trainf'].append(trainlossf)
            self.history['traing'].append(trainlossg)
            
            # Validate
            with torch.no_grad(): #disables gradient caluclation
                #call validation data
                x_val, x_conditional_val, x_weights_val = self.get_val_data(val_data)
                val_loss_g=0


                #evaluate flow parameters
                self.network.eval()
                #calculate flow validation loss
                val_loss_f = - self.network.log_prob(x_val, conditional=x_conditional_val).mean().numpy()
                total_val_loss=val_loss_f + val_loss_g
                self.history['val'].append(total_val_loss)
                self.history['valf'].append(val_loss_f)
                self.history['valg'].append(val_loss_g) #save the loss value of the training data

            #calculate average KL over all params (in latent space with KDEs)
            self.KDE_points, KL_vals = self.latent_KL(x_val,x_conditional_val, self.no_params)
            avg_kl = (1/self.no_params)*(np.sum(KL_vals))
            self.history['kl'].append(avg_kl)

            #print history
            sys.stdout.write(
                    '\r Epoch: {} || Training loss: {} || Validation loss: {} ||KL_div = {}'.format(
                    n+1, train_loss, total_val_loss, avg_kl))

            #copy the best flow model 
            if total_val_loss < best_val_loss:
                best_epoch = n
                best_val_loss = total_val_loss
                best_model = copy.deepcopy(self.network.state_dict())

        #save best model
        print(f'\n Best epoch: {best_epoch}')
        self.network.load_state_dict(best_model)
        torch.save(best_model, filename)

    def latent_KL(self, data, conditional, no_params):
        """
        Calculate KL of data in the latent space given conditionals
        for all binary params, calling KL_evaluate which calculates entropy between KDEs

        Parameters
        ----------
        data : tensor
        conditional : tensor
        no_params : int
            number of parameters in dataspace (binary parameters)
        
        Returns
        -------
        kde_points : array
            defines PDF of latent distribution of the 4 parameters, of shape [1000 x no_params]
        kl_vals : array
            KLs for each dimension, of shape [no_params]
        """
        with torch.no_grad():
            #get data samples in latent space
            z, _ = self.network.forward(data, conditional=conditional)
            #torch tensor to numpy
            z = z.cpu().detach().numpy()

        kde_points = np.zeros((1000, no_params))
        kl_vals = np.zeros(no_params)

        for i in range(no_params):
            kde_points[:,i], kl_vals[i] = self.KL_evaluate(z[:,i])
        return(kde_points, kl_vals)

    def KL_evaluate(self, samples):
        """
        Calculate KL of data in compared to gaussian

        Parameters
        ----------
        samples : tensor
        
        Returns
        -------
        pdf_samples : array
            defines PDF of latent distribution of the 4 parameters, of shape [1000]
        entropy : real
            KL between for pdf_samples and gaussian (sum(p_k * log(p_k / q_k)))
        """
        #pdf of standard Gaussian
        g = np.linspace(-5, 5, 1000)
        self.gaussian = norm.pdf(g, loc = 0, scale = 1)

        #calculates KDE of samples and stores as pdf_samples
        def pdf_evaluate(samples):
            density = gaussian_kde(samples)
            kde_points = density.pdf(g)
            return np.array(kde_points)
        pdf_samples = pdf_evaluate(samples) 

        return pdf_samples, entropy(pdf_samples, self.gaussian)

    def KL_non_gaussian(self,p_samples,q_samples):
        """
        Calculate KL of 2 non-gaussian datasets

        Parameters
        ----------
        p_samples : tensor
        q_samples : tensor
        
        Returns
        -------
        g : array
            linspace on which samples are defined shape [1000]
        p : array
            pdf of p_sampels
        q : array
            pdf of q_samples
        forward_KL : real
            KL between for p and q
        backward_KL : real
            KL between q and p
        """

        g = np.linspace(np.min(p_samples.numpy()), np.max(p_samples.numpy()), 1000)

        def pdf_evaluate(samples):
            density = gaussian_kde(samples)
            kde_points = density.pdf(g)
            return np.array(kde_points)
        
        #calculate pdfs of data over range g
        p = pdf_evaluate(p_samples)
        q = pdf_evaluate(q_samples)

        #calculate entropy between PDFs where pdfs are both >0
        forward_KL = entropy(p[np.logical_and(p>sys.float_info.min, q>sys.float_info.min)], q[
            np.logical_and(p>sys.float_info.min, q>sys.float_info.min)])
        backward_KL = entropy(q[np.logical_and(p>sys.float_info.min, q>sys.float_info.min)], p[
            np.logical_and(p>sys.float_info.min, q>sys.float_info.min)])

        return(g,p,q,forward_KL, backward_KL)

    def plot_history(self, filename=None):
        """
        Plots losses, KL, and latent space

        Parameters
        ----------
        filename : str
            where to save history values
        """

        #loss plot
        param = ['mchirp','q', 'chieff', 'z']
        fig, ax = plt.subplots(figsize = (10,5))
        ax.plot(self.history['train'][5:], label = 'Train loss')
        ax.plot(self.history['val'][5:], label = 'Validation loss')
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epochs')
        ax.legend(loc = 'best')

        #inset log plot
        axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
        trainloss = np.asarray(self.history['train'][:])
        axins.plot(trainloss, label = 'Train loss')#-np.min(trainloss)
        valloss = np.asarray(self.history['val'][:])
        axins.plot(valloss, label = 'Validation loss')#-np.min(trainloss)
        axins.set_xscale('log')
        #axins.set_yscale('log')
        plt.show()

        #KL plot
        fig, ax = plt.subplots(figsize = (10,5))
        plt.plot(self.history['kl'],'-g', label = 'KL')
        plt.ylabel('KL')
        plt.xlabel('Epochs')
        plt.legend(loc = 'best')
        plt.show()

        #latent space plot
        fig, ax = plt.subplots(figsize = (10,5))
        g = np.linspace(-5, 5, 1000)
        for i in range(self.no_params):
            plt.plot(g, self.KDE_points[:,i], label = param[i])
        plt.plot(g, self.gaussian,'k', label = 'Gaussian')
        plt.ylabel('p(z)')
        plt.xlabel('z')
        plt.legend(loc = 'best')
        plt.show()

        pd.DataFrame.to_csv(self.history,f'{filename}_means.csv')

    def get_samples(self, no_samples, hyperparameters):
        """
        Pull samples from flow given set of population hyperparameters

        Parameters
        ----------
        no_samples : int
            number of samples to take for each conditional
        hyperparameters : array
            [number hypereparameter pairs x no_conditionals]
        
        Returns
        -------
        array 
            [np.shape(hyperparameters)[0], no_samples, self.no_params]
        """
        generated_stack = np.zeros((np.shape(hyperparameters)[0], no_samples, self.no_params))

        with torch.no_grad():
            for i, new_hyperparams in enumerate(hyperparameters):
                conditional = torch.from_numpy(new_hyperparams.astype(np.float32))
                conditional = conditional.tile(no_samples,1)
                generated_stack[i,:,:] = self.network.sample(no_samples, conditional=conditional)

        return(generated_stack)

    def easy_sample(self, no_samples, conditional):
        """
        Pull samples from flow given one pair of population hyperparameters

        Parameters
        ----------
        no_samples : int
            number of samples to take for each conditional
        coditional : array
            [chi_b, alpha]
        
        Returns
        -------
        array 
            [no_samples, self.no_params]
        """
        samples = np.zeros((no_samples, self.no_params))

        with torch.no_grad():
            conditional = torch.from_numpy(conditional.astype(np.float32))
            #tile as many conditional chi_b alpha pairs as no samples
            conditional = conditional.tile(no_samples,1)
            samples = self.network.sample(no_samples, conditional=conditional)

        return(samples)

    def get_training_data(self, training_samples):
        """
        Get random batch training data from self.training_samples
        
        Returns
        -------
        xdata : tensor 
            [no_samples, self.no_params]
        x_hyperparams : tensor

        """
        #treatment of separation of training and validation data is different for 2d CE channel than 1d channels
        #differentiated by size of conditional inputs
        #2D channel has seperate populations for training and validation data, 1D mixes up samples
        if self.cond_inputs >=2:
            random_samples = np.random.choice((self.total_hps-2)
                    *self.no_binaries,size=(int(self.batch_size*0.8)))
            batched_hp_pairs = training_samples[random_samples,-2:]
        else:
            random_samples = np.random.choice(self.no_binaries,size=(int(self.batch_size*0.8)))
            batched_hp_pairs = training_samples[random_samples, -1]

        batched_samples = training_samples[random_samples,:(self.no_params)]
        batch_weights = training_samples[random_samples,-3]

        #reshape tensors
        xdata=torch.from_numpy(batched_samples.astype(np.float32))
        #xhyperparams = np.concatenate(batched_hp_pairs)
        xhyperparams = torch.from_numpy(batched_hp_pairs.astype(np.float32))
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(batch_weights.astype(np.float32))

        return(xdata, xhyperparams,xweights)

    def get_val_data(self, validation_data):
        """
        Get random batch validation data from self.validation_data
        
        Returns
        -------
        xdata : tensor 
            [no_samples, self.no_params]
        x_hyperparams : tensor

        """
        if self.cond_inputs >=2:
            random_samples = np.random.choice(2*self.no_binaries, size=(int(self.batch_size*0.2)))
            validation_hp_pairs = validation_data[random_samples,-2:]
        else:
            random_samples = np.random.choice(self.no_binaries,size=(int(self.batch_size*0.2)))
            validation_hp_pairs = validation_data[random_samples,-1]

        validation_samples = validation_data[random_samples,:(self.no_params)]
        val_weights = validation_data[random_samples,-3]

        #reshape
        xval=torch.from_numpy(validation_samples.astype(np.float32))
        xhyperparams = torch.from_numpy(validation_hp_pairs.astype(np.float32)) 
        xhyperparams = xhyperparams.reshape(-1,self.cond_inputs)
        xweights = torch.from_numpy(val_weights.astype(np.float32)) 
        return(xval, xhyperparams, xweights)

    def load_model(self,filename):
        """
        Load pre-trained flow from saved model
        """
        self.network.load_state_dict(torch.load(filename))
        self.network.eval()

    def get_logprob(self, sample, conditionals):
        """
        get log_prob given a sample of [mchirp,q,chieff,z] given conditional hyperparameters
        """
        #make sure samples in right format
        sample = torch.from_numpy(sample.astype(np.float32))
        hyperparams = torch.from_numpy(conditionals.astype(np.float32))
        hyperparams = hyperparams.reshape(-1,self.cond_inputs)
        sample = sample.reshape(-1,4)
        return self.network.log_prob(sample, hyperparams)

    #need __call__, is this what get_log_prob is going to be?
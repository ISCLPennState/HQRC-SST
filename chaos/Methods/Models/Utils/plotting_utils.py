#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
	Created by: Vlachas Pantelis, CSE-lab, ETH Zurich
	Adapted to Higher-order quantum reservoir computing by Quoc Hoan Tran

	Implemented in the framework created by Vlachas Pantelis, CSE-lab, ETH Zurich
        https://github.com/pvlachas/RNN-RC-Chaos
        [1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., 
        Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks 
        for the forecasting of complex spatiotemporal dynamics. Neural Networks (2020), 
        doi: https://doi.org/10.1016/j.neunet.2020.02.016.
"""
#!/usr/bin/env python
import numpy as np
import socket

# Plotting parameters
import matplotlib
hostname = socket.gethostname()
print("PLOTTING HOSTNAME: {:}".format(hostname))
CLUSTER = True if ((hostname[:2]=='eu')  or (hostname[:5]=='daint') or (hostname[:3]=='nid')) else False
if CLUSTER: matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib  import cm
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import colors
import six
color_dict = dict(six.iteritems(colors.cnames))

font = {'size'   : 16, 'family':'Times New Roman'}
matplotlib.rc('font', **font)


def plotTrainingLosses(model, loss_train, loss_val, min_val_error,additional_str=""):
    if (len(loss_train) != 0) and (len(loss_val) != 0):
        min_val_epoch = np.argmin(np.abs(np.array(loss_val)-min_val_error))
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), loss_train, color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), loss_val, color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, min_val_error, "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

        fig_path = model.saving_path + model.fig_dir + model.model_name + "/Loss_total_log"+ additional_str + ".png"
        fig, ax = plt.subplots()
        plt.title("Validation error {:.10f}".format(min_val_error))
        plt.plot(np.arange(np.shape(loss_train)[0]), np.log(loss_train), color=color_dict['green'], label="Train RMSE")
        plt.plot(np.arange(np.shape(loss_val)[0]), np.log(loss_val), color=color_dict['blue'], label="Validation RMSE")
        plt.plot(min_val_epoch, np.log(min_val_error), "o", color=color_dict['red'], label="optimal")
        ax.set_xlabel(r"Epoch")
        ax.set_ylabel(r"Log-Loss")
        plt.legend()
        plt.savefig(fig_path)
        plt.close()

    else:
        print("## Empty losses. Not printing... ##")



def plotAttractor(model, set_name, latent_states, ic_idx):

    print(np.shape(latent_states))
    if np.shape(latent_states)[1] >= 2:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        X = latent_states[:, 0]
        Y = latent_states[:, 1]
        epsilon = 1e-7
        # for i in range(0, len(X)-1):
        for i in range(len(X)-1):
            if np.abs(X[i+1]-X[i]) > epsilon and np.abs(Y[i+1]-Y[i]) > epsilon:
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', lw=0, length_includes_head=True, zorder=2, linestyle='')
                plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', head_width=.05, shape='full', length_includes_head=True, zorder=2)
                # plt.arrow(X[i], Y[i], X[i+1]-X[i], Y[i+1]-Y[i], color='red', shape='full', zorder=2)
        plt.plot(X, Y, 'k', linewidth = 1, label='output', zorder=1)
        plt.autoscale(enable=True, axis='both')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()
    else:
        fig, ax = plt.subplots()
        plt.title("Latent dynamics in {:}".format(set_name))
        plt.plot(latent_states[:-1, 0], latent_states[1:, 0], 'b', linewidth = 2.0, label='output')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/lattent_dynamics_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path, dpi=300)
        plt.close()

def scaleData(input_sequence):
		# data_mean = np.mean(train_input_sequence,0)
		# data_std = np.std(train_input_sequence,0)
		# train_input_sequence = (train_input_sequence-data_mean)/data_std
        data_mean = np.mean(input_sequence,0)
        data_std = np.std(input_sequence,0)
        data_min = np.min(input_sequence,0)
        data_max = np.max(input_sequence,0)
        input_sequence = np.array((input_sequence-data_min)/(data_max-data_min))
        return input_sequence


def plotIterativePrediction(model, set_name, target, prediction, error, nerror, ic_idx, dt, truth_augment=None, prediction_augment=None, warm_up=None, latent_states=None):


    if latent_states is not None:
        plotAttractor(model, set_name, latent_states, ic_idx)

    if ((truth_augment is not None) and (prediction_augment is not None)):
        normalised_pred_augment = scaleData(prediction_augment)
        normalised_truth_augment = scaleData(truth_augment)
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_augmend_{:}_{:}.png".format(set_name, ic_idx)
        #plt.plot(np.arange(np.shape(prediction_augment)[0]), prediction_augment[:,0], 'b', linewidth = 2.0, label='output')
        plt.plot(np.arange(warm_up,np.shape(normalised_pred_augment)[0]), normalised_pred_augment[warm_up:,0], 'b', linewidth = 2.0, label='output')
        plt.plot(np.arange(np.shape(normalised_truth_augment)[0]), normalised_truth_augment[:,0], 'r', linewidth = 2.0, label='target')
        #plt.plot(np.ones((100,1))*warm_up, np.linspace(np.min(truth_augment[:,0]), np.max(truth_augment[:,0]), 100), 'g--', linewidth = 2.0, label='warm-up')
        plt.plot(np.arange(0,warm_up), normalised_pred_augment[:warm_up,0], 'g--', linewidth = 2.0, label='warm-up')
        plt.legend(loc="lower right")
        plt.xlabel("Time")
        plt.ylabel("Dimension 1")
        plt.ylim(0,1)
        plt.title("warm up + prediction")
        plt.savefig(fig_path)
        plt.close()

    temp_target = target
    temp_out = prediction
    num_dimensions = temp_target.shape[1]
    # Create subplots
    fig, axs = plt.subplots(num_dimensions, 1, figsize=(10, 10))
    # Iterate over each dimension
    for dim in range(num_dimensions):
        # Normalize target and output arrays for current dimension
        target_dim = temp_target[:, dim]
        out_dim = temp_out[:, dim]
        
        norm_target_dim = scaleData(target_dim)
        norm_out_dim = scaleData(out_dim)
        err_dim = np.abs(norm_out_dim-norm_target_dim)
        # Plot target and output arrays
        axs[dim].plot(norm_target_dim, label='Target')
        axs[dim].plot(norm_out_dim, label='Prediction')
        axs[dim].fill_between(range(len(norm_out_dim)), norm_out_dim - err_dim, norm_out_dim + err_dim, alpha=0.2, label='Error')
        
        # Set y-axis limits to [0, 1]
        axs[dim].set_ylim(0, 1)
        
        # Add labels and legend
        axs[dim].set_xlabel('Time')
        axs[dim].set_ylabel(f'Dimension {dim+1}')
        axs[dim].legend()

    # Add title
    plt.suptitle('Target and Prediction for Each Dimension')
    # Adjust layout
    plt.tight_layout()
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/predictions_all_dim_{:}_{:}.png".format(set_name, ic_idx)
    plt.savefig(fig_path)
    plt.close()
    # Plot frequency
    #plt.show()
    
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}.png".format(set_name, ic_idx)
    plt.plot(prediction, 'r--', label='prediction')
    plt.plot(target, 'g--', label='target')
    plt.legend(loc="lower right")
    plt.xlabel("Time")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_error.png".format(set_name, ic_idx)
    plt.plot(error, label='error')
    plt.legend(loc="lower right")
    plt.xlabel("Time")
    plt.ylabel("RMSE error")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_log_error.png".format(set_name, ic_idx)
    plt.plot(np.log(np.arange(np.shape(error)[0])), np.log(error), label='log(error)')
    plt.legend(loc="lower right")
    plt.xlabel("Time")
    plt.ylabel("log RMSE error")
    plt.savefig(fig_path)
    plt.close()

    fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_nerror.png".format(set_name, ic_idx)
    plt.plot(nerror, label='normalised error')
    plt.legend(loc="lower right")
    plt.xlabel("Time")
    plt.ylabel("Normalised RMSE error")
    plt.savefig(fig_path)
    plt.close()

    if model.input_dim >=3: 
        createTestingContours(model, target, prediction, dt, ic_idx, set_name)
        # Extract the first three dimensions
        target_3d = temp_target[:, 2:]
        out_3d = temp_out[:, 2:]

        # Create a 3D plot
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the first three dimensions of temp_target and temp_out
        ax.plot(target_3d[:, 0], target_3d[:, 1], target_3d[:, 2], label='Target')
        ax.plot(out_3d[:, 0], out_3d[:, 1], out_3d[:, 2], label='Prediction')

        # Set labels and legend
        ax.set_xlabel('Dimension 3')
        ax.set_ylabel('Dimension 4')
        ax.set_zlabel('Dimension 5')
        ax.legend()

        # Set plot title
        plt.title('First Three Dimensions: Target vs Prediction')
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_3D_{:}_{:}.png".format(set_name, ic_idx)
        plt.savefig(fig_path)
        plt.close()


def createTestingContours(model, target, output, dt, ic_idx, set_name):
    fontsize = 12
    error = np.abs(target-output)
    # vmin = np.array([target.min(), output.min()]).min()
    # vmax = np.array([target.max(), output.max()]).max()
    vmin = target.min()
    vmax = target.max()
    vmin_error = 0.0
    vmax_error = target.max()

    print("VMIN: {:} \nVMAX: {:} \n".format(vmin, vmax))

    # Plotting the contour plot
    fig, axes = plt.subplots(nrows=1, ncols=3,figsize=(12, 6), sharey=True)
    fig.subplots_adjust(hspace=0.4, wspace = 0.4)
    axes[0].set_ylabel(r"Time $t$", fontsize=fontsize)
    createContour_(fig, axes[0], target, "Target", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[1], output, "Output", fontsize, vmin, vmax, plt.get_cmap("seismic"), dt)
    createContour_(fig, axes[2], error, "Error", fontsize, vmin_error, vmax_error, plt.get_cmap("Reds"), dt)
    for ftype in ['png']:
        fig_path = model.saving_path + model.fig_dir + model.model_name + "/prediction_{:}_{:}_contour.{}".format(set_name, ic_idx, ftype)
        plt.savefig(fig_path)
    plt.close()

def createContour_(fig, ax, data, title, fontsize, vmin, vmax, cmap, dt):
    ax.set_title(title, fontsize=fontsize)
    t, s = np.meshgrid(np.arange(data.shape[0])*dt, np.arange(data.shape[1]))
    mp = ax.contourf(s, t, np.transpose(data), 15, cmap=cmap, levels=np.linspace(vmin, vmax, 60), extend="both")
    fig.colorbar(mp, ax=ax)
    ax.set_xlabel(r"$State$", fontsize=fontsize)
    return mp

def plotSpectrum(model, sp_true, sp_pred, freq_true, freq_pred, set_name):
    fig_path = model.saving_path + model.fig_dir + model.model_name + "/frequencies_{:}.png".format(set_name)
    plt.plot(freq_pred, sp_pred, 'r--', label="prediction")
    plt.plot(freq_true, sp_true, 'g--', label="target")
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power Spectrum [dB]')
    plt.legend(loc="lower right")
    plt.savefig(fig_path)
    plt.close()







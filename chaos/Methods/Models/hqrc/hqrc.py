#!/usr/bin/env python
# # -*- coding: utf-8 -*-

"""
Created by: Quoc Hoan Tran

    Implemented in the framework created by Vlachas Pantelis, CSE-lab, ETH Zurich
        https://github.com/pvlachas/RNN-RC-Chaos
        [1] P.R. Vlachas, J. Pathak, B.R. Hunt et al., 
        Backpropagation algorithms and Reservoir Computing in Recurrent Neural Networks 
        for the forecasting of complex spatiotemporal dynamics. Neural Networks (2020), 
        doi: https://doi.org/10.1016/j.neunet.2020.02.016.
"""
#!/usr/bin/env python
import numpy as np
import pickle
import scipy as sp 
from scipy import sparse as sparse
from scipy.sparse import linalg as splinalg
from scipy.linalg import pinv2 as scipypinv2
# from scipy.linalg import lstsq as scipylstsq
# from numpy.linalg import lstsq as numpylstsq
import os
import sys
#sys.path.insert(1, '../Utils/')

from plotting_utils import *
from global_utils import *
from qrc_utils import *

import pickle
import time
from functools import partial
print = partial(print, flush=True)

from sklearn.linear_model import Ridge
from scipy.special import expit
from sklearn.utils import shuffle

# MEMORY TRACKING
import psutil


class hqrc(object):
    def delete(self):
        return 0
    
    def __init__(self, params):
        self.display_output = params["display_output"]
        print("RANDOM SEED: {:}".format(params["worker_id"]))
        np.random.seed(params["worker_id"])
        
        self.worker_id = params["worker_id"]
        self.input_dim = params["RDIM"] # energy for SVD model
        self.N_used = params["N_used"]
        
        # Parameters for high-order model
        self.nqrc = params["nqrc"]
        self.gamma = params["gamma"]
        self.alpha = params["alpha"]
        self.max_energy = params["max_energy"]
        self.dynamic = params["dynamic"]
        self.non_diag_var = params["non_diag_var"]
        self.non_diag_const = params["non_diag_const"]
        self.nonlinear = params["nonlinear"]

        self.virtual_nodes = params["virtual_nodes"]
        self.tau = params["tau"]
        self.type_input = params["type_input"]
        self.scale_input = params["scale_input"]
        self.trans_input = params["trans_input"]
        self.bias = params["bias"]
        self.use_corr = params["use_corr"]
        self.type_op = params["type_op"]
        self.type_connect = params["type_connect"] # type of connection: 0 (full higher-order), 
        # 1 (deep, only input in first qr, feedback for only previous qr)
        

        self.n_units = params["n_units"]
        self.n_envs = params["n_envs"]
        self.n_qubits = self.n_units + self.n_envs
        self.dim = 2**self.n_qubits
        # Finish

        self.dynamics_length = params["dynamics_length"]
        self.it_pred_length = params["it_pred_length"]
        self.iterative_update_length = params["iterative_update_length"]
        self.n_tests = params["n_tests"]
        self.train_data_path = params["train_data_path"]
        self.test_data_path = params["test_data_path"]
        self.fig_dir = params["fig_dir"]
        self.model_dir = params["model_dir"]
        self.logfile_dir = params["logfile_dir"]
        self.write_to_log = params["write_to_log"]
        self.results_dir = params["results_dir"]
        self.saving_path = params["saving_path"]
        self.reg = params["reg"]
        self.scaler_tt = params["scaler"]
        self.scaler_trans = params["trans"]
        self.scaler_ratio = params["ratio"]

        # Parameters for optimizers
        self.learning_rate = params["learning_rate"]
        self.number_of_epochs = params["number_of_epochs"]
        self.solver = str(params["solver"])
        self.norm_every = params["norm_every"]
        self.augment =  params["augment"]

        ##########################################
        self.scaler = scaler(self.scaler_tt, self.scaler_trans, self.scaler_ratio)
        self.noise_level = params["noise_level"]
        self.model_name = self.createModelName(params)

        self.reference_train_time = 60*60*(params["reference_train_time"]-params["buffer_train_time"])
        print("Reference train time {:} seconds / {:} minutes / {:} hours.".format(self.reference_train_time, self.reference_train_time/60, self.reference_train_time/60/60))

        os.makedirs(self.saving_path + self.model_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.fig_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.results_dir + self.model_name, exist_ok=True)
        os.makedirs(self.saving_path + self.logfile_dir + self.model_name, exist_ok=True)

    def __init_reservoir(self):
        I = [[1,0],[0,1]]
        X = [[0,1],[1,0]]
        Y = [[0,-1.j],[1.j,0]]
        Z = [[1,0],[0,-1]]
        P0 = [[1,0],[0,0]]
        P1 = [[0,0],[0,1]]

        # Create W_feed
        n_nodes = self.__get_comput_nodes()
        n_local_nodes = self.__get_qr_nodes()
        W_feed = np.zeros((n_nodes, self.nqrc))
        if self.type_connect == 0:
            W_feed = np.random.uniform(0.0, 1.0, size=(n_nodes, self.nqrc))
        elif self.type_connect == 1:
            feed_mat = np.random.uniform(0.0, 1.0, size=(n_nodes, self.nqrc))
            for i in range(self.nqrc-1):
                bg = (i+1) * n_local_nodes
                ed = (i+2) * n_local_nodes
                W_feed[bg:ed, i] = feed_mat[bg:ed, i]
        self.W_feed = W_feed

        # Create operators from tensor product
        self.Xop = [1]*self.n_qubits
        self.Xop_corr = dict()

        self.Yop = [1]*self.n_qubits
        self.Yop_corr = dict()

        self.Zop = [1]*self.n_qubits
        self.Zop_corr = dict()
        
        self.P0op = [1]
        self.P1op = [1]

        for q1 in range(self.n_qubits):
            for q2 in range(q1+1, self.n_qubits):
                self.Xop_corr[(q1, q2)] = [1]
                self.Yop_corr[(q1, q2)] = [1]
                self.Zop_corr[(q1, q2)] = [1]

        for cindex in range(self.n_qubits):
            for qindex in range(self.n_qubits):
                if cindex == qindex:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],X)
                    self.Yop[qindex] = np.kron(self.Yop[qindex],Y)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],Z)
                else:
                    self.Xop[qindex] = np.kron(self.Xop[qindex],I)
                    self.Yop[qindex] = np.kron(self.Yop[qindex],I)
                    self.Zop[qindex] = np.kron(self.Zop[qindex],I)

            if cindex == 0:
                self.P0op = np.kron(self.P0op, P0)
                self.P1op = np.kron(self.P1op, P1)
            else:
                self.P0op = np.kron(self.P0op, I)
                self.P1op = np.kron(self.P1op, I)

        # generate correlatior operators
        if self.use_corr > 0:
            for q1 in range(self.n_qubits):
                for q2 in range(q1+1, self.n_qubits):
                    cindex = (q1, q2)
                    for qindex in range(self.n_qubits):
                        if qindex == q1 or qindex == q2:
                            self.Xop_corr[cindex] = np.kron(self.Xop_corr[cindex], X)
                            self.Yop_corr[cindex] = np.kron(self.Yop_corr[cindex], Y)
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], Z)
                        else:
                            self.Xop_corr[cindex] = np.kron(self.Xop_corr[cindex], I)
                            self.Yop_corr[cindex] = np.kron(self.Yop_corr[cindex], I)
                            self.Zop_corr[cindex] = np.kron(self.Zop_corr[cindex], I)
                            
        if self.type_op == 'X':
            self.Pauli_op = self.Xop
            self.Pauli_op_corr = self.Xop_corr
        elif self.type_op == 'Y':
            self.Pauli_op = self.Yop
            self.Pauli_op_corr = self.Yop_corr
        else:
            self.Pauli_op = self.Zop
            self.Pauli_op_corr = self.Zop_corr


        # initialize current states
        self.cur_states = [None] * self.nqrc
        # initialize feedback input
        self.feed_inputs = [0]  * self.nqrc

        # create coupling strength for ion trap
        a = self.alpha
        bc = self.non_diag_const
        Nalpha = 0
        for qindex1 in range(self.n_qubits):
            for qindex2 in range(qindex1+1, self.n_qubits):
                Jij = np.abs(qindex2 - qindex1)**(-a)
                Nalpha += Jij / (self.n_qubits-1)
        if bc > 0:
            B = self.max_energy / bc # Magnetic field
        else:
            B = self.max_energy

        # Intialize evolution operators
        tmp_uops = []
        # generate hamiltonian
        for i in range(self.nqrc):
            hamiltonian = np.zeros( (self.dim, self.dim) )

            for qindex in range(self.n_qubits):
                if self.dynamic == DYNAMIC_FULL_RANDOM:
                    coef = (np.random.rand()-0.5) * 2 * self.max_energy
                elif self.dynamic == DYNAMIC_PHASE_TRANS:
                    coef = (np.random.rand()-0.5) * self.non_diag_var + self.non_diag_const
                else:
                    coef = B
                hamiltonian -= coef * self.Zop[qindex]

            for qindex1 in range(self.n_qubits):
                for qindex2 in range(qindex1+1, self.n_qubits):
                    if self.dynamic == DYNAMIC_FULL_CONST_COEFF:
                        coef =  self.max_energy
                    elif self.dynamic == DYNAMIC_ION_TRAP:
                        coef = np.abs(qindex2 - qindex1)**(-a) / Nalpha
                        coef = self.max_energy * coef
                    elif self.dynamic == DYNAMIC_PHASE_TRANS:
                        coef = (np.random.rand()-0.5) * self.max_energy
                    else:
                        coef = (np.random.rand()-0.5) * 2 * self.max_energy
                    hamiltonian -= coef * self.Xop[qindex1] @ self.Xop[qindex2]
                    
            ratio = float(self.tau) / float(self.virtual_nodes)        
            Uop = sp.linalg.expm(-1.j * hamiltonian * ratio)
            tmp_uops.append(Uop)
        self.Uops = tmp_uops.copy()

        # initialize density matrix
        tmp_rhos = generate_list_rho(self.dim, self.nqrc, rand_rho=True)
        self.init_rhos = tmp_rhos.copy()
        self.last_rhos = tmp_rhos.copy()

    def __get_qr_nodes(self):
        if self.use_corr > 0:
            qrnodes = int((self.n_qubits * (self.n_qubits + 1)) / 2)
        else:
            qrnodes = self.n_qubits
        qrnodes = qrnodes * self.virtual_nodes
        return qrnodes

    def __get_comput_nodes(self):
        return self.__get_qr_nodes() * self.nqrc
    
    def __reset_states(self):
        self.cur_states = [None] * self.nqrc

    def getKeysInModelName(self):
        keys = {
        'RDIM':'RDIM', 
        'N_used':'N_used', 
        'dynamics_length':'DL',
        'nqrc':'Nqr',
        'gamma':'G',
        #'trans':'sT',
        #'ratio':'sR',
        #'scale_input':'sI',
        'max_energy':'J',
        'virtual_nodes':'V',
        #'tau':'TAU',
        #'n_units':'UNIT',
        #'bias':'B',
        'noise_level':'NL',
        'it_pred_length':'IPL',
        'iterative_update_length':'IUL',
        'reg':'REG',
        #'scaler':'SC',
        #'norm_every':'NE',
        'augment':'AU',
        'n_tests':'NICS',
        #'worker_id':'WID', 
        }
        return keys
    
    def createModelName(self, params):
        keys = self.getKeysInModelName()
        str_ = "hqrc_" + self.solver
        for key in keys:
            str_ += "-" + keys[key] + "_{:}".format(params[key])
        return str_

    def __step_forward(self, local_rhos, input_val):
        nqrc = self.nqrc
        n_nodes = self.__get_comput_nodes()
        n_local_nodes = self.__get_qr_nodes()

        original_input = input_val.copy().ravel()
        
        q0 = np.array([1, 0]).reshape((2, 1))
        q1 = np.array([0, 1]).reshape((2, 1))

        if self.cur_states[0] is None:
            update_input = original_input
            self.feed_inputs = original_input * 0.0
        else:
            tmp_states = np.array(self.cur_states.copy(), dtype=np.float64).reshape(1, -1)
            tmp_states = (tmp_states + 1.0) / 2.0
            tmp_states = tmp_states @ self.W_feed
            tmp_states = np.ravel(tmp_states)
            
            if self.nonlinear == 1:
                tmp_states = expit(tmp_states)
            elif self.nonlinear == 2:
                # Min-max norm
                tmp_states = (tmp_states - np.min(tmp_states)) / (np.max(tmp_states) - np.min(tmp_states))
            elif self.nonlinear == 3:
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 4:
                tmp_states = expit(tmp_states)
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 5:
                # Min-max norm
                tmp_states = (tmp_states - np.min(tmp_states)) / (np.max(tmp_states) - np.min(tmp_states))
                tmp_states = shuffle(tmp_states)
            elif self.nonlinear == 6:
                tmp_states = [np.modf(x / (2*np.pi))[0] for x in tmp_states]
                # to make sure the nonegative number
                tmp_states = np.array([np.modf(x + 1.0)[0] for x in tmp_states])
            elif self.nonlinear == 7:
                tmp_states = [np.modf(x)[0] for x in tmp_states]
                tmp_states = np.array([np.modf(x + 1.0)[0] for x in tmp_states])
            #print(tmp_states, self.feed_min, self.feed_max)
            self.feed_inputs = tmp_states.copy().ravel()
            
            # normalize feed_inputs by dividing to the number of computational nodes
            if self.type_connect == 0:
                self.feed_inputs = self.feed_inputs / n_nodes
            elif self.type_connect == 1:
                self.feed_inputs = self.feed_inputs / n_local_nodes
            
            tmp_states[tmp_states < 0.0] = 0.0
            tmp_states[tmp_states > 1.0] = 1.0
           
        if True:
            for i in range(nqrc):
                Uop = self.Uops[i]
                rho = local_rhos[i]
                value = original_input[i]

                # Replace the density matrix
                # rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                # (1 + u Z)/2 = (1+u)/2 |0><0| + (1-u)/2 |1><1|
                # inv1 = (self.affine[1] + self.value) / self.affine[0]
                # inv2 = (self.affine[1] - self.value) / self.affine[0]

                if self.type_input == 0:
                    rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                    # for input in [0, 1]
                    value = clipping(value, minval=0.0, maxval=1.0)
                    rho = (1 - value) * rho + value * self.Xop[0] @ rho @ self.Xop[0]
                elif self.type_input == 1:
                    value = clipping(value, minval=-1.0, maxval=1.0)
                    rho = self.P0op @ rho @ self.P0op + self.Xop[0] @ self.P1op @ rho @ self.P1op @ self.Xop[0]
                    # for input in [-1, 1]
                    rho = ((1+value)/2) * rho + ((1-value)/2) *self.Xop[0] @ rho @ self.Xop[0]
                else:
                    value = clipping(value, minval=0.0, maxval=1.0)
                    par_rho = partial_trace(rho, keep=[1], dims=[2**self.n_envs, 2**self.n_units], optimize=False)

                    if self.type_input == 2:
                        input_state = np.sqrt(1-value) * q0 + np.sqrt(value) * q1
                    elif self.type_input == 3:
                        angle_val = 2*np.pi*value
                        input_state = np.cos(angle_val) * q0 + np.sin(angle_val) * q1
                    elif self.type_input == 4:
                        input_state = np.sqrt(1-value) * q0 + np.sqrt(value) * np.exp(1.j * 2*np.pi*value) * q1
                    else:
                        orig_contrib = clipping(original_input[i], minval=0.0, maxval=1.0)
                        if self.type_input == 5:
                            update_contrib = self.gamma * self.feed_inputs[i]
                        elif self.type_input == 6:
                            update_contrib = self.gamma * self.feed_inputs[i] + (1.0 - self.gamma) * orig_contrib
                        elif self.type_input == 7:
                            feed_contrib = 0.5 + np.arctan(self.feed_inputs[i]) / np.pi
                            update_contrib = self.gamma * feed_contrib + (1.0 - self.gamma) * orig_contrib
                        elif self.type_input == 8:
                            update_contrib = self.gamma * orig_contrib
                        elif self.type_input == 9:
                            orig_contrib =  0.5 # let this part become 0.5 (should be a parameter 8/19)
                            update_contrib = original_input[i] + self.gamma * self.feed_inputs[i]
                        else:
                            update_contrib = self.gamma

                        input_state = np.sqrt(1-orig_contrib) * q0 + np.sqrt(orig_contrib) * np.exp(1.j * 2*np.pi*update_contrib) * q1
                    
                    input_state = input_state @ input_state.T.conj() 
                    rho = np.kron(input_state, par_rho)


                current_state = []
                for v in range(self.virtual_nodes):
                    # Time evolution of density matrix
                    rho = Uop @ rho @ Uop.T.conj()
                    for qindex in range(0, self.n_qubits):
                        rvstate = np.real(np.trace(self.Pauli_op[qindex] @ rho))
                        current_state.append(rvstate)
                    
                    if self.use_corr > 0:
                        for q1 in range(0, self.n_qubits):
                            for q2 in range(q1+1, self.n_qubits):
                                cindex = (q1, q2)
                                rvstate = np.real(np.trace(self.Pauli_op_corr[cindex] @ rho))
                                current_state.append(rvstate)

                # Size of current_state is Nqubits x Nvirtuals)
                self.cur_states[i] = np.array(current_state, dtype=np.float64)
                local_rhos[i] = rho
        return local_rhos

    def __feed_forward(self, input_seq, predict, use_lastrho):
        input_length, input_dim = input_seq.shape
        print('Input length={}, dim={}'.format(input_length, input_dim))
        
        assert(input_dim == self.nqrc)
        
        predict_seq = None
        local_rhos = self.init_rhos.copy()
        if use_lastrho == True :
            local_rhos = self.last_rhos.copy()
        
        state_list, feed_list = [], []
        for time_step in range(0, input_length):
            input_val = np.ravel(input_seq[time_step])
            local_rhos = self.__step_forward(local_rhos, input_val)

            state = np.array(self.cur_states.copy(), dtype=np.float64)
            state_list.append(state.flatten())
            feed_list.append(self.feed_inputs)

        state_list = np.array(state_list, dtype=np.float64)
        feed_list  = np.array(feed_list)
        self.last_rhos = local_rhos.copy()

        if predict:
            aug_state_list = state_list.copy()
            if self.augment > 0:
                print('Augment data')
                aug_state_list = self.augmentHiddenList(aug_state_list)
                aug_state_list = np.array(aug_state_list, dtype=np.float64)
            
            stacked_state = np.hstack( [aug_state_list, np.ones([input_length, 1])])
            #print('stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
            predict_seq = stacked_state @ self.W_out
        
        return predict_seq, state_list, feed_list

    def __train(self, input_sequence, output_sequence):
        print('Training input, output shape', input_sequence.shape, output_sequence.shape)
        assert(input_sequence.shape[0] == output_sequence.shape[0])
        Nout = output_sequence.shape[1]
        self.W_out = np.random.rand(self.getReservoirSize() + 1, Nout)

        # After washing out, use last density matrix to update
        _, state_list, _ = self.__feed_forward(input_sequence, predict=False, use_lastrho=True)

        state_list = np.array(state_list, dtype=np.float64)
        print('State list shape', state_list.shape)

        print("\nSOLVER used to find W_out: {:}. \n\n".format(self.solver))
        if self.solver == "pinv_naive":
            """
            Learn mapping to S with Penrose Pseudo-Inverse
            No augment data
            """
            X = np.reshape(state_list, [-1, self.getReservoirSize()])
            X = np.hstack( [state_list, np.ones([X.shape[0], 1]) ] )
            Y = np.reshape(output_sequence, [output_sequence.shape[0], -1])
            print('TEACHER FORCING ENDED; direct mapping X Y shape', X.shape, Y.shape)
            W_out = np.linalg.pinv(X, rcond = self.reg) @ Y
        else:
            X, Y = [], []
            # Augment data and using batch normalization
            XTX = np.zeros((self.getReservoirSize() + 1, self.getReservoirSize() + 1))
            XTY = np.zeros((self.getReservoirSize() + 1, output_sequence.shape[1]))
            for t in range(state_list.shape[0]):
                h = state_list[t]
                h_aug = self.augmentHidden(h)
                X.append(h_aug)
                Y.append(output_sequence[t])
                if self.norm_every > 0 and (t+1) % self.norm_every == 0:
                    # Batched approach used in the pinv case
                    X = np.array(X)
                    X = np.hstack( [X, np.ones([X.shape[0], 1])] )
                    Y = np.array(Y)
                    
                    XTX += X.T @ X
                    XTY += X.T @ Y
                    X, Y = [], []
            
            if len(X) != 0:
                # add the reaming batch
                X = np.array(X)
                X = np.hstack( [X, np.ones([X.shape[0], 1])] )
                Y = np.array(Y)
                
                XTX += X.T @ X
                XTY += X.T @ Y

            print('TEACHER FORCING ENDED. shape XTX, XTY',  np.shape(XTX), np.shape(XTY))
                
            if self.solver == "pinv":
                I = np.identity(np.shape(XTX)[1])	
                pinv_ = scipypinv2(XTX + self.reg * I)
                W_out = pinv_ @ XTY
            elif self.solver in ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag"]:
                """
                Learns mapping V to S with Ridge Regression
                """
                ridge = Ridge(alpha=self.reg, fit_intercept=False, normalize=False, copy_X=True, solver=self.solver)
                ridge.fit(XTX, XTY) 
                # ridge.fit(A, B) -> A: n_samples x n_features, B: n_samples x n_targets
                # ridge.coef_ -> ndarray of shape (n_features,) or (n_targets, n_features)
                W_out = np.array(ridge.coef_).reshape((-1, Nout))    
            else:
                raise ValueError("Undefined solver.")

        print('Finalizing weights Wout', W_out.shape)
        self.W_out = W_out
        self.n_trainable_parameters = np.size(self.W_out)
        self.n_model_parameters = np.size(self.Uops[0]) * self.nqrc + np.size(self.W_out)
        print("Number of trainable parameters: {}".format(self.n_trainable_parameters))
        print("Total number of parameters: {}".format(self.n_model_parameters))
        print("SAVING MODEL...")
        self.saveModel()

    def train(self):
        self.start_time = time.time()
        dynamics_length = self.dynamics_length
        input_dim = self.input_dim
        N_used = self.N_used

        with open(self.train_data_path, "rb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"]
            print("Adding noise to the training data. {:} per mille ".format(self.noise_level))
            train_input_sequence = addNoise(train_input_sequence, self.noise_level)
            N_all, dim = np.shape(train_input_sequence)
            if input_dim > dim: raise ValueError("Requested input dimension is wrong.")
            train_input_sequence = train_input_sequence[:N_used, :input_dim]
            dt = data["dt"]
            del data
        print("##Using {:}/{:} dimensions and {:}/{:} samples ##".format(input_dim, dim, N_used, N_all))
        if N_used > N_all: raise ValueError("Not enough samples in the training data.")
        print("SCALING")
        
        # Initialize reservoir
        self.__init_reservoir()
        
        train_input_sequence = self.scaler.scaleData(train_input_sequence)
        N, input_dim = np.shape(train_input_sequence)
        print('Train input sequence shape', train_input_sequence.shape)
        # Replicate intput sequence to fed into high-order machines
        nqrc = self.nqrc
        if int(nqrc) % int(input_dim) != 0:
            ValueError("Number of qrc does not divide input's dimension.")
        K = int(nqrc / input_dim)
        rep_train_input_seq = np.tile(train_input_sequence, (1, K))
        print('Replicate train input sequence shape', train_input_sequence.shape)
        # TRAINING LENGTH
        tl = N - dynamics_length

        print("TRAINING: Dynamics prerun...with layer strength={}".format(self.alpha))
        self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=False, use_lastrho=False)
        print("\n")

        print("TRAINING: Teacher forcing...")
        # Create output
        Y = []
        for t in range(tl - 1):
            target = np.reshape(train_input_sequence[t + dynamics_length + 1], (-1,1))
            Y.append(target[:, 0])
        train_output_sequence = np.array(Y, dtype=np.float64)
        print('train_output_sequence shape', train_output_sequence.shape)
        
        out_length, out_dim = train_output_sequence.shape
        print("TRAINING: Output shape", train_output_sequence.shape)
        print("TEACHER FORCING ENDED.")
        print("\n")
        self.__train(rep_train_input_seq[dynamics_length:(dynamics_length + out_length)], train_output_sequence)

    def isWallTimeLimit(self):
        training_time = time.time() - self.start_time
        if training_time > self.reference_train_time:
            print("## Maximum train time reached. ##")
            return True
        else:
            return False

    def augmentHidden(self, h):
        h_aug = h.copy().ravel()
        h_aug = (1.0 + h_aug) / 2.0
        if self.augment > 0:
            h_aug[::2] = pow(h_aug[::2], 2.0)
        return h_aug

    def augmentHiddenList(self, hs):
        hs_aug = [self.augmentHidden(h) for h in hs]
        return hs_aug

    def getReservoirSize(self): 
        return self.__get_comput_nodes()
    
    def predictSequence(self, input_sequence):
        dynamics_length = self.dynamics_length
        it_pred_length = self.it_pred_length
        iterative_update_length = self.iterative_update_length

        N, input_dim = np.shape(input_sequence)
        print('Shape of predict sequence:', input_sequence.shape)
        # PREDICTION LENGTH
        if N != it_pred_length + dynamics_length: 
            raise ValueError("Error! N != it_pred_length + dynamics_length")
        
        nqrc = self.nqrc
        if int(nqrc) % int(input_dim) != 0:
            ValueError("Number of qrc does not divide input's dimension.")
        K = int(nqrc / input_dim)
        rep_train_input_seq = np.tile(input_sequence, (1, K))

        self.__reset_states()
        prediction_warm_up, state_list = \
            self.__feed_forward(rep_train_input_seq[:dynamics_length], predict=True, use_lastrho=False)
        print("\n")

        target = input_sequence[dynamics_length:]
        prediction = []

        if True:
            print('Closed loop to generate chaotic signals')
            local_rhos = self.last_rhos.copy()
            nqrc = self.nqrc
            for t in range(it_pred_length):
                state = np.array(self.current_states, dtype=np.float64)
                state_aug = self.augmentHidden(state).reshape((1, -1))
                stacked_state = np.hstack( [state_aug, np.ones([1, 1])])
                #print('PREDICT stage: stacked state {}; Wout {}'.format(stacked_state.shape, self.W_out.shape))
                out = stacked_state @ self.W_out
                prediction.append(out)
                out = out.reshape(1, -1)
                #if np.max(np.abs(out)) > 10:
                #    print('out signal, t={}/{}'.format(t, it_pred_length))
                #    print(out)
                # out[out < 0] = 0.0
                # out[out > 1.0] = 1.0
                # if np.isnan(out) == False:
                #     print('out', out)
                if iterative_update_length > 0 and (t+1) % iterative_update_length == 0:
                    intput_val = rep_train_input_seq[t]
                else:
                    input_val = np.tile(out, (1, K))[0]
                local_rhos = self.__step_forward(local_rhos, input_val)
            self.last_rhos = local_rhos.copy()

        print("\n")
        prediction = np.array(prediction, dtype=np.float64).reshape((it_pred_length,-1))
        print('Prediction shape', prediction.shape)
        prediction_warm_up = np.array(prediction_warm_up, dtype=np.float64)
        print('shape prediction and warm_up', prediction.shape, prediction_warm_up.shape)
        target_augment = input_sequence
        prediction_augment = np.concatenate((prediction_warm_up, prediction), axis=0)
        return prediction, target, prediction_augment, target_augment

    def testing(self):
        if self.loadModel() ==  0:
            self.testingOnTrainingSet()
            self.testingOnTestingSet()
            self.saveResults()
        return 0
    
    def testingOnTrainingSet(self):
        n_tests = self.n_tests
        with open(self.test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            dt = data["dt"]
            del data

        with open(self.train_data_path, "rb") as file:
            data = pickle.load(file)
            train_input_sequence = data["train_input_sequence"][:, :self.input_dim]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(train_input_sequence, testing_ic_indexes, dt, "TRAIN")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TRAIN = {:s}".format(var_name, var_name))
        return 0

    def testingOnTestingSet(self):
        n_tests = self.n_tests
        with open(self.test_data_path, "rb") as file:
            data = pickle.load(file)
            testing_ic_indexes = data["testing_ic_indexes"]
            test_input_sequence = data["test_input_sequence"][:, :self.input_dim]
            dt = data["dt"]
            del data
            
        rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred = self.predictIndexes(test_input_sequence, testing_ic_indexes, dt, "TEST")
        
        for var_name in getNamesInterestingVars():
            exec("self.{:s}_TEST = {:s}".format(var_name, var_name))
        return 0

    def predictIndexes(self, input_sequence, ic_indexes, dt, set_name):
        n_tests = self.n_tests
        input_sequence = self.scaler.scaleData(input_sequence, reuse=1)
        predictions_all = []
        truths_all = []
        rmse_all = []
        rmnse_all = []
        num_accurate_pred_005_all = []
        num_accurate_pred_050_all = []
        for ic_num in range(n_tests):
            if self.display_output == True:
                print("IC {:}/{:}, {:2.3f}%".format(ic_num, n_tests, ic_num/n_tests*100))
            ic_idx = ic_indexes[ic_num]
            input_sequence_ic = input_sequence[ic_idx-self.dynamics_length:ic_idx+self.it_pred_length]
            prediction, target, prediction_augment, target_augment = self.predictSequence(input_sequence_ic)
            prediction = self.scaler.descaleData(prediction)
            target = self.scaler.descaleData(target)
            rmse, rmnse, num_accurate_pred_005, num_accurate_pred_050, abserror = computeErrors(target, prediction, self.scaler.data_std)
            predictions_all.append(prediction)
            truths_all.append(target)
            rmse_all.append(rmse)
            rmnse_all.append(rmnse)
            num_accurate_pred_005_all.append(num_accurate_pred_005)
            num_accurate_pred_050_all.append(num_accurate_pred_050)
            # PLOTTING ONLY THE FIRST THREE PREDICTIONS
            if ic_num < 3: plotIterativePrediction(self, set_name, target, prediction, rmse, rmnse, ic_idx, dt, target_augment, prediction_augment, self.dynamics_length)

        predictions_all = np.array(predictions_all, dtype=np.float64)
        truths_all = np.array(truths_all, dtype=np.float64)
        rmse_all = np.array(rmse_all, dtype=np.float64)
        rmnse_all = np.array(rmnse_all, dtype=np.float64)
        num_accurate_pred_005_all = np.array(num_accurate_pred_005_all, dtype=np.float64)
        num_accurate_pred_050_all = np.array(num_accurate_pred_050_all, dtype=np.float64)

        print("TRAJECTORIES SHAPES:")
        print(np.shape(truths_all))
        print(np.shape(predictions_all))
        rmnse_avg = np.mean(rmnse_all)
        print("AVERAGE RMNSE ERROR: {:}".format(rmnse_avg))
        num_accurate_pred_005_avg = np.mean(num_accurate_pred_005_all)
        print("AVG NUMBER OF ACCURATE 0.05 PREDICTIONS: {:}".format(num_accurate_pred_005_avg))
        num_accurate_pred_050_avg = np.mean(num_accurate_pred_050_all)
        print("AVG NUMBER OF ACCURATE 0.5 PREDICTIONS: {:}".format(num_accurate_pred_050_avg))
        freq_pred, freq_true, sp_true, sp_pred, error_freq = computeFrequencyError(predictions_all, truths_all, dt)
        print("FREQUENCY ERROR: {:}".format(error_freq))

        plotSpectrum(self, sp_true, sp_pred, freq_true, freq_pred, set_name)
        return rmnse_avg, num_accurate_pred_005_avg, num_accurate_pred_050_avg, error_freq, predictions_all, truths_all, freq_pred, freq_true, sp_true, sp_pred

    def saveResults(self):

        if self.write_to_log == 1:
            logfile_test = self.saving_path + self.logfile_dir + self.model_name  + "/test.txt"
            writeToTestLogFile(logfile_test, self)
            
        data = {}
        for var_name in getNamesInterestingVars():
            exec("data['{:s}_TEST'] = self.{:s}_TEST".format(var_name, var_name))
            exec("data['{:s}_TRAIN'] = self.{:s}_TRAIN".format(var_name, var_name))
        data["model_name"] = self.model_name
        data["n_tests"] = self.n_tests
        data_path = self.saving_path + self.results_dir + self.model_name + "/results.pickle"
        with open(data_path, "wb") as file:
            # Pickle the "data" dictionary using the highest protocol available.
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0

    def loadModel(self):
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        try:
            with open(data_path, "rb") as file:
                data = pickle.load(file)
                self.W_out = data["W_out"]
                self.Zop = data["Zop"]
                self.Xop = data["Xop"]
                self.P0op = data["P0op"]
                self.P1op = data["P1op"]
                self.W_feed = data["W_feed"]
                self.Uops = data["Uops"]
                self.scaler = data["scaler"]
                self.init_rhos = data["init_rhos"]
                self.Pauli_op = data["Pauli_op"]
                self.Pauli_op_corr = data["Pauli_op_corr"]
                del data
            return 0
        except:
            print("MODEL {:s} NOT FOUND.".format(data_path))
            return 1

    def saveModel(self):
        print("Recording time...")
        self.total_training_time = time.time() - self.start_time
        print("Total training time is {:}".format(self.total_training_time))

        print("MEMORY TRACKING IN MB...")
        process = psutil.Process(os.getpid())
        memory = process.memory_info().rss/1024/1024
        self.memory = memory
        print("Script used {:} MB".format(self.memory))
        print("SAVING MODEL...")

        if self.write_to_log == 1:
            logfile_train = self.saving_path + self.logfile_dir + self.model_name  + "/train.txt"
            writeToTrainLogFile(logfile_train, self)

        data = {
            "memory":self.memory,
            "n_trainable_parameters":self.n_trainable_parameters,
            "n_model_parameters":self.n_model_parameters,
            "total_training_time":self.total_training_time,
            "W_out":self.W_out,
            "Pauli_op":self.Pauli_op,
            "Pauli_op_corr":self.Pauli_op_corr,
            "P0op":self.P0op,
            "P1op":self.P1op,
            "W_feed":self.W_feed,
            "Uops":self.Uops,
            "init_rhos":self.init_rhos,
            "scaler":self.scaler,
        }
        data_path = self.saving_path + self.model_dir + self.model_name + "/data.pickle"
        with open(data_path, "wb") as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)
            del data
        return 0



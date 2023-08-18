import numpy as np
import itertools

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from labeler.lenet_weak_labeler import LeNetWeakLabeler
from labeler.cifar_weak_labeler import CifarWeakLabeler
from labeler.bootstrapping import bootstrap_xy_balanced_class
from random import random, randint
import os

import torch

if torch.cuda.is_available():
    print("Cuda available")
    DEVICE = "cuda:0"
else:
    print("Cuda not available")
    DEVICE = "cpu"


def adjust_minval_to_zero(scores):
    return scores + scores.min()


def adjust_range_zero_to_one(scores):
    tmp = adjust_minval_to_zero(scores)
    return tmp / (tmp.max() + 1e-15)


def score_to_prob(scores, threshold=0.0):
    # assume the given score is in range [0, 1]
    tmp = np.nan_to_num(scores)
    score_above_th_mask = tmp >= threshold
    score_above_th = tmp * score_above_th_mask
    return score_above_th / (sum(score_above_th) + 1e-15)


class Synthesizer(object):
    """
    A class to synthesize heuristics from primitives and validation labels
    """

    # cnn: if we are using LenetWeakLabler as weak labeler
    # num_cnn_labelers: number of guesses for one weak labeler
    # task: 'mnist', 'fashion', 'cifar10', 'omniglot' or 'all': generate labelers with different architectures
    def __init__(self, primitive_matrix, val_ground, b=0.5, cnn=False, num_cnn_labelers=10, task='mnist',
                 lr=1e-3, n_batches=5, n_epochs=80, n_classes=2, bootstrap_size_per_class=25):
        """
        Initialize Synthesizer object

        b: class prior of most likely class
        beta: threshold to decide whether to abstain or label for heuristics
        """
        self.val_primitive_matrix = primitive_matrix
        self.val_ground = val_ground
        self.p = np.shape(self.val_primitive_matrix)[1]
        self.b = b
        self.cnn = cnn
        if self.cnn:
            self.num_cnn_labelers = num_cnn_labelers
            self.task = task
            self.lr = lr
            self.n_batches = n_batches
            self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.bootstrap_size_per_class = bootstrap_size_per_class

    # 1. Not used in our CNN version
    def generate_feature_combinations(self, cardinality=1):
        """
        Create a list of primitive index combinations for given cardinality

        max_cardinality: max number of features each heuristic operates over
        """
        primitive_idx = range(self.p)
        feature_combinations = []

        for comb in itertools.combinations(primitive_idx, cardinality):
            feature_combinations.append(comb)

        return feature_combinations

    # 2. In our CNN version, use all features
    def fit_function(self, comb, model, model_file=None, model_task_index=0, finetune_transferred_wls=False):
        """
        Fits a single logistic regression or decision tree model

        comb: feature combination to fit model over
        model: fit logistic regression or a decision tree
        """
        if not self.cnn:
            X = self.val_primitive_matrix[:, comb]
            if np.shape(X)[0] == 1:
                X = X.reshape(-1, 1)
        else:
            X = self.val_primitive_matrix

        # fit decision tree or logistic regression or knn
        if model == 'dt':
            dt = DecisionTreeClassifier(max_depth=len(comb))
            dt.fit(X, self.val_ground)
            return dt

        elif model == 'lr':
            lr = LogisticRegression()
            lr.fit(X, self.val_ground)
            return lr

        elif model == 'nn':
            nn = KNeighborsClassifier(algorithm='kd_tree')
            nn.fit(X, self.val_ground)
            return nn

        # Additional option to Snuba: LeNetWeakLabeler as labeling function
        # Randomization on bootstrapping data
        elif model == 'cnn':
            dict_training_param = {'learning_rate': self.lr, 'num_batches': self.n_batches, 'num_epochs': self.n_epochs}
            X_boot, y_boot = bootstrap_xy_balanced_class(self.val_primitive_matrix, self.val_ground,
                                                         size_per_class=self.bootstrap_size_per_class)
            if self.task == 'mnist' or self.task == 'fashion':
                cnn = LeNetWeakLabeler(in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=2,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar10':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=2,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'mnist_5_way':
                cnn = LeNetWeakLabeler(in_dim_h=28, in_dim_w=28, in_dim_c=1, out_dim=5,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar10_10_way':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=10,
                                       dict_training_param=dict_training_param).to(DEVICE)
            elif self.task == 'cifar100_5_way':
                cnn = CifarWeakLabeler(in_dim_h=32, in_dim_w=32, in_dim_c=3, out_dim=10,
                                       dict_training_param=dict_training_param).to(DEVICE)
            else:
                raise NotImplementedError
            if model_file is not None:
                # print("\tUse pre-trained weak labelers")
                cnn.load_state_dict(torch.load(model_file))
                cnn.update_info_transferred_task(model_task_index)
            cnn.train()
            if (model_file is None) or ((model_file is not None) and finetune_transferred_wls):
                cnn.train_cnn(X_boot, y_boot)
            cnn.eval()
            return cnn

    def generate_heuristics(self, model, max_cardinality=1, sample_earlier_lfs=False, sample_earlier_lfs_prob=0.5,
                            task_similarity_score=None, task_similarity_score_threshold=0.0, path_earlier_lfs=None,
                            unified_score=False, finetune_transferred_wls=False):
        """
        Generates heuristics over given feature cardinality

        model: fit logistic regression or a decision tree
        max_cardinality: max number of features each heuristic operates over
        """
        # have to make a dictionary?? or feature combinations here? or list of arrays?
        feature_combinations_final = []
        heuristics_final = []

        if not self.cnn:
            for cardinality in range(1, max_cardinality + 1):
                feature_combinations = self.generate_feature_combinations(cardinality)
                heuristics = []
                for i, comb in enumerate(feature_combinations):
                    heuristics.append(self.fit_function(comb, model, model_file=None, model_task_index=-1))
                feature_combinations_final.append(feature_combinations)
                heuristics_final.append(heuristics)
        else:  # 3. Use all pixels in image for cnn; no feature selection needed
            comb = range(self.val_primitive_matrix.shape[1])
            feature_combinations_final.append([comb])
            heuristics = []
            if unified_score and sample_earlier_lfs:
                # collects weak labelers of earlier tasks
                earlier_wls, earlier_wls_transferscore = [], []
                earlier_wls_taskids, earlier_wls_filename = [], []
                for task_id, (earlier_lfs_dir) in enumerate(path_earlier_lfs):
                    for file_name in os.listdir(earlier_lfs_dir):
                        if '.pt' in file_name:
                            earlier_wls.append(
                                self.fit_function(comb, 'cnn', model_file=os.path.join(earlier_lfs_dir, file_name),
                                                  model_task_index=task_id, finetune_transferred_wls=False))
                            earlier_wls_transferscore.append(task_similarity_score[task_id])
                            earlier_wls_taskids.append(task_id)
                            earlier_wls_filename.append(os.path.join(earlier_lfs_dir, file_name))
                earlier_wls_transferscore = np.array(earlier_wls_transferscore)

                # compute unified score
                L_val = np.array([])
                beta_opt = np.array([])
                max_cardinality = len(earlier_wls)
                for i in range(max_cardinality):
                    # Note that the LFs are being applied to the entire val set though they were developed on a subset...
                    beta_opt_temp = self.find_optimal_beta([earlier_wls[i]], self.val_primitive_matrix,
                                                           feature_combinations_final[0], self.val_ground)
                    L_temp_val = self._apply_heuristics([earlier_wls[i]], self.val_primitive_matrix,
                                                        feature_combinations_final[0], beta_opt_temp)
                    beta_opt = np.append(beta_opt, beta_opt_temp)
                    if i == 0:
                        L_val = np.append(L_val, L_temp_val)  # converts to 1D array automatically
                        L_val = np.reshape(L_val, np.shape(L_temp_val))
                    else:
                        L_val = np.concatenate((L_val, L_temp_val), axis=1)

                # Use F1 trade-off for reliability
                acc_cov_scores = [f1_score(self.val_ground, L_val[:, i], average='micro') for i in
                                  range(np.shape(L_val)[1])]
                acc_cov_scores = np.nan_to_num(acc_cov_scores)

                combined_scores = 0.5 * adjust_range_zero_to_one(earlier_wls_transferscore) + 0.5 * acc_cov_scores
                wl_sample_prob = score_to_prob(combined_scores, threshold=task_similarity_score_threshold)
                for i in range(self.num_cnn_labelers):
                    if random() <= sample_earlier_lfs_prob and np.max(wl_sample_prob) > 0.0:
                        wl_index = np.random.choice(len(earlier_wls), p=wl_sample_prob)
                        if finetune_transferred_wls:
                            # Need to regenerate weak labelers with fine-tuning
                            heuristics.append(self.fit_function(comb, 'cnn', model_file=earlier_wls_filename[wl_index],
                                                                model_task_index=earlier_wls_taskids[wl_index],
                                                                finetune_transferred_wls=finetune_transferred_wls))
                        else:
                            # Sample weak labelers from a collection of earlier weak labelers
                            heuristics.append(earlier_wls[wl_index])
                    else:
                        # train new weak labeler
                        heuristics.append(self.fit_function(comb, 'cnn', model_file=None, model_task_index=-1,
                                                            finetune_transferred_wls=finetune_transferred_wls))
            else:
                for i in range(self.num_cnn_labelers):
                    model_file, model_task_index = self._initialize_heuristics_function(
                        sample_earlier_lfs=sample_earlier_lfs, sample_earlier_lfs_prob=sample_earlier_lfs_prob,
                        task_similarity_score=task_similarity_score,
                        task_similarity_score_threshold=task_similarity_score_threshold,
                        path_earlier_lfs=path_earlier_lfs)
                    heuristics.append(
                        self.fit_function(comb, 'cnn', model_file=model_file, model_task_index=model_task_index,
                                          finetune_transferred_wls=finetune_transferred_wls))
            heuristics_final.append(heuristics)

        return heuristics_final, feature_combinations_final

    def _initialize_heuristics_function(self, sample_earlier_lfs=False, sample_earlier_lfs_prob=0.5,
                                        task_similarity_score=None, task_similarity_score_threshold=0.0,
                                        path_earlier_lfs=None):
        def normalize_prob_dist(unnormalized_prob):
            tmp = np.nan_to_num(unnormalized_prob)
            return tmp / (sum(tmp) + 1e-15)

        def convert_leep_to_prob(leep_score):
            return np.exp(leep_score)

        sampled_lf_file_name, task_to_sample = None, -1
        if sample_earlier_lfs and random() <= sample_earlier_lfs_prob:
            # sample lf from earlier tasks
            num_earlier_tasks = len(task_similarity_score)
            transfer_score_above_th_mask = task_similarity_score >= task_similarity_score_threshold
            transfer_score_above_th_tmp = convert_leep_to_prob(task_similarity_score) * transfer_score_above_th_mask

            if sum(transfer_score_above_th_tmp) > 1e-10:
                transfer_score_above_th = normalize_prob_dist(transfer_score_above_th_tmp)
                task_to_sample = np.random.choice(num_earlier_tasks,
                                                  p=transfer_score_above_th)  # index in task list to sample

                # sample lf file in the directory of the selected earlier task
                lf_file_name_list, sampled_earlier_task = [], path_earlier_lfs[task_to_sample]
                for file_name in os.listdir(sampled_earlier_task):
                    if '.pt' in file_name:
                        lf_file_name_list.append(file_name)

                if len(lf_file_name_list) > 0:
                    # print("\tUse pre-trained weak labelers")
                    sampled_lf_file_name = os.path.join(sampled_earlier_task,
                                                        lf_file_name_list[randint(0, len(lf_file_name_list) - 1)])
        return sampled_lf_file_name, task_to_sample

    def _apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
        """
        Apply given heuristics to given feature matrix X and abstain by beta

        heuristics: list of pre-trained logistic regression models
        feat_combos: primitive indices to apply heuristics to
        beta: best beta value for associated heuristics
        """

        def marginals_to_labels(hf, X, beta):
            if self.n_classes == 2:
                if not self.cnn:
                    marginals = hf.predict_proba(X)[:, 1]  # binary version: marginals is prob of class 1
                else:
                    marginals = hf.marginal(X)
                labels_cutoff = np.zeros(np.shape(marginals))  # class 0, not abstain
                labels_cutoff[marginals <= (self.b - beta)] = -1.  # both classes, abstain
                labels_cutoff[marginals >= (self.b + beta)] = 1.  # class 1, not abstain
            else:
                # 2. Multiclass version of assigning labels: if abstain, assign -1; otherwise, assign 0 .. n_classes - 1
                if not self.cnn:
                    marginals = hf.predict_proba(X)
                else:
                    marginals = hf.prob_matrix(X.astype('float32'))
                labels_cutoff = np.zeros(np.shape(marginals)[0])
                marginals_best = np.amax(marginals, axis=1)  # compute top prob
                labels_cutoff[marginals_best >= (self.b + beta)] = np.argmax(marginals, axis=1)[
                    marginals_best >= (self.b + beta)]  # top prob >= b+beta, not abstain
                labels_cutoff[marginals_best < (self.b + beta)] = -1  # otherwise, abstain
            return labels_cutoff

        L = np.zeros((np.shape(primitive_matrix)[0], len(heuristics)))
        if not self.cnn:
            for i, hf in enumerate(heuristics):
                L[:, i] = marginals_to_labels(hf, primitive_matrix[:, feat_combos[i]], beta_opt[i])
        else:  # For CNN, all heuristics are generated from all pixels (the only feature combo)
            for i, hf in enumerate(heuristics):
                L[:, i] = marginals_to_labels(hf, primitive_matrix, beta_opt[i])
        return L

    def beta_optimizer(self, marginals, ground):
        """
        Returns the best beta parameter for abstain threshold given marginals
        Uses F1 score that maximizes the F1 score

        marginals: confidences for data from a single heuristic
        """

        # Set the range of beta params
        # 0.25 instead of 0.0 as a min makes controls coverage better
        beta_params = np.linspace(0.25, 0.45, 10)
        f1 = []
        for beta in beta_params:
            # 4. Binary version same as before
            # Multiclass version: as in binary version, assign -1 to abstain, 0 ... n_classes-1 to others
            # marginals in multiclass version is matrix of all prob
            if self.n_classes == 2:
                labels_cutoff = np.zeros(np.shape(marginals))
                labels_cutoff[marginals <= (self.b - beta)] = -1.
                labels_cutoff[marginals >= (self.b + beta)] = 1.
                f1.append(f1_score(ground, labels_cutoff, average='micro'))
            else:
                labels_cutoff = np.zeros(np.shape(marginals)[0])
                marginals_best = np.max(marginals, axis=1)
                # only higher than chance makes sense because of 3-way
                labels_cutoff[marginals_best >= (self.b + beta)] = np.argmax(marginals, axis=1)[marginals_best >= (
                        self.b + beta)]
                labels_cutoff[marginals_best < (self.b + beta)] = -1.
                f1.append(f1_score(ground, labels_cutoff, average='micro'))

        f1 = np.nan_to_num(f1)
        return beta_params[np.argsort(np.array(f1))[-1]]

    def find_optimal_beta(self, heuristics, X, feat_combos, ground):
        """
        Returns optimal beta for given heuristics

        heuristics: list of pre-trained logistic regression models
        X: primitive matrix
        feat_combos: feature indices to apply heuristics to
        ground: ground truth associated with X data
        """

        beta_opt = []
        for i, hf in enumerate(heuristics):
            if not self.cnn:
                # marginals = hf.predict_proba(X[:,feat_combos[i]])[:,1]
                marginals = hf.predict_proba(X[:, feat_combos[i]])
            else:  # X is a float tensor for cnn
                if self.n_classes == 2:
                    marginals = hf.marginal(X.astype('float32'))
                else:
                    marginals = hf.prob_matrix(X.astype('float32'))
            # labels_cutoff = np.zeros(np.shape(marginals))
            beta_opt.append((self.beta_optimizer(marginals, ground)))
        return beta_opt

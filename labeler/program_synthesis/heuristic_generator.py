import numpy as np
from sklearn.metrics import f1_score
from labeler.program_synthesis.synthesizer import Synthesizer
from labeler.program_synthesis.verifier import Verifier

import torch
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"


# 1. Calculate accuracy and coverage: abstain should be defined as top probability < threshold
# threshold = b + 0.1 = 1/n_classes + 0.1
def calculate_accuracy(marginals, b, ground):
    marginals_best = np.amax(marginals, axis=1)
    num_total = np.shape(np.where(marginals_best >= b + 0.1))[1]
    if num_total == 0:
        return 0
    labels = np.argmax(marginals, axis=1)[marginals_best >= b + 0.1]
    num_correct = np.sum(ground[marginals_best >= b + 0.1] == labels)
    return num_correct / float(num_total)


def calculate_coverage(marginals, b, ground):
    marginals_best = np.amax(marginals, axis=1)
    num_total = np.shape(np.where(marginals_best >= b))[1]
    if num_total == 0:
        return 0
    return num_total / float(np.shape(marginals)[0])


class HeuristicGenerator(object):
    """
    A class to go through the synthesizer-verifier loop
    """

    def __init__(self, train_primitive_matrix, val_primitive_matrix,
                 val_ground, train_ground=None, b=0.5, cnn=False, num_cnn_labelers=10, task='mnist',
                 lr=1e-3, n_batches=5, n_epochs=80, n_classes=2, bootstrap_size_per_class=25):
        """ 
        Initialize HeuristicGenerator object

        b: class prior of most likely class (TODO: use somewhere)
        beta: threshold to decide whether to abstain or label for heuristics
        gamma: threshold to decide whether to call a point vague or not
        """

        self.train_primitive_matrix = train_primitive_matrix
        self.val_primitive_matrix = val_primitive_matrix
        self.val_ground = val_ground
        self.train_ground = train_ground
        self.b = b

        self.vf = None
        self.syn = None
        self.hf = []
        self.feat_combos = []

        self.cnn = cnn
        if self.cnn:
            self.num_cnn_labelers = num_cnn_labelers
            self.task = task
            self.lr = lr
            self.n_batches = n_batches
            self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.bootstrap_size_per_class = bootstrap_size_per_class

    # Update lr, n_batches and n_epochs
    def update_training_configs(self, lr, n_batches, n_epochs):
        self.lr = lr
        self.n_batches = n_batches
        self.n_epochs = n_epochs

    # Add CNN option
    def apply_heuristics(self, heuristics, primitive_matrix, feat_combos, beta_opt):
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
                labels_cutoff[marginals <= (self.b-beta)] = -1.  # both classes, abstain
                labels_cutoff[marginals >= (self.b+beta)] = 1.  # class 1, not abstain
            else:
                # 2. Multiclass version of assigning labels: if abstain, assign -1; otherwise, assign 0 .. n_classes - 1
                if not self.cnn:
                    marginals = hf.predict_proba(X)
                else:
                    marginals = hf.prob_matrix(X.astype('float32'))
                labels_cutoff = np.zeros(np.shape(marginals)[0])
                marginals_best = np.amax(marginals, axis=1)  # compute top prob
                labels_cutoff[marginals_best >= (self.b+beta)] = np.argmax(marginals, axis=1)[marginals_best >= (self.b+beta)]  # top prob >= b+beta, not abstain
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

    # 3. No difference for pruner: L matrices are the same after apply_heuristics
    def prune_heuristics(self, heuristics, feat_combos, keep=1):
        """ 
        Selects the best heuristic based on Jaccard Distance and Reliability Metric

        keep: number of heuristics to keep from all generated heuristics
        """

        def calculate_jaccard_distance(num_labeled_total, num_labeled_L):
            scores = np.zeros(np.shape(num_labeled_L)[1])
            for i in range(np.shape(num_labeled_L)[1]):
                scores[i] = np.sum(np.minimum(num_labeled_L[:,i],num_labeled_total))/np.sum(np.maximum(num_labeled_L[:,i],num_labeled_total))
            return 1-scores
        
        L_val = np.array([])
        L_train = np.array([])
        beta_opt = np.array([])
        max_cardinality = len(heuristics)
        for i in range(max_cardinality):
            #Note that the LFs are being applied to the entire val set though they were developed on a subset...
            beta_opt_temp = self.syn.find_optimal_beta(heuristics[i], self.val_primitive_matrix, feat_combos[i], self.val_ground)
            L_temp_val = self.apply_heuristics(heuristics[i], self.val_primitive_matrix, feat_combos[i], beta_opt_temp)
            L_temp_train = self.apply_heuristics(heuristics[i], self.train_primitive_matrix, feat_combos[i], beta_opt_temp)
            
            beta_opt = np.append(beta_opt, beta_opt_temp)
            if i == 0:
                L_val = np.append(L_val, L_temp_val) #converts to 1D array automatically
                L_val = np.reshape(L_val,np.shape(L_temp_val))
                L_train = np.append(L_train, L_temp_train) #converts to 1D array automatically
                L_train = np.reshape(L_train,np.shape(L_temp_train))
            else:
                L_val = np.concatenate((L_val, L_temp_val), axis=1)
                L_train = np.concatenate((L_train, L_temp_train), axis=1)
        
        #Use F1 trade-off for reliability
        acc_cov_scores = [f1_score(self.val_ground, L_val[:,i], average='micro') for i in range(np.shape(L_val)[1])] 
        acc_cov_scores = np.nan_to_num(acc_cov_scores)
        
        if self.vf != None:
            #Calculate Jaccard score for diversity
            train_num_labeled = np.sum(np.abs(self.vf.L_train.T), axis=0) 
            jaccard_scores = calculate_jaccard_distance(train_num_labeled,np.abs(L_train))
        else:
            jaccard_scores = np.ones(np.shape(acc_cov_scores))

        #Weighting the two scores to find best heuristic
        # combined_scores = 0.5*acc_cov_scores + 0.5*jaccard_scores

        # Data-based weighting for Jaccard and F1 (review)
        if self.vf != None:
            train_coverage = calculate_coverage(self.vf.train_marginals, self.b, self.train_ground)
        else:
            train_coverage = 0.5
        combined_scores = train_coverage * acc_cov_scores + (1.0 - train_coverage) * jaccard_scores

        sort_idx = np.argsort(combined_scores)[::-1][0:keep]
        return sort_idx

    # 4. No difference except for inside Synthesizer
    def run_synthesizer(self, max_cardinality=1, idx=None, keep=1, model='cnn', sample_earlier_lfs=False,
                        sample_earlier_lfs_prob=0.5, task_similarity_score=None, task_similarity_score_threshold=0.0,
                        path_earlier_lfs=None, unified_score=False, finetune_transferred_wls=False):
        """
        Generates Synthesizer object and saves all generated heuristics

        max_cardinality: max number of features candidate programs take as input
        idx: indices of validation set to fit programs over
        keep: number of heuristics to pass to verifier
        model: train logistic regression ('lr') or decision tree ('dt')
        """
        
        if idx == None:
            primitive_matrix = self.val_primitive_matrix
            ground = self.val_ground
        else:
            primitive_matrix = self.val_primitive_matrix[idx, :]
            ground = self.val_ground[idx]

        #Generate all possible heuristics
        self.syn = Synthesizer(primitive_matrix, ground, b=self.b, cnn=self.cnn, num_cnn_labelers=self.num_cnn_labelers,
                               task=self.task, lr=self.lr, n_batches=self.n_batches, n_epochs=self.n_epochs,
                               n_classes=self.n_classes, bootstrap_size_per_class=self.bootstrap_size_per_class)

        #Un-flatten indices, not used for our CNN version
        def index(a, inp):
            i = 0
            remainder = 0
            while inp >= 0:
                remainder = inp
                inp -= len(a[i])
                i+=1
            try:
                return a[i-1][remainder] #TODO: CHECK THIS REMAINDER THING WTF IS HAPPENING
            except:
                import pdb; pdb.set_trace()

        #Select keep best heuristics from generated heuristics
        # syn_start = time.time()
        hf, feat_combos = self.syn.generate_heuristics(model, max_cardinality, sample_earlier_lfs=sample_earlier_lfs,
                                                       sample_earlier_lfs_prob=sample_earlier_lfs_prob,
                                                       task_similarity_score=task_similarity_score,
                                                       task_similarity_score_threshold=task_similarity_score_threshold,
                                                       path_earlier_lfs=path_earlier_lfs, unified_score=unified_score,
                                                       finetune_transferred_wls=finetune_transferred_wls)
        sort_idx = self.prune_heuristics(hf, feat_combos, keep)
        
        if not self.cnn:
            for i in sort_idx:
                self.hf.append(index(hf,i)) 
                self.feat_combos.append(index(feat_combos,i))
        else:
            for i in sort_idx:
                self.hf.append(hf[0][i])
                self.feat_combos.append(feat_combos)

        #create appended L matrices for validation and train set
        beta_opt = self.syn.find_optimal_beta(self.hf, self.val_primitive_matrix, self.feat_combos, self.val_ground)
        self.L_val = self.apply_heuristics(self.hf, self.val_primitive_matrix, self.feat_combos, beta_opt)
        self.L_train = self.apply_heuristics(self.hf, self.train_primitive_matrix, self.feat_combos, beta_opt)

        # return optimal betas
        return beta_opt

    # 5. Use Snorkel instead of LabelAggregator for generative model; LabelAggregator might not support multiclass
    def run_verifier(self):
        """ 
        Generates Verifier object and saves marginals
        """
        ###THIS IS WHERE THE SNORKEL FLAG IS SET!!!!
        self.vf = Verifier(self.L_train, self.L_val, self.val_ground, has_snorkel=True, n_classes=self.n_classes)
        self.vf.train_gen_model()
        self.vf.assign_marginals()

    # 6. Same gamma optimizer
    def gamma_optimizer(self,marginals):
        """ 
        Returns the best gamma parameter for abstain threshold given marginals

        marginals: confidences for data from a single heuristic
        """
        m = len(self.hf)
        gamma = 0.5-(1/(m**(3/2.))) 
        return gamma

    # 7. Same feedback
    def find_feedback(self):
        """ 
        Finds vague points according to gamma parameter

        self.gamma: confidence past 0.5 that relates to a vague or incorrect point
        """
        #TODO: flag for re-classifying incorrect points
        #incorrect_idx = self.vf.find_incorrect_points(b=self.b)

        gamma_opt = self.gamma_optimizer(self.vf.val_marginals)
        gamma_opt = np.maximum(gamma_opt, 0)  # if gamma < 0, gamma = 0
        #gamma_opt = self.gamma
        vague_idx = self.vf.find_vague_points(b=self.b, gamma=gamma_opt)
        incorrect_idx = vague_idx
        self.feedback_idx = list(set(list(np.concatenate((vague_idx, incorrect_idx)))))

    # The following functions are not used in our workflow, so not updated to multiclass

    # def evaluate(self):
    #     """
    #     Calculate the accuracy and coverage for train and validation sets
    #     """
    #     # directly output from Snorkel
    #     self.val_marginals = self.vf.val_marginals
    #     self.train_marginals = self.vf.train_marginals
    #
    #     # Replace 0 labels to -1
    #     val_ground = np.copy(self.val_ground)
    #     val_ground[val_ground == 0] = -1
    #     train_ground = np.copy(self.train_ground)
    #     train_ground[train_ground == 0] = -1
    #
    #     self.val_accuracy = calculate_accuracy(self.val_marginals, self.b, val_ground)
    #     self.train_accuracy = calculate_accuracy(self.train_marginals, self.b, train_ground)
    #     self.val_coverage = calculate_coverage(self.val_marginals, self.b, val_ground)
    #     self.train_coverage = calculate_coverage(self.train_marginals, self.b, train_ground)
    #     return self.val_accuracy, self.train_accuracy, self.val_coverage, self.train_coverage


    # def heuristic_stats(self):
    #     '''For each heuristic, we want the following:
    #     - idx of the features it relies on
    #     - if dt, then the thresholds?
    #     '''
    #     def calculate_accuracy(marginals, b, ground):
    #         total = np.shape(np.where(marginals != 0.5))[1]
    #         if total == 0:
    #             return 0
    #         labels = np.sign(2*(marginals - 0.5))
    #         return np.sum(labels == ground)/float(total)
    #
    #     def calculate_coverage(marginals, b, ground):
    #         total = np.shape(np.where(marginals != 0))[1]
    #         if total == 0:
    #             return 0
    #         labels = marginals
    #         return total/float(len(labels))
    #
    #     # Replace 0 labels to -1
    #     val_ground = np.copy(self.val_ground)
    #     val_ground[val_ground == 0] = -1
    #     train_ground = np.copy(self.train_ground)
    #     train_ground[train_ground == 0] = -1
    #
    #     stats_table = np.zeros((len(self.hf),6))
    #     for i in range(len(self.hf)):
    #         stats_table[i,0] = int(self.feat_combos[i][0])
    #         try:
    #             stats_table[i,1] = int(self.feat_combos[i][1])
    #         except:
    #             stats_table[i,1] = -1.
    #         stats_table[i,2] = calculate_accuracy(self.L_val[:,i], self.b, val_ground)
    #         stats_table[i,3] = calculate_accuracy(self.L_train[:,i], self.b, train_ground)
    #         stats_table[i,4] = calculate_coverage(self.L_val[:,i], self.b, val_ground)
    #         stats_table[i,5] = calculate_coverage(self.L_train[:,i], self.b, train_ground)
    #
    #     #Make table
    #     column_headers = ['Feat 1', 'Feat 2', 'Val Acc', 'Train Acc', 'Val Cov', 'Train Cov']
    #     pandas_stats_table = pd.DataFrame(stats_table, columns=column_headers)
    #     return pandas_stats_table


            



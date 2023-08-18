import os
import timeit

import numpy as np
import tensorflow as tf
import torch
from scipy.io import savemat

from lml.utils.utils import compute_coupling, compute_CE
from lml.utils.utils_data import mnist_data_print_info, cifar_data_print_info
from lml.classification.train import model_generation

from labeler.weak_labeler_generator import generate_labeler_for_one_task
from labeler.strong_labels_generator import generate_strong_label

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
DPSSCL_DATA_ROOT = os.path.join(PROJECT_ROOT, 'Data', 'task_data')

_debug_mode = False

def get_sorted_directories_under_path(path_to_check, return_abs_path=False):
    tmp_dir_list, return_list = os.listdir(path_to_check), []
    tmp_dir_list.sort()
    for elem in tmp_dir_list:
        if os.path.isdir(os.path.join(path_to_check, elem)):
            if return_abs_path:
                return_list.append(os.path.join(path_to_check, elem))
            else:
                return_list.append(elem)
    return return_list

def remove_weak_labeler_modelfiles(task_root_path, model_file_ext='.pt'):
    # remove model file (i.e. lf0.pt) under the given task root (i.e. task_data/mnist/)
    for task_dir in os.listdir(task_root_path):
        task_path = os.path.join(task_root_path, task_dir)
        if os.path.isdir(task_path):
            for filename in os.listdir(task_path):
                if model_file_ext in filename:
                    os.remove(os.path.join(task_path, filename))
    print("Remove all saved weak labeling functions!")

def train_with_transferred_wlf(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, run_cnt=0, task_order=None, weak_labeler_root=None):
    print("Training function for semi-supervised lifelong learning (DPSSCL)!")
    assert ('den' not in model_architecture and 'dynamically' not in model_architecture), "Use train function appropriate to the architecture"

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_device)
        if _up_to_date_tf:
            ## TF version >= 1.14
            gpu = tf.config.experimental.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            ## TF version < 1.14
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        print("GPU %d is used" % (GPU_device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("CPU is used")

    if task_order is None:
        task_training_order = list(range(train_hyperpara['num_tasks']))
    else:
        task_training_order = list(task_order)
    print("\tTask order : ", task_training_order)
    task_change_epoch = [1]

    ### set-up data
    labeled_data, unlabeled_data, test_data_torch = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(labeled_data, unlabeled_data, test_data_torch, True, print_info=False)
        if weak_labeler_root is None:
            weak_labeler_root = os.path.join(DPSSCL_DATA_ROOT, 'mnist_bin')
        remove_weak_labeler_modelfiles(weak_labeler_root)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(labeled_data, unlabeled_data, test_data_torch, True, print_info=False)
        if weak_labeler_root is None:
            weak_labeler_root = os.path.join(DPSSCL_DATA_ROOT, 'cifar10_bin')
        remove_weak_labeler_modelfiles(weak_labeler_root)
    elif 'cifar100' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(labeled_data, unlabeled_data, test_data_torch, True, print_info=False)
        if weak_labeler_root is None:
            weak_labeler_root = os.path.join(DPSSCL_DATA_ROOT, 'cifar100_bin')
        remove_weak_labeler_modelfiles(weak_labeler_root)
    else:
        raise ValueError("No specified dataset!")
    task_dir_list = get_sorted_directories_under_path(weak_labeler_root, return_abs_path=True)
    x_shape = labeled_data[0][0].shape
    if len(x_shape) > 2:
        reshape_x = True
        x_dim = x_shape[1]*x_shape[2]*x_shape[3]
    else:
        reshape_x = False
    train_data, validation_data, test_data = [0 for _ in range(num_task)], [0 for _ in range(num_task)], [0 for _ in range(num_task)]

    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task])
    if not generation_success:
        return (None, None)

    ### Training Procedure
    learning_step = -1
    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

    if train_hyperpara['confusion_matrix']:
        confusion_matrices = np.zeros((num_task,), dtype=np.object)
    task_similarity_score_threshold = train_hyperpara['task_similarity_score_threshold']
    sample_lf_from_earlier_task_prob = train_hyperpara['sample_lf_from_earlier_task_prob']
    strong_label_method = train_hyperpara['strong_labeler']
    num_unlabeled_data = train_hyperpara['num_unlabeled_data']
    lg_min, lg_max, lg_keep = train_hyperpara['num_weak_labelers_min'], train_hyperpara['num_weak_labelers_max'], train_hyperpara['num_weak_labelers_keep']
    unified_score, finetune_transferred_wls = train_hyperpara['unified_score'], train_hyperpara['finetune_wls']
    transfer_score, transfer_score_hist = None, np.zeros((len(task_training_order),), dtype=np.object)
    transferred_labelers_info, labelers_converged_epochs, snuba_iterations = [], [], []

    if train_hyperpara['debug_mode']:
        hfs_quality_transfer_f1score = np.zeros((len(task_training_order),), dtype=np.object)
        hfs_quality_transfer_jaccard = np.zeros((len(task_training_order),), dtype=np.object)
        hfs_quality_transfer_coverage = np.zeros((len(task_training_order)))
        hfs_quality_transfer_trainacc_hist = np.zeros((len(task_training_order),), dtype=np.object)
        hfs_quality_nontransfer_f1score = np.zeros((len(task_training_order),), dtype=np.object)
        hfs_quality_nontransfer_jaccard = np.zeros((len(task_training_order),), dtype=np.object)
        hfs_quality_nontransfer_coverage = np.zeros((len(task_training_order)))
        hfs_quality_nontransfer_trainacc_hist = np.zeros((len(task_training_order),), dtype=np.object)


    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\n%d-th Task - %d" % (train_task_cnt, task_for_train))
        tf.reset_default_graph()

        # generate weak labelers, and use ensemble of these weak labelers for label generation
        data_for_labeler_gen = [labeled_data[task_for_train][0], labeled_data[task_for_train][1], unlabeled_data[task_for_train][0], unlabeled_data[task_for_train][1]]
        if 'mnist' in data_type:
            labelers_transferred_task_ids, labelers_converged_epoch_ratio, snuba_iter_cnt, hfs_quality_score = generate_labeler_for_one_task(task_dir=os.path.join(weak_labeler_root, task_dir_list[task_for_train]), task='mnist', data=data_for_labeler_gen, task_similarity_score=transfer_score, task_similarity_score_threshold=task_similarity_score_threshold, sample_lf_from_earlier_task_prob=sample_lf_from_earlier_task_prob, path_to_earlier_task_lf=[task_dir_list[task_training_order[a]] for a in range(train_task_cnt)], lg_min=lg_min, lg_max=lg_max, lg_keep=lg_keep, return_snuba_iter_cnt=True, unified_score=unified_score, finetune_transferred_wls=finetune_transferred_wls, debug_mode=train_hyperpara['debug_mode'])
            # ensemble weak labelers
            y_u_prime = generate_strong_label(task='mnist', task_dir=os.path.join(weak_labeler_root, task_dir_list[task_for_train]), cardinality=2, n_u=num_unlabeled_data, method=strong_label_method, data=data_for_labeler_gen)
        elif ('cifar10' in data_type) and not ('cifar100' in data_type):
            labelers_transferred_task_ids, labelers_converged_epoch_ratio, snuba_iter_cnt, hfs_quality_score = generate_labeler_for_one_task(task_dir=os.path.join(weak_labeler_root, task_dir_list[task_for_train]), task='cifar10', data=data_for_labeler_gen, task_similarity_score=transfer_score, task_similarity_score_threshold=task_similarity_score_threshold, sample_lf_from_earlier_task_prob=sample_lf_from_earlier_task_prob, path_to_earlier_task_lf=[task_dir_list[task_training_order[a]] for a in range(train_task_cnt)], lg_min=lg_min, lg_max=lg_max, lg_keep=lg_keep, return_snuba_iter_cnt=True, unified_score=unified_score, finetune_transferred_wls=finetune_transferred_wls, debug_mode=train_hyperpara['debug_mode'])
            # ensemble weak labelers
            y_u_prime = generate_strong_label(task='cifar10', task_dir=os.path.join(weak_labeler_root, task_dir_list[task_for_train]), cardinality=2, n_u=num_unlabeled_data, method=strong_label_method, data=data_for_labeler_gen)
        transferred_labelers_info.append(labelers_transferred_task_ids)
        labelers_converged_epochs.append(labelers_converged_epoch_ratio)
        snuba_iterations.append(snuba_iter_cnt)
        if (hfs_quality_score is not None) and (train_hyperpara['debug_mode']):
            hfs_quality_transfer_f1score[train_task_cnt] = hfs_quality_score[0][0]
            hfs_quality_transfer_jaccard[train_task_cnt] = hfs_quality_score[0][1]
            hfs_quality_transfer_coverage[train_task_cnt] = hfs_quality_score[0][2]
            hfs_quality_transfer_trainacc_hist[train_task_cnt] = hfs_quality_score[0][3]
            hfs_quality_nontransfer_f1score[train_task_cnt] = hfs_quality_score[1][0]
            hfs_quality_nontransfer_jaccard[train_task_cnt] = hfs_quality_score[1][1]
            hfs_quality_nontransfer_coverage[train_task_cnt] = hfs_quality_score[1][2]
            hfs_quality_nontransfer_trainacc_hist[train_task_cnt] = hfs_quality_score[1][3]

            tmp = {}
            tmp['transfer_f1score'] = hfs_quality_score[0][0]
            tmp['transfer_jaccard'] = hfs_quality_score[0][1]
            tmp['transfer_coverage'] = hfs_quality_score[0][2]
            tmp['transfer_trainacc'] = hfs_quality_score[0][3]
            tmp['nontransfer_f1score'] = hfs_quality_score[1][0]
            tmp['nontransfer_jaccard'] = hfs_quality_score[1][1]
            tmp['nontransfer_coverage'] = hfs_quality_score[1][2]
            tmp['nontransfer_trainacc'] = hfs_quality_score[1][3]
            savemat('./Result/hfs_debug_taskorder%d.mat'%(train_task_cnt), {'hfs_quality_metrics': tmp})
        print("Generated pseudo-labels for unlabeled data!")

        # convert channels of image tensors & flatten imgs for lifelong learning code
        X_l, y_l = np.transpose(labeled_data[task_for_train][0], (0, 2, 3, 1)), labeled_data[task_for_train][1]
        X_u_train, y_u_train = np.transpose(unlabeled_data[task_for_train][0][:num_unlabeled_data], (0, 2, 3, 1)), y_u_prime[:num_unlabeled_data]
        X_u_valid, y_u_valid = np.transpose(unlabeled_data[task_for_train][0][num_unlabeled_data:], (0, 2, 3, 1)), unlabeled_data[task_for_train][1][num_unlabeled_data:]
        X_test, y_test = np.transpose(test_data_torch[task_for_train][0], (0, 2, 3, 1)), test_data_torch[task_for_train][1]

        if reshape_x:
            X_l = np.reshape(X_l, [-1, x_dim])
            X_u_train = np.reshape(X_u_train, [-1, x_dim])
            X_u_valid = np.reshape(X_u_valid, [-1, x_dim])
            X_test = np.reshape(X_test, [-1, x_dim])

        train_data[task_for_train] = (np.concatenate((X_l, X_u_train), axis=0), np.concatenate((y_l, y_u_train), axis=0))
        validation_data[task_for_train] = (X_u_valid, y_u_valid)
        test_data[task_for_train] = (X_test, y_test)

        with tf.Session(config=config) as sess:
            # run lifelong learning code below
            print("\nLifelong Learning %d-th Task - %d" % (train_task_cnt, task_for_train))
            learning_model.add_new_task(y_depth[task_for_train], task_for_train, single_input_placeholder=True)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d' % (model_architecture, run_cnt, train_task_cnt), sess.graph)

            if save_param and _debug_mode:
                para_file_name = param_folder_path + '/init_model_parameter_taskC%d.mat' % (train_task_cnt)
                curr_param = learning_model.get_params_val(sess)
                savemat(para_file_name, {'parameter': curr_param})

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step + 1

                #### training & performance measuring process
                if learning_step > 0:
                    learning_model.train_one_epoch(sess, train_data[task_for_train][0], train_data[task_for_train][1], learning_step-1, task_for_train, None, dropout_prob=0.5)

                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt + 1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                if classification_prob:
                    ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                    train_error, valid_error, test_error = -(sum(train_error_tmp) / (num_learned_tasks)), -(sum(validation_error_tmp) / (num_learned_tasks)), -(sum(test_error_tmp) / (num_learned_tasks))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = -train_error_tmp[task_for_train], -validation_error_tmp[task_for_train], -test_error_tmp[task_for_train]
                else:
                    train_error, valid_error, test_error = np.sqrt(np.array(train_error_tmp) / (num_learned_tasks)), np.sqrt(np.array(validation_error_tmp) / (num_learned_tasks)), np.sqrt(np.array(test_error_tmp) / (num_learned_tasks))
                    train_error_tmp, validation_error_tmp, test_error_tmp = list(np.sqrt(np.array(train_error_tmp))), list(np.sqrt(np.array(validation_error_tmp))), list(np.sqrt(np.array(test_error_tmp)))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error_tmp[task_for_train], validation_error_tmp[task_for_train], test_error_tmp[task_for_train]

                #### error related process
                print('epoch %d - Train : %f, Validation : %f' % (
                learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

                if valid_error_to_compare < best_valid_error:
                    str_temp = ''
                    if valid_error_to_compare < best_valid_error * improvement_threshold:
                        patience = max(patience, (learning_step - epoch_bias) * patience_multiplier)
                        str_temp = '\t<<'
                    best_valid_error, best_epoch = valid_error_to_compare, learning_step
                    test_error_at_best_epoch = test_error_to_compare
                    print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                # if learning_step >= epoch_bias+min(patience, learning_step_max//num_task):
                if learning_step >= epoch_bias + min(patience, learning_step_max // len(task_training_order)):
                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat' % (train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order) - 1:
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step + 1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        # compute LEEP score on next task w/ prev task models
                        transfer_score = np.zeros(train_task_cnt + 1)
                        next_task_to_eval = task_training_order[train_task_cnt + 1]
                        if train_hyperpara['dpsscl_transfer_score'].lower() == 'leep':
                            print("Compute LEEP Score")
                            for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                                if task_index_to_eval in task_training_order[:tmp_cnt]:
                                    # if it is a re-visited task
                                    continue
                                transfer_score[tmp_cnt] = learning_model.compute_transferability_score_one_task(sess, np.transpose(labeled_data[next_task_to_eval][0], (0, 2, 3, 1)).reshape([-1, x_dim]), labeled_data[next_task_to_eval][1], task_index_to_eval)
                        elif train_hyperpara['dpsscl_transfer_score'].lower() == 'otce':
                            print("Compute OTCE Score")
                            for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                                if task_index_to_eval in task_training_order[:tmp_cnt]:
                                    # if it is a re-visited task
                                    continue
                                X_src, Y_src = torch.tensor(np.transpose(labeled_data[task_index_to_eval][0], (0, 2, 3, 1)).reshape([-1, x_dim]), dtype=torch.float), labeled_data[task_index_to_eval][1]
                                X_tar, Y_tar = torch.tensor(np.transpose(labeled_data[next_task_to_eval][0], (0, 2, 3, 1)).reshape([-1, x_dim]), dtype=torch.float), labeled_data[next_task_to_eval][1]
                                P, W = compute_coupling(X_src, X_tar, Y_src, Y_tar)
                                ce = compute_CE(P, Y_src, Y_tar)
                                lambda_w, lambda_ce = (1.0/200.0)*0.5, 0.5
                                otce = lambda_w*W + lambda_ce*ce
                                transfer_score[tmp_cnt] = otce
                        else:
                            raise NotImplementedError("Please choose task transferability score between following options: LEEP, OTCE")
                        print('\t', transfer_score)
                        transfer_score_hist[train_task_cnt + 1] = np.array(transfer_score)

                        learning_model.convert_tfVar_to_npVar(sess)
                        print('\n\t>>Change to new task!<<\n')
                    break

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" % (end_time - start_time))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x + 1]) for x in range(len(task_change_epoch) - 1)] + [(task_change_epoch[-1], learning_step + 1)]
    result_summary['task_changed_epoch'] = task_change_epoch

    ## confusion matrix
    if train_hyperpara['confusion_matrix']:
        result_summary['confusion_matrix'] = confusion_matrices

    result_summary['transferred_labelers_info'] = transferred_labelers_info
    result_summary['labelers_converged_epochs'] = labelers_converged_epochs
    result_summary['snuba_iterations'] = snuba_iterations
    result_summary['dpsscl_suitability_score'] = transfer_score_hist

    ## quality of the generated weak labeling functions
    if train_hyperpara['debug_mode']:
        result_summary['WLF_transfer_f1score'] = hfs_quality_transfer_f1score
        result_summary['WLF_transfer_jaccard'] = hfs_quality_transfer_jaccard
        result_summary['WLF_transfer_coverage'] = hfs_quality_transfer_coverage
        result_summary['WLF_transfer_trainacc'] = hfs_quality_transfer_trainacc_hist
        result_summary['WLF_nontransfer_f1score'] = hfs_quality_nontransfer_f1score
        result_summary['WLF_nontransfer_jaccard'] = hfs_quality_nontransfer_jaccard
        result_summary['WLF_nontransfer_coverage'] = hfs_quality_nontransfer_coverage
        result_summary['WLF_nontransfer_trainacc'] = hfs_quality_nontransfer_trainacc_hist


    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var

"""
Step 1, generating weak labelers by a modified version of Snuba while maintaining guarantee
"""

import os
import numpy as np
import sys
import torch
from labeler.label_generator import LabelGenerator
if torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
TASK_ROOT = os.path.join(PROJECT_ROOT, 'Data', 'task_data')


# Open the directory of one task, generate weak labelers for that directory
def generate_labeler_for_one_task(task_dir, task, data=None, task_similarity_score=None, task_similarity_score_threshold=1.0, sample_lf_from_earlier_task_prob=0.0, path_to_earlier_task_lf=None, lg_min=5, lg_max=25, lg_keep=10, return_snuba_iter_cnt=False, unified_score=False, finetune_transferred_wls=False, debug_mode=False):
    if data is None:
        # Load data from .npy directly (when running labeler generation and lifelong learner independently)
        X_l = np.load(os.path.join(task_dir, 'X_l.npy'))
        y_l = np.load(os.path.join(task_dir, 'y_l.npy'))
        X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
        y_u = np.load(os.path.join(task_dir, 'y_u.npy'))
    else:
        X_l, y_l, X_u, y_u = data

    # tuned hyperparams, don't change!
    if task in ['mnist', 'fashion']:
        init_acc_threshold = 0.85
        bootstrap_size_per_class = 30
    elif task in ['cifar10']:
        init_acc_threshold = 0.8
        bootstrap_size_per_class = 350
    elif task in ['mnist_5_way']:
        init_acc_threshold = 0.8
        bootstrap_size_per_class = 50
    elif task in ['cifar10_10_way']:
        init_acc_threshold = 0.6
        bootstrap_size_per_class = 750
    elif task in ['cifar100_5_way']:
        init_acc_threshold = 0.6
        bootstrap_size_per_class = 200
    else:
        raise NotImplementedError

    # guess 100 times for each weak labeler, initially keep 20, then add keep 1 each time
    # max num weak labelers = 25, min = 5
    lg = LabelGenerator(X_u=X_u, y_u=y_u, X_l=X_l, y_l=y_l, task=task, num_guesses=100, keep=lg_keep,
                        max_labelers=lg_max, min_labelers=lg_min,
                        init_acc_threshold=init_acc_threshold,
                        bootstrap_size_per_class=bootstrap_size_per_class)
    task_parent_dir = os.path.dirname(task_dir)
    snuba_iter_cnt, hfs_quality_scores = -1, None
    if task_similarity_score is None or sample_lf_from_earlier_task_prob<=0.0 or path_to_earlier_task_lf is None:
        return_outputs = lg.generate_snuba_lfs(log_file=os.path.join(task_parent_dir, 'log.txt'), return_snuba_iter_cnt=return_snuba_iter_cnt, debug_mode=debug_mode)
        if return_snuba_iter_cnt and debug_mode:
            hfs, snuba_iter_cnt, hfs_quality_scores = return_outputs[0], return_outputs[1], return_outputs[2]
        elif return_snuba_iter_cnt:
            hfs, snuba_iter_cnt = return_outputs[0], return_outputs[1]
        elif debug_mode:
            hfs, hfs_quality_scores = return_outputs[0], return_outputs[1]
        else:
            hfs = return_outputs
    else:
        print("\tUse pre-trained weak labelers")
        return_outputs = lg.generate_snuba_lfs(log_file=os.path.join(task_parent_dir, 'log.txt'), sample_earlier_lfs=True, sample_earlier_lfs_prob=sample_lf_from_earlier_task_prob, path_earlier_lfs=path_to_earlier_task_lf, task_similarity_score=task_similarity_score, task_similarity_score_threshold=task_similarity_score_threshold, return_snuba_iter_cnt=return_snuba_iter_cnt, unified_score=unified_score, finetune_transferred_wls=finetune_transferred_wls, debug_mode=debug_mode)
        if return_snuba_iter_cnt and debug_mode:
            hfs, snuba_iter_cnt, hfs_quality_scores = return_outputs[0], return_outputs[1], return_outputs[2]
        elif return_snuba_iter_cnt:
            hfs, snuba_iter_cnt = return_outputs[0], return_outputs[1]
        elif debug_mode:
            hfs, hfs_quality_scores = return_outputs[0], return_outputs[1]
        else:
            hfs = return_outputs

    labelers_transferred_task_ids, labelers_converged_epoch_ratio = [], []
    for i in range(len(hfs)):
        torch.save(hfs[i].state_dict(), os.path.join(task_dir, 'lf' + str(i) + '.pt'))
        labelers_transferred_task_ids.append(hfs[i].earlier_task_id)
        labelers_converged_epoch_ratio.append(hfs[i].converged_epoch_ratio)
    print(task_dir + ': ' + str(len(hfs)) + ' labelers saved')
    # returns a list of task_id that weak labeler uses knowledge transferred from
    # task_id==-1 means training from scratch
    return labelers_transferred_task_ids, labelers_converged_epoch_ratio, snuba_iter_cnt, hfs_quality_scores


# Generate and save weak labelers by modified Snuba
def generate_weak_labelers(task):
    if task in ['mnist', 'fashion', 'cifar10']:
        task_parent_dir = os.path.join(TASK_ROOT, task + '_bin')
    elif task in ['mnist_5_way', 'cifar10_10_way', 'cifar100_5_way']:
        task_parent_dir = os.path.join(TASK_ROOT, task)
    else:
        raise NotImplementedError

    for d in os.listdir(task_parent_dir):
        task_dir = os.path.join(task_parent_dir, d)
        generate_labeler_for_one_task(task_dir=task_dir, task=task)
    return


def generate_weak_labelers_main(args):
    if len(args) < 2:
        dataset = 'mnist'
    else:
        dataset = args[1]
    if dataset in ['mnist', 'fashion', 'cifar10', 'mnist_5_way', 'cifar10_10_way', 'cifar100_5_way']:
        generate_weak_labelers(dataset)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    generate_weak_labelers_main(sys.argv)


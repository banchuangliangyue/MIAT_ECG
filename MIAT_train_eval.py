from __future__ import division
import os
import json
import pickle
import random
import wfdb
import math
import numpy as np

np.set_printoptions(suppress=True)
import warnings
import traceback
import time
from time import strftime, localtime
import matplotlib.pyplot as plt

plt.switch_backend('agg')
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, \
    roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from sklearn.metrics import confusion_matrix, recall_score
from scipy import stats
from decimal import Decimal
import argparse
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.nn.functional as F
import MIAT_utils as utils
from torch.backends import cudnn
import torch.nn as nn
from torch.autograd import Variable
from tsne import vis
import block_network


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')


parser = argparse.ArgumentParser()
parser.add_argument("--lr", dest="init_lr", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda", dest="_lambda", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda_e", dest="ent", type=float, metavar='<float>', default=0.001)
parser.add_argument("--lambda_i", dest="inconsist", type=float, metavar='<float>', default=0.001)
parser.add_argument("--alpha", dest="mix_alpha", type=float, metavar='<float>', default=2.)
parser.add_argument("--run_id", dest="id", type=int, metavar='<int>', default=0)
parser.add_argument("--weight", dest="weights_decay", type=float, metavar='<float>', default=0.005)
parser.add_argument("--n", dest="nb_epoch", type=int, metavar='<int>', default=0)
parser.add_argument('--s', dest="source", type=str, default='DS1')
parser.add_argument('--t', dest="target", type=str, default='DS2')
parser.add_argument("--thresh", dest="thresh", type=float, metavar='<float>', default=0.99)
parser.add_argument("--r", dest="radius", type=float, metavar='<float>', default=0.5)
parser.add_argument("--focal", dest="use_focal", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--decay", dest="use_decay", type=str2bool, metavar='<bool>', default=True)
parser.add_argument("--logit", dest="logit", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--class", dest="class_num", type=int, metavar='<int>', default=5)
parser.add_argument("--seed", dest="seeds", type=int, metavar='<int>', default=666)
parser.add_argument("--epochs", dest="EPOCHS", type=int, metavar='<int>', default=150)
parser.add_argument("--num", dest="init_cand_num", type=int, metavar='<int>', default=1000)
parser.add_argument('--gpu', dest="GPU", type=str, default=7, help='cuda_visible_devices')
parser.add_argument("--mix", dest="mixup", type=str2bool, metavar='<bool>', default=True)
parser.add_argument("--vat", dest="VAT", type=str2bool, metavar='<bool>', default=True)
parser.add_argument("--prevat", dest="PreVAT", type=str2bool, metavar='<bool>', default=False)
parser.add_argument("--tsne", dest="TSNE", type=str2bool, metavar='<bool>', default=False)

args = parser.parse_args()


cudnn.benchmark = False  # if benchmark=True, deterministic will be False
cudnn.deterministic = True
torch.manual_seed(666)
torch.cuda.manual_seed(666)
torch.cuda.manual_seed_all(666)
random.seed(666)
np.random.seed(666)

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.GPU)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

batch_size = 128
test_batch_size = 128*4
N_CLASS = args.class_num
nb_epoch = args.nb_epoch
beta = (0.9, 0.999)

tmp_dir = 'exp1/'
weight_path = 'exp1/weights/'
train_results_save_path = 'exp1/train_results/'
train_results_img_save_path = 'exp1/train_results/imgs/'

if not (os.path.exists(train_results_img_save_path)):
    os.makedirs(train_results_img_save_path)
if not (os.path.exists(train_results_save_path)):
    os.makedirs(train_results_save_path)
if not (os.path.exists(weight_path)):
    os.makedirs(weight_path)

with open('exp1/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
    f.write('\n\n==============' + __file__ + '====================\n')
    f.write(args.__str__() + '\n')

cls_num_list = torch.Tensor(np.array([45847, 944, 3788, 414, 8])).cuda()

print('N_class:', N_CLASS)

x_train, y_train, x_val, y_val, x_test, y_test, x_target, y_target, class_center = utils.get_dataset(args.source,
                                                                                       args.target, n_class=N_CLASS)

label_target = torch.zeros((x_target.shape[0])).long()

def pre_train(train_x, train_y, target_x, epoch):

    G.train()
    F1.train()
    F2.train()
    Ft.train()

    p1_loss = []
    p2_loss = []
    pt_loss = []
    w_diff_loss = []
    total_loss = []
    p1_acc = []
    p2_acc = []
    pt_acc = []



    gen_source_only_batch = utils.batch_generator([train_x, train_y], batch_size, shuffle=True)

    num_steps = train_x.shape[0] // batch_size + 1


    for i in range(num_steps):

        x0, y0 = gen_source_only_batch.__next__()
        x0, y0 = x0.to(device), y0.to(device)
        zero_grad()
        # w_loss = args.inconsist * torch.mm(F1.classifier[0].weight, F2.classifier[0].weight.transpose(1, 0)).mean()
        # w_loss = args.inconsist * (F.cosine_similarity(F1.classifier[0].weight, F2.classifier[0].weight, dim=1).mean())
        w_loss = torch.tensor([0.]).float().to(device)
        if args.PreVAT:
            lam = np.random.beta(args.mix_alpha, args.mix_alpha)
            index = np.random.permutation(x0.shape[0])
            x_0, y_0 = x0[index], y0[index]
            mixed_x = lam * x0 + (1 - lam) * x_0
            feature = G(mixed_x)
            # feature = G(x0)
            pred_1 = F1(feature)
            pred_2 = F2(feature)
            pred_t = Ft(feature)

            pred_1_loss = lam * loss_func(pred_1, y0) + (1 - lam) * loss_func(pred_1, y_0)
            pred_2_loss = lam * loss_func(pred_2, y0) + (1 - lam) * loss_func(pred_2, y_0)
            pred_t_loss = lam * loss_func(pred_t, y0) + (1 - lam) * loss_func(pred_t, y_0)

        else:
            feature = G(x0)
            pred_1 = F1(feature)
            pred_2 = F2(feature)
            pred_t = Ft(feature)
            pred_1_loss = loss_func(pred_1, y0)
            pred_2_loss = loss_func(pred_2, y0)
            pred_t_loss = loss_func(pred_t, y0)

        # w_loss = w_loss_ratio * torch.mm(F1.classifier[0].weight, F2.classifier[0].weight.transpose(1, 0)).mean()
        sum_loss = pred_1_loss + pred_2_loss + pred_t_loss + w_loss
        sum_loss.backward()
        optimizer_F.step()
        optimizer_F1.step()
        optimizer_F2.step()
        optimizer_Ft.step()

        pred_label_1 = torch.argmax(pred_1, dim=1)
        pred_1_acc = (pred_label_1 == y0).float().mean().detach().cpu().numpy()
        pred_label_2 = torch.argmax(pred_2, dim=1)
        pred_2_acc = (pred_label_2 == y0).float().mean().detach().cpu().numpy()
        pred_label_t = torch.argmax(pred_t, dim=1)
        pred_t_acc = (pred_label_t == y0).float().mean().detach().cpu().numpy()

        p1_loss.append(pred_1_loss.item())
        p2_loss.append(pred_2_loss.item())
        pt_loss.append(pred_t_loss.item())
        w_diff_loss.append(w_loss.item())
        total_loss.append(sum_loss.item())
        p1_acc.append(pred_1_acc)
        p2_acc.append(pred_2_acc)
        pt_acc.append(pred_t_acc)


    print('---Num_steps: {}, S_samples: {}, T_nums: {}, S+T_samples: {}, '
          ' Source_batch: {} '.format(num_steps, train_x.shape[0], 0, train_x.shape[0], x0.shape[0]))


    return np.mean(total_loss), np.mean(p1_loss), np.mean(p2_loss), \
           np.mean(pt_loss), np.mean(w_diff_loss), np.mean(p1_acc), \
           np.mean(p2_acc), np.mean(pt_acc)



def train(train_x, train_y, target_x, model, optimizer, epoch, w_loss_ratio=None, new_data=None,
          new_label=None, new_prob=None, target_weights=None, val_f1_class=None):


    G.train()
    F1.train()
    F2.train()
    Ft.train()

    p1_loss = []
    p2_loss = []
    pt_loss = []
    w_diff_loss = []
    total_loss = []
    p1_acc = []
    p2_acc = []
    pt_acc = []


    if len(new_data) == 0:
        new_data = train_x
        new_label = train_y
        print('====No Pseudo Label!  Replaced by source data====')
    train_dom_label = torch.zeros(train_x.shape[0]).float()
    new_dom_label = torch.ones(new_data.shape[0]).float()
    source_dom_label = torch.cat((train_dom_label, new_dom_label))
    source_train = torch.cat((train_x, new_data), 0)
    source_label = torch.cat((train_y, new_label))

    gen_source_batch = utils.batch_generator([source_train, source_label,], batch_size // 2, shuffle=True)
    # gen_new_batch = utils.batch_generator([new_data, new_label, new_prob], batch_size, shuffle=True)

    gen_new_batch = utils.batch_generator([new_data, new_label], batch_size, shuffle=True)

    gen_all_target_batch = utils.batch_generator([x_target, label_target], batch_size, shuffle=True)


    gen_source_only_batch = utils.batch_generator([train_x, train_y], batch_size, shuffle=True)

    target_class = np.array([torch.eq(new_label, 0).sum().item(), torch.eq(new_label, 1).sum().item(),
                             torch.eq(new_label, 2).sum().item(), torch.eq(new_label, 3).sum().item(),
                             torch.eq(new_label, 4).sum().item()])

    print('target_class:', target_class)
    target_class = torch.Tensor(target_class).cuda()
    # print('target_weight:', target_weights)
    # src_tgt_cls_num = torch.Tensor(np.array([torch.eq(source_label, 0).sum().item(), torch.eq(source_label, 1).sum().item(),
    #                          torch.eq(source_label, 2).sum().item(), torch.eq(source_label, 3).sum().item(),
    #                          torch.eq(source_label, 4).sum().item()])).cuda()
    # print('src_tgt_cls_num:', src_tgt_cls_num)
    src_cls_num = torch.Tensor(np.array([torch.eq(train_y, 0).sum().item(), torch.eq(train_y, 1).sum().item(),
                                          torch.eq(train_y, 2).sum().item(), torch.eq(train_y, 3).sum().item(),
                                          torch.eq(train_y, 4).sum().item()])).cuda()
    # print('src_cls_num:', src_cls_num)
    '''训练2k-way joint predictor时 num_steps= source_train.shape[0] // batch_size'''
    # num_steps = (train_x.shape[0]+x_target.shape[0]) // batch_size + 1
    # num_steps = 500
    num_steps = int(source_train.shape[0] // batch_size * 2 + 1)
    # num_steps = train_x.shape[0] // batch_size + 1

    for i in range(num_steps):

        ###Traning F, F1, F2
        x0, y0 = gen_source_batch.__next__()
        x0, y0 = x0.to(device), y0.to(device)

        x1, y1 = gen_source_only_batch.__next__()
        x1, y1 = x1.to(device), y1.to(device)

        x2, y2 = gen_all_target_batch.__next__()
        x2, y2 = x2.to(device), y2.to(device)

        x3, y3  = gen_new_batch.__next__()
        x3, y3  = x3.to(device), y3.to(device)

        zero_grad()
        feature = G(x0)
        pred_1 = F1(feature)
        pred_2 = F2(feature)
        pred_1_loss = loss_func(pred_1, y0)
        pred_2_loss = loss_func(pred_2, y0)

        loss_s = pred_1_loss + pred_2_loss
        loss_s.backward()
        optimizer_F.step()
        optimizer_F1.step()
        optimizer_F2.step()


        ##Traning F, Ft
        zero_grad()
        # lam = np.random.beta(args.mix_alpha, args.mix_alpha)
        # lam = max(lam, 1 - lam)
        if args.mixup:
            lam = np.random.beta(args.mix_alpha, args.mix_alpha)
            lam = max(lam, 1 - lam)
            feature_s = G(x1)
            feature_t = G(x2)
            mix_feature = lam * feature_s + (1 - lam) * feature_t
            mix_input = lam * x1 + (1 - lam) * x2
            mix_output = G(mix_input)
            w_loss = args._lambda * F.mse_loss(mix_feature, mix_output)

            if args.VAT:
                # lam = np.random.beta(args.mix_alpha, args.mix_alpha)
                index = np.random.permutation(x3.shape[0])
                x_3, y_3 = x3[index], y3[index]
                mixed_x = lam * x3 + (1 - lam) * x_3
                feature_t = G(mixed_x)
                pred_t = Ft(feature_t)
                pred_t_loss = lam * loss_func(pred_t, y3) + (1 - lam) * loss_func(pred_t, y_3) + w_loss

            else:
                feature_t = G(x3)
                pred_t = Ft(feature_t)
                pred_t_loss = loss_func(pred_t, y3) + w_loss


        else:
            w_loss = torch.tensor([0.]).float().to(device)
            if args.VAT:
                lam = np.random.beta(args.mix_alpha, args.mix_alpha)
                index = np.random.permutation(x3.shape[0])
                x_3, y_3 = x3[index], y3[index]
                mixed_x = lam * x3 + (1 - lam) * x_3
                feature_t = G(mixed_x)
                pred_t = Ft(feature_t)
                pred_t_loss = lam * loss_func(pred_t, y3) + (1 - lam) * loss_func(pred_t, y_3)

            else:
                feature_t = G(x3)
                pred_t = Ft(feature_t)
                pred_t_loss = loss_func(pred_t, y3)

        pred_t_loss.backward()
        optimizer_F.step()
        optimizer_Ft.step()

        sum_loss = loss_s + pred_t_loss
        pred_label_1 = torch.argmax(pred_1, dim=1)
        pred_1_acc = (pred_label_1 == y0).float().mean().detach().cpu().numpy()
        pred_label_2 = torch.argmax(pred_2, dim=1)
        pred_2_acc = (pred_label_2 == y0).float().mean().detach().cpu().numpy()
        pred_label_t = torch.argmax(pred_t, dim=1)
        pred_t_acc = (pred_label_t == y3).float().mean().detach().cpu().numpy()

        p1_loss.append(pred_1_loss.item())
        p2_loss.append(pred_2_loss.item())
        pt_loss.append(pred_t_loss.item())
        w_diff_loss.append((w_loss).item())
        total_loss.append(sum_loss.item())
        p1_acc.append(pred_1_acc)
        p2_acc.append(pred_2_acc)
        pt_acc.append(pred_t_acc)

    print('---Num_steps: {}, S_samples: {}, T_nums: {}, S+T_samples: {}, '
          ' Target_batch: {},  Source_Target_batch: {} '.format(num_steps, train_x.shape[0], new_data.shape[0],
                                                                source_train.shape[0], x2.shape[0], x0.shape[0]))

    return np.mean(total_loss), np.mean(p1_loss), np.mean(p2_loss), \
           np.mean(pt_loss), np.mean(w_diff_loss), np.mean(p1_acc), \
           np.mean(p2_acc), np.mean(pt_acc)


def sample_candidatas(dataset, labels, candidates_num, shuffle=True):
    indices = np.arange(len(dataset))
    if shuffle:
        # indices = np.arange(len(dataset))
        np.random.shuffle(indices)
    excerpt = indices[0:candidates_num]
    candidates = dataset[excerpt]
    true_label = labels[excerpt]
    return candidates, true_label


def assign_pseudo_label(target_x, label_target, model, epoch, init_num, logit, thresh=None):
    # if epoch == 0:
    #     rate = 1000
    # else:
    #     rate= min(int((epoch + 2) * target_x.shape[0] / 20), target_x.shape[0])
    # print('rate:', rate)
    # rate = target_x.shape[0]
    # cand_data, cand_label = target_x, label_target
    # thresh = min(args.thresh + epoch * 0.006, 1)

    rate = min(max(int((epoch + 1) / (EPOCHS // 2) * target_x.shape[0]), init_num), target_x.shape[0])
    cand_data, cand_label = sample_candidatas(target_x, label_target, rate, shuffle=True)

    G.eval()
    F1.eval()
    F2.eval()
    Ft.eval()

    with torch.no_grad():
        step = 0
        pred1_stack = np.zeros((0, N_CLASS))
        pred2_stack = np.zeros((0, N_CLASS))
        predt_stack = np.zeros((0, N_CLASS))
        feature_stack = np.zeros((0, 2560))

        stack_num = np.ceil(cand_data.shape[0] / test_batch_size)
        gen_target_batch = utils.batch_generator(
            [cand_data, cand_label], test_batch_size, shuffle=False, test=True)

        while step < stack_num:
            x1, y1 = gen_target_batch.__next__()
            x1, y1 = x1.to(device), y1.to(device)
            fea = G(x1)
            pred_1 = F1(fea)
            pred_2 = F2(fea)
            pred_1 = pred_1.detach().cpu().numpy()
            pred_2 = pred_2.detach().cpu().numpy()
            # pred_t = pred_t.detach().cpu().numpy()
            feature = fea.detach().cpu().numpy()
            pred1_stack = np.r_[pred1_stack, pred_1]
            pred2_stack = np.r_[pred2_stack, pred_2]

            feature_stack = np.r_[feature_stack, feature]
            step += 1

        pred1_stack, pred2_stack = torch.tensor(pred1_stack), torch.tensor(pred2_stack)

        feature_stack = torch.tensor(feature_stack)
        assert cand_data.shape[0] == pred1_stack.shape[0]
        if not logit:
            # print('---logit:', logit)
            pred1_stack, pred2_stack = F.softmax(pred1_stack, dim=1), F.softmax(pred2_stack, dim=1)



        new_data, new_label, new_prob = utils.judge_func(cand_data,
                                                       pred1_stack,
                                                       pred2_stack,
                                                       feature_stack,
                                                       thresh=None,
                                                       num_class=N_CLASS)

    print('---target_total_iters: {}, current_assign_rate: {}, new_data_nums: {}'.
          format(stack_num, rate, new_data.shape[0]))


    return new_data, new_label


def val(val_x, val_y, model, epoch, best_val_f1, w_loss_ratio=None, alpha=None):
    G.eval()
    F1.eval()
    F2.eval()
    Ft.eval()

    with torch.no_grad():

        p1_loss = []
        p2_loss = []
        pt_loss = []
        w_diff_loss = []
        total_loss = []
        p1_acc = []
        p2_acc = []
        pt_acc = []
        size_val = 0
        y_true = np.array([]).reshape((0, 1))
        y_pred = np.array([]).reshape((0, 1))

        gen_val_batch = utils.batch_generator(
            [val_x, val_y], test_batch_size, test=True)
        num_iter = int(val_x.shape[0] // test_batch_size) + 1
        step = 0

        while step < num_iter:
            x1, y1 = gen_val_batch.__next__()
            x1, y1 = x1.to(device), y1.to(device)

            features = G(x1)
            pred_1 = F1(features)
            pred_2 = F2(features)
            pred_t = Ft(features)

            pred_1_loss = loss_func(pred_1, y1)
            pred_2_loss = loss_func(pred_2, y1)
            pred_t_loss = loss_func(pred_t, y1)
            # w_loss = w_loss_ratio * torch.mm(F1.classifier[0].weight, F2.classifier[0].weight.transpose(1, 0)).mean()
            # w_loss = w_loss_ratio *discrepancy(pred_1, pred_2)
            # w_loss = args.inconsist * (F.cosine_similarity(F1.classifier[0].weight, F2.classifier[0].weight, dim=1).mean())
            w_loss = pred_t_loss
            sum_loss = pred_1_loss + pred_2_loss + pred_t_loss
            pred_label_1 = torch.argmax(pred_1, dim=1)
            pred_1_acc = (pred_label_1 == y1).float().mean().detach().cpu().numpy()
            pred_label_2 = torch.argmax(pred_2, dim=1)
            pred_2_acc = (pred_label_2 == y1).float().mean().detach().cpu().numpy()
            pred_label_t = torch.argmax(pred_t, dim=1)
            pred_t_acc = (pred_label_t == y1).float().mean().detach().cpu().numpy()

            pred = pred_label_t.detach().cpu().numpy().reshape(y1.shape[0], 1)
            label = y1.detach().cpu().numpy().reshape(y1.shape[0], 1)
            y_pred = np.concatenate((y_pred, pred), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)

            p1_loss.append(pred_1_loss.item())
            p2_loss.append(pred_2_loss.item())
            pt_loss.append(pred_t_loss.item())
            w_diff_loss.append((w_loss).item())
            total_loss.append(sum_loss.item())
            p1_acc.append(pred_1_acc)
            p2_acc.append(pred_2_acc)
            pt_acc.append(pred_t_acc)
            size_val += x1.shape[0]

            step += 1

        print('---num_iter: {}, size_val: {}'.format(num_iter, size_val))


        Val_f1 = f1_score(y_true, y_pred, average=None)
        val_f1 = np.mean(Val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                'G_state_dict': G.state_dict(),
                'F1_state_dict': F1.state_dict(),
                'F2_state_dict': F2.state_dict(),
                'Ft_state_dict': Ft.state_dict(),
                'epoch': epoch,
                'best val f1': best_val_f1,
                 }, weight_path + str(id) + '_torch_best_f1_TriTraning_model.pt')

            print("best f1: {:.4f}".format(best_val_f1))
        else:
            print("=======val f1: {:.4f}, but not the best f1======== ".format(val_f1))
        # else:
        #     print("=======val f1: {:.4f} ======== ".format(val_f1))

    return np.mean(total_loss), np.mean(p1_loss), np.mean(p2_loss), np.mean(pt_loss), np.mean(w_diff_loss), \
           np.mean(p1_acc), np.mean(p2_acc), np.mean(pt_acc), best_val_f1, val_f1, Val_f1


def test(test_x, test_y, model, domain='target_test'):
    print('Load model...')
    checkpoint = torch.load(weight_path + str(id) + '_torch_best_f1_TriTraning_model.pt')
    print('epoch:', checkpoint['epoch'])
    print('best val f1:', checkpoint['best val f1'])

    G.load_state_dict(checkpoint['G_state_dict'])
    F1.load_state_dict(checkpoint['F1_state_dict'])
    F2.load_state_dict(checkpoint['F2_state_dict'])
    Ft.load_state_dict(checkpoint['Ft_state_dict'])
    G.eval()
    F1.eval()
    F2.eval()
    Ft.eval()

    with torch.no_grad():
        y_true = np.array([]).reshape((0, 1))
        y_pred = np.array([]).reshape((0, 1))
        feature = np.array([]).reshape((0, 2560))
        gen_target_batch = utils.batch_generator(
            [test_x, test_y], test_batch_size, test=True)
        num_iter = int(test_x.shape[0] // test_batch_size) + 1
        print('---test_num_iter:', num_iter)
        step = 0
        while step < num_iter:
            x1, y1 = gen_target_batch.__next__()
            x1, y1 = x1.to(device), y1.to(device)
            fea = G(x1)
            pred_1 = F1(fea)
            pred_2 = F2(fea)
            pred_t = Ft(fea)
            pred_label_t = torch.argmax(pred_t, dim=1)
            pred = pred_label_t.detach().cpu().numpy().reshape(y1.shape[0], 1)
            label = y1.detach().cpu().numpy().reshape(y1.shape[0], 1)
            fea = fea.detach().cpu().numpy().reshape(y1.shape[0], -1)
            y_pred = np.concatenate((y_pred, pred), axis=0)
            y_true = np.concatenate((y_true, label), axis=0)
            feature = np.concatenate((feature, fea), axis=0)
            step += 1

    # if domain != 'target_train':
    print(classification_report(y_true, y_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))

    print('========== confusion matrix ==========')
    print('domian:', domain)
    print(confusion_matrix(y_true, y_pred))

    # print('===Save features for tSNE !!=======')
    # np.savez(train_results_save_path + 'exp_' + str(args.id) + "_" + domain + '_fea', feature=feature, label=y_true)

    if domain != 'target_train':
        with open('../exp1/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
            f.write('best epoch:' + str(checkpoint['epoch']) + '  ' + 'best val f1:' + str(
                checkpoint['best val f1']) + '\n')
            f.write(classification_report(y_true, y_pred, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4) + '\n')
            f.write(str(confusion_matrix(y_true, y_pred)) + '\n')



def plot_statistic(train_statistic, val_statistic, i=0):
    iters = [i + 1 for i in range(0, len(train_statistic['total_loss']))]
    plt.figure(figsize=(10, 10))
    plt.plot(iters, train_statistic['w_diff_loss'], color='red', label='weight_diff_loss')
    plt.plot(iters, train_statistic['p1_acc'], color='green', label='train_p1_acc')
    plt.plot(iters, train_statistic['p2_acc'], color='blue', label='train_p2_acc')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(train_results_img_save_path + 'atda_train_process_' + str(i) + '_.png', bbox_inches='tight')

    plt.figure(figsize=(30, 10))
    plt.subplot(131)
    plt.plot(iters, train_statistic['p1_loss'], color='red', label='train_p1_loss')
    plt.plot(iters, val_statistic['p1_loss'], color='darkred', label='val_p1_loss')
    plt.plot(iters, train_statistic['p2_loss'], color='green', label='train_p2_loss')
    plt.plot(iters, val_statistic['p2_loss'], color='lightgreen', label='val_p2_loss')
    plt.plot(iters, train_statistic['pt_loss'], color='blue', label='train_pt_loss')
    plt.plot(iters, val_statistic['pt_loss'], color='slateblue', label='val_pt_loss')
    plt.plot(iters, train_statistic['w_diff_loss'], color='pink', label='weight_diff_loss')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(132)
    plt.plot(iters, train_statistic['p1_acc'], color='red', label='train_p1_acc')
    plt.plot(iters, val_statistic['p1_acc'], color='darkred', label='val_p1_acc')
    plt.plot(iters, train_statistic['p2_acc'], color='green', label='train_p2_acc')
    plt.plot(iters, val_statistic['p2_acc'], color='lightgreen', label='val_p2_acc')
    plt.plot(iters, train_statistic['pt_acc'], color='blue', label='train_pt_acc')
    plt.plot(iters, val_statistic['pt_acc'], color='slateblue', label='val_pt_acc')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)

    plt.subplot(133)
    plt.plot(iters, val_statistic['val_f1'], color='red', label='val_f1')
    plt.xlabel('epoch')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(train_results_img_save_path + 'atda_loss_acc_f1_' + str(i) + '_.png', bbox_inches='tight')


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=1) - F.softmax(out2, dim=1)))


def zero_grad():
    optimizer_F.zero_grad()
    optimizer_F1.zero_grad()
    optimizer_F2.zero_grad()
    optimizer_Ft.zero_grad()

def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.xavier_normal(m.weight)
    if type(m) == nn.Linear:
        nn.init.xavier_normal(m.weight)

if __name__ == '__main__':
    since = time.time()
    LEARNING_RATE = args.init_lr
    WEIGHT_DECAY = 5e-4
    MOMENTUM = 0.9
    EPOCHS = args.EPOCHS
    id = args.id
    logit = args.logit

    # if args.use_focal:
    #     # loss_func = utils.FocalLoss(alpha=args.focal_loss_alpha)
    #     loss_func = utils.focal_loss_zhihu
    #     print('==use foacl loss====')
    # else:
    loss_func = F.cross_entropy
    print('==use CE loss====')

    if args.target == 'DS2' and args.source == 'DS1':
        G = block_network.AlexNetforEcg_DS1_to_DS2().to(device)
    else:
        G = block_network.AlexNetforEcg_DS1_to_DS2().to(device)
        # G = block_network.AlexNetforEcg_mitdb_to_svdb().to(device)

    F1 = block_network.EcgClassifier(dropout_keep=0.5, num_classes=N_CLASS).to(device)
    F2 = block_network.EcgClassifier(dropout_keep=0.5, num_classes=N_CLASS).to(device)
    Ft = block_network.EcgClassifier(dropout_keep=0.3, num_classes=N_CLASS).to(device)

    print('G:', G)
    print('F2:', F2)
    print('Ft:', Ft)

    optimizer_F = torch.optim.Adam(G.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)
    optimizer_F1 = torch.optim.Adam(F1.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)
    optimizer_F2 = torch.optim.Adam(F2.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)
    optimizer_Ft = torch.optim.Adam(Ft.parameters(), lr=LEARNING_RATE, weight_decay=args.weights_decay)

    optimizer = [optimizer_F, optimizer_F1, optimizer_F2, optimizer_Ft]
    model = [G, F1, F2, Ft]
    train_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                       'p1_acc': [], 'p2_acc': [], 'pt_acc': [], 'new_labeled_samples': []}
    val_statistic = {'total_loss': [], 'p1_loss': [], 'p2_loss': [], 'pt_loss': [], 'w_diff_loss': [],
                     'p1_acc': [], 'p2_acc': [], 'pt_acc': [], 'val_f1': []}

    pseudo_acc_per_step = []
    labeled_samples_per_step = []

    print('\n==============Training...==================')

    best_val_f1 = 0
    best_train_acc = 0
    best_loss = 1000
    step = 0

    for e in range(0, nb_epoch):


        lr_t = LEARNING_RATE / (1 + 10 * e / nb_epoch) ** 0.75
        if args.use_decay:
            print('\n==Using lr decay====')
            optimizer_F = torch.optim.Adam(G.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_F1 = torch.optim.Adam(F1.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_F2 = torch.optim.Adam(F2.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_Ft = torch.optim.Adam(Ft.parameters(), lr=lr_t, weight_decay=args.weights_decay)


        print('\n###current lr:{:.6f}  weight_decay:{:.4f}  lambda:{:.4f}  mix_alpha:{:.2f}  mix:{}  vat:{}  prevat:{} '
              .format(optimizer_F.param_groups[0]['lr'], args.weights_decay, args._lambda, args.mix_alpha, args.mixup,
                      args.VAT, args.PreVAT ))

        total_loss, p1_loss, p2_loss, pt_loss, w_diff_loss, p1_acc, p2_acc, pt_acc = pre_train(x_train, y_train, optimizer, e)

        print('---Pre_Train EPOCH: {}/{}, total_loss: {:.4f}, p1_loss: {:.4f}, p2_loss: {:.4f}, '
              'pt_loss: {:.4f}, w_diff_loss: {:.6f}, p1_acc: {:.4f}, p2_acc: {:.4f}, pt_acc: {:.4f}'
              .format(e + 1, nb_epoch , total_loss, p1_loss, p2_loss, pt_loss, w_diff_loss, p1_acc, p2_acc, pt_acc))


        if e == nb_epoch - 1:
            val_total_loss, val_p1_loss, val_p2_loss, val_pt_loss, val_w_diff_loss, val_p1_acc, val_p2_acc, \
                val_pt_acc, best_val_f1, current_val_f1, val_f1_cls = val(x_val, y_val, model, e, best_val_f1)

            best_val_f1 = 0

    new_data, new_label = assign_pseudo_label(x_target, label_target, model, 0, args.init_cand_num, logit,)



    for e in range(0, EPOCHS):
        # if e > 0 and e % 50 == 0:
        #     optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.5
        # optimizer.param_groups[0]['lr'] = LEARNING_RATE * (1.0 + math.cos((e + 1) * math.pi / EPOCHS))
        lr_t = LEARNING_RATE / (1 + 10 * e / EPOCHS) ** 0.75
        if e > 150:
            lr_t = LEARNING_RATE / 10

        # w_loss_ratio = args._lambda
        # w_loss_ratio = args.ent*(2 / (1 + math.exp(-10 * (e) / EPOCHS)) - 1)
        w_loss_ratio = args.ent* math.exp(-2 * (1 - e / EPOCHS))
        # if e <= 80:
        #     w_loss_ratio = args._lambda * math.exp(-5 * (1 - e / 80) ** 2)
        #     # w_loss_ratio = args._lambda * ((e + 1) / 100)
        #     # w_loss_ratio = 0
        # else:
        #     w_loss_ratio = args._lambda
        if args.use_decay:
            print('\n==Using lr decay====')
            optimizer_F = torch.optim.Adam(G.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_F1 = torch.optim.Adam(F1.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_F2 = torch.optim.Adam(F2.parameters(), lr=lr_t, weight_decay=args.weights_decay)
            optimizer_Ft = torch.optim.Adam(Ft.parameters(), lr=lr_t, weight_decay=args.weights_decay)

        print('\n###current lr:{:.6f}  weight_decay:{:.4f}  lambda:{:.4f}  mix_alpha:{:.2f}  mix:{}  vat:{}  PreVAT:{}  '
              'pre_epoch:{}'
              .format(optimizer_F.param_groups[0]['lr'], args.weights_decay, args._lambda, args.mix_alpha, args.mixup,
                      args.VAT, args.PreVAT, args.nb_epoch))



        total_loss, p1_loss, p2_loss, pt_loss, w_diff_loss, p1_acc, \
            p2_acc, pt_acc = train(x_train, y_train, x_target, model, optimizer, e, w_loss_ratio, new_data, new_label)

        print('---Train EPOCH: {}/{}, total_loss: {:.4f}, p1_loss: {:.4f}, p2_loss: {:.4f}, '
              'pt_loss: {:.4f}, w_diff_loss: {:.6f}, p1_acc: {:.4f}, p2_acc: {:.4f}, pt_acc: {:.4f}'
              .format(e + 1, EPOCHS, total_loss, p1_loss, p2_loss, pt_loss, w_diff_loss, p1_acc, p2_acc, pt_acc))

        new_data, new_label = assign_pseudo_label(x_target, label_target, model, e, args.init_cand_num, logit)

        train_statistic['total_loss'].append(total_loss)
        train_statistic['p1_loss'].append(p1_loss)
        train_statistic['p2_loss'].append(p2_loss)
        train_statistic['pt_loss'].append(pt_loss)
        train_statistic['w_diff_loss'].append(w_diff_loss)
        train_statistic['p1_acc'].append(p1_acc)
        train_statistic['p2_acc'].append(p2_acc)
        train_statistic['pt_acc'].append(pt_acc)
        train_statistic['new_labeled_samples'].append(len(new_label))


        val_total_loss, val_p1_loss, val_p2_loss, val_pt_loss, val_w_diff_loss, val_p1_acc, \
            val_p2_acc, val_pt_acc, best_val_f1, current_val_f1, val_f1_cls = val(x_val, y_val, model, e, best_val_f1,
                                                                                  w_loss_ratio)

        print('===val_total_loss: {:.4f}, val_p1_loss: {:.4f}, val_p2_loss: {:.4f}, '
              'val_pt_loss: {:.6f}, val_w_diff_loss: {:.6f}, val_p1_acc: {:.4f}, '
              'val_p2_acc: {:.4f}, val_pt_acc: {:.4f}\n'
              .format(val_total_loss, val_p1_loss, val_p2_loss, val_pt_loss, val_w_diff_loss, val_p1_acc,
                      val_p2_acc, val_pt_acc, ))

        val_statistic['total_loss'].append(val_total_loss)
        val_statistic['p1_loss'].append(val_p1_loss)
        val_statistic['p2_loss'].append(val_p2_loss)
        val_statistic['pt_loss'].append(val_pt_loss)
        val_statistic['w_diff_loss'].append(val_w_diff_loss)
        val_statistic['p1_acc'].append(val_p1_acc)
        val_statistic['p2_acc'].append(val_p2_acc)
        val_statistic['pt_acc'].append(val_pt_acc)
        val_statistic['val_f1'].append(current_val_f1)


    plot_statistic(train_statistic, val_statistic, id)

    print('\n==============Final Evaluation...==================')

    print('Evaluate target....')

    test(x_test, y_test, model, domain='target_test')
    test(x_target, y_target, model, domain='target_train')
    test(x_train, y_train, model, domain='source_train')
    X_test = torch.cat((x_target, x_test), dim=0)
    Y_test = torch.cat((y_target, y_test), dim=0)
    test(X_test, Y_test, model, domain='target_full')


    if args.TSNE:
        print('=== TSNE plot =====')
        vis(args.id)

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))

    with open('exp1/' + 'run_' + str(args.id) + '_results.txt', 'a') as f:
        f.write('current_time: ' + strftime("%Y-%m-%d %H:%M:%S", localtime()) + '\n\n')
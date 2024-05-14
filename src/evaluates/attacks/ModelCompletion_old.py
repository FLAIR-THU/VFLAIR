import sys, os

sys.path.append(os.pardir)

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
import numpy as np
import copy
import pickle
import matplotlib.pyplot as plt
import itertools

from evaluates.attacks.attacker import Attacker
from models.global_models import *
from utils.basic_functions import cross_entropy_for_onehot, append_exp_res
from utils.pmc_functions import *  # precision_recall,interleave_offsets,interleave, SemiLoss,WeightEMA,AverageMeter,InferenceHead,accuracy
from dataset.party_dataset import ActiveDataset

import torch.nn.init as init


def label_to_one_hot(target, num_classes=10):
    # print('label_to_one_hot:', target, type(target))
    try:
        _ = target.size()[1]
        # print("use target itself", target.size())
        onehot_target = target.type(torch.float32)
    except:
        target = torch.unsqueeze(target, 1)
        # print("use unsqueezed target", target.size())
        onehot_target = torch.zeros(target.size(0), num_classes)
        onehot_target.scatter_(1, target, 1)
    return onehot_target


class ModelCompletion(Attacker):
    def __init__(self, top_vfl, args):
        super().__init__(args)
        self.args = args
        # get information for launching BLI attack
        self.vfl_info = top_vfl.final_state
        # prepare parameters
        self.device = args.device
        self.num_classes = args.num_classes
        self.k = args.k  # party number
        self.party = args.attack_configs['party']  # parties that launch attacks

        self.lr = args.attack_configs['lr']
        self.epochs = args.attack_configs['epochs']
        self.label_size = args.num_classes
        self.batch_size = args.attack_configs['batch_size']
        self.val_iteration = args.attack_configs['val_iteration']
        self.n_labeled_per_class = args.attack_configs['n_labeled_per_class']
        # self.hidden_size = args.attack_configs['hidden_size']

        self.dummy_active_top_trainable_model = None
        self.optimizer_trainable = None  # construct later
        self.dummy_active_top_non_trainable_model = None
        self.optimizer_non_trainable = None  # construct later
        self.criterion = cross_entropy_for_onehot

        # self.file_name = 'attack_result.txt'
        # self.exp_res_dir = f'exp_result/main/{args.dataset}/attack/PMC/'
        # self.exp_res_path = ''

    def set_seed(self, seed=0):
        # random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    def calc_label_recovery_rate(self, dummy_label, gt_label):
        success = torch.sum(torch.argmax(dummy_label, dim=-1) == torch.argmax(gt_label, dim=-1)).item()
        total = dummy_label.shape[0]
        return success / total

    def train(self, labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch,
              num_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        losses_x = AverageMeter()
        losses_u = AverageMeter()
        ws = AverageMeter()
        end = time.time()

        labeled_train_iter = iter(labeled_trainloader)
        unlabeled_train_iter = iter(unlabeled_trainloader)

        model.train()

        val_iteration = self.val_iteration  # len(labeled_trainloader)
        for batch_idx in range(val_iteration):  # args.val_iteration
            try:
                inputs_x, targets_x = next(labeled_train_iter)
                # inputs_x, targets_x = labeled_trainloader.dataset[batch_idx]
            except StopIteration:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_train_iter)
            try:
                inputs_u, _ = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _ = next(unlabeled_train_iter)

            # measure data loading time
            data_time.update(time.time() - end)

            inputs_x = inputs_x.type(torch.float)
            inputs_u = inputs_u.type(torch.float)

            batch_size = inputs_x.size(0)

            # Transform label to one-hot

            if len(targets_x.size()) == 1:
                targets_x = targets_x.unsqueeze(1)
            if targets_x.size()[1] == 1:
                targets_x = label_to_one_hot(targets_x, num_classes=num_classes)
            # targets_x = targets_x.view(-1, 1).type(torch.long)
            # targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x, 1)

            inputs_x, targets_x = inputs_x.to(self.device), targets_x.to(self.device)  # .cuda(non_blocking=True)
            inputs_u = inputs_u.to(self.device)

            with torch.no_grad():
                targets_x.view(-1, 1).type(torch.long)  # compute guessed labels of unlabel samples
                outputs_u = model(inputs_u)
                p = torch.softmax(outputs_u, dim=1)
                pt = p ** (1 / 0.8)
                targets_u = pt / pt.sum(dim=1, keepdim=True)
                targets_u = targets_u.detach()

            # mixup
            all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
            all_targets = torch.cat([targets_x, targets_u], dim=0)

            l = np.random.beta(0.75, 0.75)  # alpha = 0.75

            l = max(l, 1 - l)

            idx = torch.randperm(all_inputs.size(0))

            input_a, input_b = all_inputs, all_inputs[idx]
            target_a, target_b = all_targets, all_targets[idx]

            mixed_input = l * input_a + (1 - l) * input_b
            mixed_target = l * target_a + (1 - l) * target_b

            # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
            mixed_input = list(torch.split(mixed_input, batch_size))
            mixed_input = interleave(mixed_input, batch_size)

            logits = [model(mixed_input[0])]
            for input in mixed_input[1:]:
                logits.append(model(input))

            # put interleaved samples back
            logits = interleave(logits, batch_size)
            logits_x = logits[0]
            logits_u = torch.cat(logits[1:], dim=0)
            # (self,outputs_x, targets_x, outputs_u, targets_u, epoch, all_epochs):
            x_length = logits_x.size()[0]  # x_length batch_size
            Lx, Lu, w = criterion(logits_x, mixed_target[:x_length], logits_u, mixed_target[x_length:],
                                  epoch + batch_idx / val_iteration, self.epochs)  #
            loss = Lx + w * Lu

            # record loss
            losses.update(loss.item(), inputs_x.size(0))
            losses_x.update(Lx.item(), inputs_x.size(0))
            losses_u.update(Lu.item(), inputs_x.size(0))
            ws.update(w, inputs_x.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema_optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print('one batch training done')
            if batch_idx % 250 == 0:
                print("batch_idx:", batch_idx, " loss:", losses.avg)
        return losses.avg, losses_x.avg, losses_u.avg

    def validate(self, valloader, model, criterion, epoch, mode, num_classes):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        topk = AverageMeter()
        precision = AverageMeter()
        recall = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(valloader):
                # in vertical federated learning scenario, attacker(party A) only has part of features, i.e. half of the img
                # inputs = clip_function(inputs, args.half)

                # measure data loading time
                data_time.update(time.time() - end)

                inputs, targets = inputs.to(self.device), targets.to(self.device)  # .cuda(non_blocking=True)
                # compute output
                inputs = inputs.type(torch.float)
                outputs = model(inputs).type(torch.float)
                targets = targets.type(torch.float)

                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, preck = accuracy(outputs, targets, topk=(1, 2))  # top k accuracy default k=2
                # if num_classes == 2:
                #     print('outputs,targets:',outputs)
                #     prec, rec = precision_recall(outputs, targets)
                #     precision.update(prec, inputs.size(0))
                #     recall.update(rec, inputs.size(0))

                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                topk.update(preck.item(), inputs.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()
                # print('one batch done')
        print("Dataset Overall Statistics:")
        # if num_classes == 2:
        #     print("  precision", precision.avg, end='')
        #     print("  recall", recall.avg, end='')
        #     if (precision.avg + recall.avg) != 0:
        #         print("  F1", 2 * (precision.avg * recall.avg) / (precision.avg + recall.avg), end='')
        #     else:
        #         print(f"F1:0")
        print("top 1 accuracy:{}, top {} accuracy:{}".format(top1.avg, 2, topk.avg))  # topkacc default k=2
        return losses.avg, top1.avg

    def attack(self):
        self.set_seed(123)
        for ik in self.party:  # attacker party #ik
            index = ik
            batch_size = self.batch_size
            num_classes = self.label_size
            # get full data
            aux_data = self.vfl_info["aux_data"][index]
            aux_label = self.vfl_info["aux_label"][-1]
            train_data = self.vfl_info["train_data"][index]
            train_label = self.vfl_info["train_label"][-1]
            test_data = self.vfl_info["test_data"][index]
            test_label = self.vfl_info["test_label"][-1]  # onli active party have data

            n_labeled_per_class = self.n_labeled_per_class

            train_label = label_to_one_hot(train_label, self.num_classes)
            aux_label = label_to_one_hot(aux_label, self.num_classes)
            test_label = label_to_one_hot(test_label, self.num_classes)

            print('all_train_data:', train_data.size())
            print('all_train_label:', train_label.size())
            # labels = np.array((torch.argmax(aux_label, dim=-1)))
            # train_labeled_idxs = []
            # train_unlabeled_idxs = []
            # for i in range(self.num_classes):
            #     idxs = np.where(labels == i)[0]
            #     np.random.shuffle(idxs)
            #     train_labeled_idxs.extend(idxs[:n_labeled_per_class])
            # np.random.shuffle(train_labeled_idxs)

            # aux_data = aux_data[train_labeled_idxs]
            # aux_label = aux_label[train_labeled_idxs]

            if len(train_label.size()) == 2:
                labels = np.array((torch.argmax(train_label.cpu(), dim=-1)))
            else:
                labels = np.array(train_label.cpu())

            train_labeled_idxs = []
            train_unlabeled_idxs = []
            for i in range(num_classes):
                idxs = np.where(labels == i)[0]
                np.random.shuffle(idxs)
                train_labeled_idxs.extend(idxs[:n_labeled_per_class])
                train_unlabeled_idxs.extend(idxs[n_labeled_per_class:])
            np.random.shuffle(train_labeled_idxs)
            np.random.shuffle(train_unlabeled_idxs)

            aux_data = train_data[train_labeled_idxs]
            aux_label = train_label[train_labeled_idxs]

            train_data = train_data[train_unlabeled_idxs]
            train_label = train_label[train_unlabeled_idxs]

            print('train_unlabeled_idxs:', len(train_unlabeled_idxs))
            # while min(get_times) < n_label_per_class and idx<len(aux_label):
            #     current_label = int(torch.argmax(aux_label[idx], dim=-1))

            #     if get_times[current_label]<n_label_per_class:
            #         aux_indx.append(idx)
            #         get_times[current_label] = get_times[current_label] +1
            #     idx = idx +1

            # aux_data = aux_data[aux_indx]
            # aux_label = aux_label[aux_indx]

            # print(get_times)
            # print(torch.argmax(aux_label, dim=-1))
            # print(aux_data.size())

            aux_dst = ActiveDataset(aux_data, aux_label)
            aux_loader = DataLoader(aux_dst, batch_size=batch_size, shuffle=True)

            print('train_data:', train_data.size())
            print('batch_size:', batch_size)
            train_dst = ActiveDataset(train_data, train_label)
            train_loader = DataLoader(train_dst, batch_size=batch_size, shuffle=True)

            test_dst = ActiveDataset(test_data, test_label)
            test_loader = DataLoader(test_dst, batch_size=10 * batch_size, shuffle=True)

            # complete_train_data = torch.cat([aux_data,train_data],dim=0)
            # complete_train_label = torch.cat([aux_label,train_label],dim=0)
            # complete_train_dst = ActiveDataset(complete_train_data, complete_train_label)
            # complete_train_loader = DataLoader(complete_train_dst, batch_size=batch_size)

            bottom_model = self.vfl_info['model'][index].to(self.device)  # local bottom model for attacker
            bottom_model.eval()

            def create_model(bottom_model, ema=False, size_bottom_out=10, num_classes=10):
                model = BottomModelPlus(bottom_model, size_bottom_out, num_classes,
                                        num_layer=1,
                                        activation_func_type='ReLU',
                                        use_bn=True)
                model = model

                if ema:
                    for param in model.parameters():
                        param.detach_()

                return model

            model = create_model(copy.deepcopy(bottom_model), ema=False, size_bottom_out=num_classes,
                                 num_classes=num_classes)
            # fix parameters
            ema_model = create_model(copy.deepcopy(bottom_model), ema=True, size_bottom_out=num_classes,
                                     num_classes=num_classes)

            model = model.to(self.device)  # dummy top model
            ema_model = ema_model.to(self.device)

            # Optimizers & Criterion
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            ema_optimizer = WeightEMA(model, ema_model, lr=self.lr, alpha=0.999)  # ema_decay = 0.999
            train_criterion = SemiLoss()
            criterion = nn.CrossEntropyLoss()

            # === Begin Attack ===
            print(f"MC Attack, self.device={self.device}")
            start_time = time.time()
            test_accs = []

            best_acc = 0
            print("---Label inference on complete training dataset:")
            # Attack iterations
            for epoch in range(self.epochs):  # self.epochs: cifar-5, BC_IDC-1, liver-5
                print('\nEpoch: [%d | %d]' % (epoch + 1, self.epochs))

                train_loss, train_loss_x, train_loss_u = self.train(aux_loader, train_loader, model, optimizer,
                                                                    ema_optimizer, \
                                                                    train_criterion, epoch, num_classes)

                print("---MC: Label inference on test dataset:")
                _, test_acc = self.validate(test_loader, ema_model, criterion, epoch, mode='Test Stats',
                                            num_classes=num_classes)
                best_acc = max(test_acc, best_acc)

            print(f"MC, if self.args.apply_defense={self.args.apply_defense}")
            print('MC Best top 1 accuracy:', best_acc)
            # print('PMC Best top 1 accuracy:',p_best_acc)

            # print(f'batch_size=%d,class_num=%d,party_index=%d,recovery_rate=%lf' % (batch_size, self.label_size, index, best_acc))

        print("returning from PMC/AMC")
        return best_acc
        # return recovery_history

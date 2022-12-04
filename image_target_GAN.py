import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
from sklearn.metrics import confusion_matrix
import random, pdb, math, copy
import loss
import torch.nn.functional as F
from torch_geometric.nn import GraphSAGE


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_tar = open(args.t_dset_path).readlines()[:128]
    txt_test = open(args.test_dset_path).readlines()[:128]

    if not args.da == 'uda':
        label_map_s = {}
        for i in range(len(args.src_classes)):
            label_map_s[args.src_classes[i]] = i

        new_tar = []
        for i in range(len(txt_tar)):
            rec = txt_tar[i]
            reci = rec.strip().split(' ')
            if int(reci[1]) in args.tar_classes:
                if int(reci[1]) in args.src_classes:
                    line = reci[0] + ' ' + str(label_map_s[int(reci[1])]) + '\n'
                    new_tar.append(line)
                else:
                    line = reci[0] + ' ' + str(len(label_map_s)) + '\n'
                    new_tar.append(line)
        txt_tar = new_tar.copy()
        txt_test = txt_tar.copy()

    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"], batch_size=train_bs, shuffle=True, num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"], batch_size=train_bs * 3, shuffle=False, num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, netF, netB, netC, flag=False):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            inputs = inputs.cuda()
            outputs = netC(netB(netF(inputs)))
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    mean_ent = torch.mean(loss.Entropy(nn.Softmax(dim=1)(all_output))).cpu().data.item()

    if flag:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        return aacc, acc
    else:
        return accuracy * 100, mean_ent


def test_target(args, zz=''):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bootleneck(type=args.classifier, feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer, class_num=args.class_num, bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F_' + str(zz) + '.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B_' + str(zz) + '.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C_' + str(zz) + '.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    netF.eval()
    netB.eval()
    netC.eval()

    acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, netC, args.dset == "visda17")
    log_str = '\nZz: {}, Task: {}, Accuracy = {:.2f}%'.format(zz, args.name, acc) + '\n' + str(acc_list)
    args.out_file.write(log_str + '\n')
    args.out_file.flush()
    print(log_str + '\n')


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


def train_target_bait(args, zz=''):
    dset_loaders = data_load(args)
    ## set base network
    if args.net[0:3] == 'res':
        netF = network.ResBase(res_name=args.net).cuda()

    netB = network.feat_bottleneck(type=args.classifier,
                                   feature_dim=netF.in_features,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()
    # oldC = network.feat_classifier(type=args.layer,
    #                                class_num=args.class_num,
    #                                bottleneck_dim=args.bottleneck).cuda()

    args.modelpath = args.output_dir_src + '/source_F' + str(zz) + '.pt'
    netF.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_B' + str(zz) + '.pt'
    netB.load_state_dict(torch.load(args.modelpath))
    args.modelpath = args.output_dir_src + '/source_C' + str(zz) + '.pt'
    netC.load_state_dict(torch.load(args.modelpath))
    # oldC.load_state_dict(torch.load(args.modelpath))
    # oldC.eval()
    netC.train()
    # for k, v in oldC.named_parameters():
    #     v.requires_grad = False

    param_group = []
    param_group_b = []
    for k, v in netF.named_parameters():
        param_group += [{'params': v, 'lr': args.lr * args.lr_decay1}]  # 0.1
    for k, v in netB.named_parameters():
        param_group_b += [{'params': v, 'lr': args.lr * args.lr_decay2}]  # 1
    for k, v in netC.named_parameters():
        param_group_b += [{'params': v, 'lr': args.lr * args.lr_decay2}]  # 1
    optimizer = optim.SGD(param_group,
                          momentum=0.9,
                          weight_decay=5e-4,
                          nesterov=True)
    optimizer_c = optim.SGD(param_group_b,
                            momentum=0.9,
                            weight_decay=5e-4,
                            nesterov=True)
    optimizer = op_copy(optimizer)
    optimizer_c = op_copy(optimizer_c)

    netF.train()
    netB.train()

    iter_num = 0
    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval

    while iter_num < max_iter:
        try:
            inputs_test, _, tar_idx = iter_target.next()
        except:
            iter_target = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_target.next()

        if inputs_test.size(0) == 1:
            continue


        if iter_num % interval_iter == 0 and args.cls_par > 0:
            netF.eval()
            netB.eval()
            node, edge, fea_lookup = obtain_vec(dset_loaders['test'], netF, netB, netC, args)
            netF.train()
            netB.train()

        inputs_test = inputs_test.cuda()
        batch_size = inputs_test.shape[0]
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        iter_num += 1

        features_test = netB(netF(inputs_test))
        # outputs_test = netC(features_test)

        if True:
            print("maximize netB and graph node embedding...")
            total_loss = 0
            features_test = netB(netF(inputs_test))
            features_test_gan = fea_lookup[tar_idx].to(torch.device("cuda:0"))

            outputs_test = netC(features_test)
            outputs_test_gan = netC(features_test_gan)
            # outputs_test_old = netC(features_test)

            softmax_out = nn.Softmax(dim=1)(outputs_test)
            softmax_out_old = nn.Softmax(dim=1)(outputs_test_gan)

            loss_cast = loss.SKL(softmax_out, softmax_out_old).sum(dim=1)

            entropy_old = Entropy(softmax_out_old)
            entropy = Entropy(softmax_out)

            indx = entropy_old.topk(int(batch_size * 0.2), largest=True)[-1]
            ones_mask = torch.ones(batch_size).cuda() * -1
            ones_mask[indx] = 1
            loss_cast = loss_cast * ones_mask
            total_loss -= torch.mean(loss_cast) * 10
            total_loss += entropy_old + entropy

            optimizer_c.zero_grad()
            total_loss.backward()
            optimizer_c.step()

        for _ in range(2):
            print("minimize generator network")
            total_loss = 0
            features_test = netB(netF(inputs_test))
            outputs_test = netC(features_test)
            softmax_out = nn.Softmax(dim=1)(outputs_test)

            features_test_gan = fea_lookup[tar_idx].to(torch.device("cuda:0"))
            outputs_test_gan = netC(features_test_gan)
            softmax_out_old = nn.Softmax(dim=1)(outputs_test_gan)

            loss_dis = torch.mean(torch.abs(features_test - features_test_gan))
            # entropy_old = Entropy(softmax_out_old)
            entropy = Entropy(softmax_out)
            total_loss += entropy
            total_loss += loss_dis

            # features_test = netB(netF(inputs_test))
            # outputs_test = netC(features_test)
            #
            # softmax_out = nn.Softmax(dim=1)(outputs_test)
            #
            # outputs_test_old = netC(features_test)
            # softmax_out_old = nn.Softmax(dim=1)(outputs_test_old)
            #
            # msoftmax = softmax_out_old.mean(dim=0)
            # cb_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            # total_loss += cb_loss
            #
            # msoftmax = softmax_out.mean(dim=0)
            # cb_loss = torch.sum(msoftmax * torch.log(msoftmax + 1e-5))
            # total_loss += cb_loss
            #
            # loss_bite = (-softmax_out_old * torch.log(
            #     softmax_out + 1e-5)).sum(1) - (softmax_out * torch.log(
            #     softmax_out_old + 1e-5)).sum(1)
            # total_loss += torch.mean(loss_bite)  # *0.8

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        if iter_num % int(args.interval * len(dset_loaders["target"])) == 0:
            netF.eval()
            netB.eval()
            netC.eval()

            # acc, acc_list = cal_acc(dset_loaders['test'], netF, netB, oldC,
            #                         args.dset == "visda17")
            # log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, \
            #                                                             args.max_epoch * len(dset_loaders["target"]),
            #                                                             acc) + '\n' + str(acc_list)
            acc_s_te, _ = cal_acc(dset_loaders['test'], netF, netB, netC, False)
            log_str = 'Task: {}, Iter:{}/{}; Accuracy = {:.2f}%'.format(args.name, iter_num, max_iter, acc_s_te)

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')

            netF.train()
            netB.train()
            netC.train()
    return netF, netB, netC


def obtain_vec(loader, netF, netB, netC, args):
    print("generating node embedding")
    num_sample = len(loader.dataset)
    # fea_bank = torch.randn(num_sample, 256)
    # score_bank = torch.randn(num_sample, 12).cuda()
    # score_bank = torch.randn(num_sample, 30)   # change class num
    # office: 30; office-home: 65; visda-2017: 12
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for _ in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            labels = data[1]
            # indx = data[2]
            inputs = inputs.cuda()
            # inputs = inputs
            feas = netB(netF(inputs))
            outputs = netC(feas)

            # feature (node) and score (edge) bank update
            # output_norm = F.normalize(feas)  # might remove
            # fea_bank[indx] = output_norm.detach().clone().cpu()
            # score_bank[indx] = outputs.detach().clone()

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    # num_samples
    # all_fea -> data.x
    # graph -> data.edge_index

    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()
    edge = all_fea @ all_fea.T  # edge graph
    threshold = 0.8

    edge_index = []
    node_i = []
    node_j = []

    for i in range(num_sample):
        for j in range(num_sample):
            if edge[i, j] >= threshold:
                node_i.append(i)
                node_j.append(j)

    edge_index.append(node_i)
    edge_index.append(node_j)

    model = GraphSAGE(all_fea.size()[1], 256, 10).to(torch.device("cuda:0"))
    fea_lookup = model(all_fea.to(torch.device("cuda:0")), torch.LongTensor(edge_index).to(torch.device("cuda:0")))

    # TODO:
    #  1. Add edge weight
    #  2. Try cache
    #  3. Try different value of layers
    #  4. Refactor

    return all_fea, edge_index, fea_lookup.detach()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BAIT on VisDA')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch', type=int, default=15, help="max iterations")
    parser.add_argument('--interval', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=64, help="batch_size")
    parser.add_argument('--worker', type=int, default=0, help="number of workers")
    parser.add_argument('--dset', type=str, default='office-home',
                        choices=['VISDA-C', 'office', 'office-home', 'office-caltech'])
    parser.add_argument('--lr', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--net', type=str, default='resnet101', help="resnet50, resnet101")
    parser.add_argument('--seed', type=int, default=2020, help="random seed")
    parser.add_argument('--cls_par', type=float, default=0.3)
    parser.add_argument('--lr_decay1', type=float, default=0.1)
    parser.add_argument('--lr_decay2', type=float, default=1.0)

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--smooth', type=float, default=0.1)
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--trte', type=str, default='val', choices=['full', 'val'])
    parser.add_argument('--zz', type=str, default='val', choices=['5', '10', '15', '20', '25', '30', 'val'])
    parser.add_argument('--savename', type=str, default='bait')

    parser.add_argument('--layer', type=str, default="wn", choices=["linear", "wn"])
    parser.add_argument('--classifier', type=str, default="bn", choices=["ori", "bn"])
    parser.add_argument('--distance', type=str, default='cosine', choices=["euclidean", "cosine"])
    parser.add_argument('--output', type=str, default='san')
    parser.add_argument('--output_src', type=str, default='san')
    parser.add_argument('--issave', type=bool, default=True)
    args = parser.parse_args()

    args.interval = args.max_epoch / 10

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'Real_World']
        args.class_num = 65
    if args.dset == 'office':
        names = ['amazon', 'dslr', 'webcam']
        args.class_num = 31

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    # torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = './data/'
        args.s_dset_path = folder + args.dset + '/' + names[args.s] + '_list.txt'
        args.t_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'
        args.test_dset_path = folder + args.dset + '/' + names[args.t] + '_list.txt'

        if args.dset == 'office-home':
            if args.da == 'pda':
                args.class_num = 65
                args.src_classes = [i for i in range(65)]
                args.tar_classes = [i for i in range(25)]

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset, names[args.s][0].upper())
        args.output_dir = osp.join(args.output, args.da, args.dset, names[args.s][0].upper() + names[args.t][0].upper())
        args.name = names[args.s][0].upper() + names[args.t][0].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.savename = 'par_' + str(args.cls_par)
        if args.da == 'pda':
            args.gent = ''
            args.savename = 'par_' + str(args.cls_par) + '_thr' + str(args.threshold)
        args.out_file = open(osp.join(args.output_dir, 'log_' + args.savename + '.txt'), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        print(print_args(args))
        train_target_bait(args)

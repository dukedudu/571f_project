import argparse
import os, sys
import os.path as osp
import torchvision
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import network, loss
from torch.utils.data import DataLoader
from data_list import ImageList, ImageList_idx
import random, pdb, math, copy
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F
import tqdm

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter)**(-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def image_train(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224, alexnet=False):
    if not alexnet:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    else:
        normalize = Normalize(meanfile='./ilsvrc_2012_mean.npy')
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(), normalize
    ])


def data_load(args):
    ## prepare data
    dsets = {}
    dset_loaders = {}
    train_bs = args.batch_size
    txt_src = open(args.s_dset_path).readlines()
    txt_tar = open(args.t_dset_path).readlines()
    txt_test = open(args.test_dset_path).readlines()

    dsize = len(txt_src)
    tr_size = int(0.9 * dsize)
    # print(dsize, tr_size, dsize - tr_size)
    _, te_txt = torch.utils.data.random_split(txt_src,
                                              [tr_size, dsize - tr_size])
    tr_txt = txt_src

    dsets["source_tr"] = ImageList(tr_txt, transform=image_train())
    dset_loaders["source_tr"] = DataLoader(dsets["source_tr"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["source_te"] = ImageList(te_txt, transform=image_test())
    dset_loaders["source_te"] = DataLoader(dsets["source_te"],
                                           batch_size=train_bs,
                                           shuffle=True,
                                           num_workers=args.worker,
                                           drop_last=False)
    dsets["target"] = ImageList_idx(txt_tar, transform=image_train())
    dset_loaders["target"] = DataLoader(dsets["target"],
                                        batch_size=train_bs,
                                        shuffle=True,
                                        num_workers=args.worker,
                                        drop_last=False)
    dsets["test"] = ImageList_idx(txt_test, transform=image_test())
    dset_loaders["test"] = DataLoader(dsets["test"],
                                      batch_size=train_bs * 3,
                                      shuffle=False,
                                      num_workers=args.worker,
                                      drop_last=False)

    return dset_loaders


def cal_acc(loader, fea_bank, socre_bank, netF, netB, netC,args, flag=False):
    start_test = True
    num_sample = len(loader.dataset)
    label_bank = torch.randn(num_sample)  #.cuda()
    pred_bank = torch.randn(num_sample)
    nu=[]
    s=[]
    var_all=[]

    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = iter_test.next()
            inputs = data[0]
            labels = data[1]
            indx = data[-1]
            inputs = inputs.cuda()
            fea = netF(inputs)
            fea = netB(fea)
            if args.var:
                var_batch=fea.var()
                var_all.append(var_batch)

            #if args.singular:
            _, ss, _ = torch.svd(fea)
            s10=ss[:10]/ss[0]
            s.append(s10)

            outputs = netC(fea)
            softmax_out=nn.Softmax()(outputs)
            nu.append(torch.mean(torch.svd(softmax_out)[1]))
            label_bank[indx] = labels.float().detach().clone()  #.cpu()
            pred_bank[indx] = outputs.max(-1)[1].float().detach().clone().cpu()
            if start_test:
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(
        torch.squeeze(predict).float() == all_label).item() / float(
            all_label.size()[0])

    _, socre_bank_ = torch.max(socre_bank, 1)

    import time
    
    splits = 4
    split_idx = num_sample//splits
    idx_near = torch.zeros(num_sample, 4, dtype=torch.long) 
    tic = time.time()
    for i in range(splits):
        distance = fea_bank[split_idx*i:min(len(fea_bank), split_idx*(i+1))].cpu() @ fea_bank.cpu().T
        # print(distance.shape)
        _, idx_near_split = torch.topk(distance, dim=-1, largest=True, k=4)
        distance = []
        idx_near[split_idx*i:min(len(fea_bank), split_idx*(i+1))] = idx_near_split
    print('Time consumed:', time.time() - tic)

    score_near = socre_bank_[idx_near[:, :]].float().cpu()  #N x 4

    '''acc1 = (score_near.mean(
        dim=-1) == score_near[:, 0]).sum().float() / score_near.shape[0]'''
    acc1 = ((score_near.mean(dim=-1) == score_near[:, 0]) &
            (score_near[:, 0] == pred_bank)).sum().float() / score_near.shape[0]
    acc2 = (
        (score_near.mean(dim=-1) == score_near[:, 0]) &
        (score_near[:, 0] == label_bank)).sum().float() / score_near.shape[0]

    if True:
        matrix = confusion_matrix(all_label, torch.squeeze(predict).float())
        acc = matrix.diagonal() / matrix.sum(axis=1) * 100
        aacc = acc.mean()
        aa = [str(np.round(i, 2)) for i in acc]
        acc = ' '.join(aa)
        if True:
            return aacc, acc, acc1, acc2
        
    else:
        return accuracy * 100, mean_ent


def hyper_decay(x,beta=-2,alpha=1):
    weight=(1 + 10 * x)**(-beta) * alpha
    return weight


def SKL(out1, out2):
    out2_t = out2.clone()
    out2_t = out2_t.detach()
    out1_t = out1.clone()
    out1_t = out1_t.detach()
    return (F.kl_div(F.log_softmax(out1), out2_t, reduction='none') +
            F.kl_div(F.log_softmax(out2), out1_t, reduction='none')) / 2

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=-1)
    return entropy 

def get_entropy_loss(p_softmax):
    mask = p_softmax.ge(0.000001)
    mask_out = torch.masked_select(p_softmax, mask)
    entropy = 0
    for i in range(p_softmax.shape[0]):
        entropy += -(torch.sum(mask_out[i] * torch.log(mask_out[i])))
    return entropy / float(p_softmax.size(0))   

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = nn.Softmax(dim=1)(inputs)
        bce_loss = F.binary_cross_entropy(inputs.squeeze(),  targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def train_target(args):
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.CrossEntropyLoss()
    dset_loaders = data_load(args)
    ## set base network
    netF = network.ResBase(res_name='resnet101').cuda()

    netB = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=2048,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    netF_source = network.ResBase(res_name='resnet101').cuda()

    netB_source = network.feat_bootleneck(type=args.classifier,
                                   feature_dim=2048,
                                   bottleneck_dim=args.bottleneck).cuda()
    netC_source = network.feat_classifier(type=args.layer,
                                   class_num=args.class_num,
                                   bottleneck_dim=args.bottleneck).cuda()

    modelpath = args.output_dir_src + '/source_F.pt'
    netF.load_state_dict(torch.load(modelpath))
    netF_source.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_B.pt'
    netB.load_state_dict(torch.load(modelpath))
    netB_source.load_state_dict(torch.load(modelpath))
    modelpath = args.output_dir_src + '/source_C.pt'
    netC.load_state_dict(torch.load(modelpath))
    netC_source.load_state_dict(torch.load(modelpath))

    param_group = []
    param_group_c = []
    for k, v in netF.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}] #0.1

    for k, v in netB.named_parameters():
        if True:
            param_group += [{'params': v, 'lr': args.lr * 0.1}] # 1
    for k, v in netC.named_parameters():
        param_group_c += [{'params': v, 'lr': args.lr * 0.1}] #1

    optimizer = optim.SGD(param_group)
    optimizer = op_copy(optimizer)

    optimizer_c = optim.SGD(param_group_c)
    optimizer_c = op_copy(optimizer_c)

    #building feature bank and score bank
    loader = dset_loaders["target"]
    num_sample = len(loader.dataset)
    fea_bank = torch.randn(num_sample, args.fea_bank_dim) # can ca
    score_bank = torch.randn(num_sample, args.class_num).cuda()

    netF.eval()
    netB.eval()
    netC.eval()

    #initialize
    print("Initialize...")
    with torch.no_grad():
        iter_test = iter(loader)
        for i in tqdm.tqdm(range(len(loader))):
        # for i in range(10):
            data = iter_test.next()
            # print('what is in data:', data[0], data[-1])
            
            inputs = data[0]
            indx = data[-1]
            #labels = data[1]
            inputs = inputs.cuda()
            feature = netF(inputs)
            fea_norm = F.normalize(feature)
            logits = netB(feature)
            logits = netC(logits)
            outputs = nn.Softmax(-1)(logits)
            fea_bank[indx] = fea_norm.detach().clone().cpu()
            score_bank[indx] = outputs.detach().clone()  #.cpu()

    max_iter = args.max_epoch * len(dset_loaders["target"])
    interval_iter = max_iter // args.interval
    iter_num = 0

    netF.train()
    netB.train()
    netC.train()
    netF_source.eval()
    netB_source.eval()
    netC_source.eval()
    acc_log = 0
    
    real_max_iter = max_iter

    print("Start training...")

    for iter_num in tqdm.tqdm(range(real_max_iter)):
        try:
            inputs_test, _, tar_idx = iter_test.next()
        except:
            iter_test = iter(dset_loaders["target"])
            inputs_test, _, tar_idx = iter_test.next()

        if inputs_test.size(0) == 1:
            continue

        inputs_test = inputs_test.cuda()
        if True:
            alpha = (1 + 10 * iter_num / max_iter)**(-args.beta) * args.alpha
        else:
            alpha = args.alpha

        iter_num += 1
        lr_scheduler(optimizer, iter_num=iter_num, max_iter=max_iter)
        lr_scheduler(optimizer_c, iter_num=iter_num, max_iter=max_iter)

        features_test = netF(inputs_test)
        output_f_norm = F.normalize(features_test)

        logits_test = netB(features_test)
        logits_test = netC(logits_test)
        softmax_out = nn.Softmax(dim=1)(logits_test)

        with torch.no_grad():
            output_f_ = output_f_norm.cpu().detach().clone()
            fea_bank[tar_idx] = output_f_.detach().clone().cpu()
            score_bank[tar_idx] = softmax_out.detach().clone()
            # logit_bank[tar_idx] = logits_test_norm.detach().clone()

        distance = output_f_ @ fea_bank.T # batch*num_sample

        _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k= args.K + 1)
        idx_near = idx_near[:, 1:]  #batch x K

        score_near = score_bank[idx_near]  #batch x K x C

        score_near_mean = torch.mean(score_near)

        feature = torch.cat([softmax_out, score_near_mean], dim=0)
        labels = torch.cat([torch.arange(softmax_out.shape[0]).repeat_interleave(1) for i in range(2)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda()
        similarity_matrix = feature@feature.T

        A = torch.ones(labels.shape[0],1,1, dtype=torch.bool)
        mask = torch.block_diag(*A).cuda()

        labels = labels[~mask].view(labels.shape[0], -1)
        
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)

        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / args.temperature

        loss = criterion(logits, labels)

        score_near_mean = torch.mean(score_near, 1)
        soft_score_loss = torch.mean(-torch.sum(score_near_mean *nn.LogSoftmax()(softmax_out), 1, keepdim=True))
        loss += soft_score_loss

        optimizer.zero_grad()
        optimizer_c.zero_grad()
        loss.backward()
        optimizer.step()
        optimizer_c.step()

        if iter_num % interval_iter == 0 or iter_num == max_iter:
            print("Calculate accuracy...")
            netF.eval()
            netB.eval()
            netC.eval()
            if args.dset == 'visda-2017':
                acc, accc, acc1_knn, acc2_knn = cal_acc(
                    dset_loaders['test'],
                    fea_bank,
                    score_bank,
                    netF,
                    netB,
                    netC,
                    args,
                    flag=True)
                log_str = 'Task: {}, Iter:{}/{}, Epoch:{}/{};  Acc on target: {:.2f}, percentage of shared label: {:.2f}, percentage of correct shared label: {:.2f}'.format( #var
                    args.name, iter_num, max_iter, iter_num//len(dset_loaders["target"]), args.max_epoch, acc, acc1_knn * 100,
                    acc2_knn * 100) + '\n' + 'T: ' + accc

            args.out_file.write(log_str + '\n')
            args.out_file.flush()
            print(log_str + '\n')
            netF.train()
            netB.train()
            netC.train()
            '''if acc>acc_log:
                acc_log = acc
                torch.save(
                    netF.state_dict(),
                    osp.join(args.output_dir, "target_F_" + '2021_'+str(args.tag) + ".pt"))
                torch.save(
                    netB.state_dict(),
                    osp.join(args.output_dir,
                                "target_B_" + '2021_' + str(args.tag) + ".pt"))
                torch.save(
                    netC.state_dict(),
                    osp.join(args.output_dir,
                                "target_C_" + '2021_' + str(args.tag) + ".pt"))'''

    return netF, netB, netC


def print_args(args):
    s = "==========================================\n"
    for arg, content in args.__dict__.items():
        s += "{}:{}\n".format(arg, content)
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LPA')
    parser.add_argument('--gpu_id',
                        type=str,
                        nargs='?',
                        default='0',
                        help="device id to run")
    parser.add_argument('--s', type=int, default=0, help="source")
    parser.add_argument('--t', type=int, default=1, help="target")
    parser.add_argument('--max_epoch',
                        type=int,
                        default=10,
                        help="max iterations")
    parser.add_argument('--interval', type=int, default=150)
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help="batch_size")
    parser.add_argument('--worker',
                        type=int,
                        default=0,
                        help="number of workers")
    parser.add_argument('--dset', type=str, default='visda-2017')
    parser.add_argument('--lr', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--fea_bank_dim', type=float, default=2048, help="feature bank dimension")
    parser.add_argument('--net', type=str, default='resnet101')
    parser.add_argument('--seed', type=int, default=2021, help="random seed")

    parser.add_argument('--bottleneck', type=int, default=256)
    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--epsilon', type=float, default=1e-5)
    parser.add_argument('--layer',
                        type=str,
                        default="wn",
                        choices=["linear", "wn"])
    parser.add_argument('--classifier',
                        type=str,
                        default="bn",
                        choices=["ori", "bn"])
    parser.add_argument('--output', type=str, default='weight/target/')
    parser.add_argument('--output_src', type=str, default='weight/source/')
    parser.add_argument('--tag', type=str, default='AAD')
    parser.add_argument('--da', type=str, default='uda')
    parser.add_argument('--issave', type=bool, default=True)
    parser.add_argument('--cc', default=False, action='store_true')
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--alpha_decay', default=True)
    parser.add_argument('--nuclear', default=False, action='store_true')
    parser.add_argument('--var', default=False, action='store_true')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--dataset_folder', type=str, default="C:/Users/wang0918.stu/Desktop/Datasets")
    args = parser.parse_args()

    if args.dset == 'office-home':
        names = ['Art', 'Clipart', 'Product', 'RealWorld']
        args.class_num = 65
    if args.dset == 'visda-2017':
        names = ['train', 'validation']
        args.class_num = 12

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    for i in range(len(names)):
        if i == args.s:
            continue
        args.t = i

        folder = args.dataset_folder
        if 'jiaho' in folder:
            args.s_dset_path = folder + args.dset + '/' + names[
                args.s] + '_list_win.txt'
            args.t_dset_path = folder + args.dset + '/' + names[
                args.t] + '_list_win.txt'
            args.test_dset_path = folder + args.dset + '/' + names[
                args.t] + '_list_win.txt'
        else:
            args.s_dset_path = folder + '/' + args.dset + '/' + names[
                args.s] + '_list.txt'
            args.t_dset_path = folder + '/' + args.dset + '/' + names[
                args.t] + '_list.txt'
            args.test_dset_path = folder + '/' + args.dset + '/' + names[
                args.t] + '_list.txt'

        args.output_dir_src = osp.join(args.output_src, args.da, args.dset,
                                       names[args.s].upper())
        args.output_dir = osp.join(
            args.output, args.da, args.dset,
            names[args.s].upper() + '-' + names[args.t].upper())
        args.name = names[args.s].upper() + '-' + names[args.t].upper()

        if not osp.exists(args.output_dir):
            os.system('mkdir -p ' + args.output_dir)
        if not osp.exists(args.output_dir):
            os.mkdir(args.output_dir)

        args.out_file = open(osp.join(args.output_dir, 'log_concat5knn_{}_K_{}.txt'.format(args.tag, args.K)), 'w')
        args.out_file.write(print_args(args) + '\n')
        args.out_file.flush()
        train_target(args)

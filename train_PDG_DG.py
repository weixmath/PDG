from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics
import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from data.dataloader import *
from model.PDG import best
from model.pointnet import PointNet
from model.dgcnn import DGCNN
import numpy as np
from torch.utils.data import DataLoader
import time
import random
from utils.loss import SupConLoss
from torch.utils.tensorboard import SummaryWriter
class IOStream():

    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()
def _init_():
    if not os.path.exists('checkpoint'):
        os.makedirs('checkpoint')
    if not os.path.exists('checkpoint/' + args.exp_name):
        os.makedirs('checkpoint/' + args.exp_name)
    if not os.path.exists('checkpoint/' + args.exp_name + '/' + 'models'):
        os.makedirs('checkpoint/' + args.exp_name + '/' + 'models')
    # os.system('cp main.py checkpoint' + '/' + args.exp_name + '/' + 'main.py.backup')
    # os.system('cp model.py checkpoint' + '/' + args.exp_name + '/' + 'model.py.backup')
    # os.system('cp util.py checkpoint' + '/' + args.exp_name + '/' + 'util.py.backup')
    # os.system('cp data.py checkpoint' + '/' + args.exp_name + '/' + 'data.py.backup')

def entropy_prob(x):
    entropy = -(torch.softmax(x,-1)*torch.log_softmax(x,-1)).sum(-1)
    return entropy.mean()

def train(args, io):
    print('this')
    writer = SummaryWriter('./checkpoint/' + args.exp_name)
    source_train_loader = DataLoader(myBatchPaddingData_DG(pc_input_num=args.num_points, status='train', aug=True, swapax=swapax,
                                                pc_root='data/data_DG/'+args.src_dataset,sample_num=1,batch_size=1), num_workers=12,
                                    batch_size=32, shuffle=True, drop_last=True, pin_memory=True)
    target_test_loader = DataLoader(PaddingData(pc_input_num=args.num_points, status='test', aug=False, swapax=False,
                                                pc_root='data/data_DG/'+args.trg_dataset), num_workers=4,
                                    batch_size=32, shuffle=True, drop_last=False, pin_memory=True)
    target_test_loader2 = DataLoader(PaddingData(pc_input_num=args.num_points, status='test', aug=False, swapax=False,
                                                pc_root='data/data_DG/' + args.trg_dataset2), num_workers=4,
                                    batch_size=32, shuffle=True, drop_last=False, pin_memory=True)
    device = torch.device("cuda" if args.cuda else "cpu")
    # Try to load models
    # if args.model == 'pointnet':
    #     model = PointNet(args,output_channels=num_class).to(device)
    # elif args.model == 'dgcnn':
    #     model = DGCNN(args,output_channels=num_class).to(device)
    model = best(args, output_channels=num_class).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(opt, args.epochs,1e-5)
    CSL = SupConLoss(temperature=0.07, contrast_mode='all', base_temperature=0.07)
    best_test_acc = 0
    best_test_acc_bg = 0
    last_five_epoch = []
    last_five_epoch_bg = []
    for i in range(args.epochs):
        train_loss = 0.0
        count = 0.0
        train_pred = []
        train_true = []
        model.train()
        for data_s in source_train_loader:
            idx,data, pn, label, data_ori, pn_ori = data_s
            data,pn,data_ori,pn_ori, label = data.to(device).transpose(1, 2),pn.to(device),data_ori.to(
                device).transpose(1, 2), pn_ori.to(device),label.to(device).squeeze()
            batch_size = data.size()[0]
            label_score_L = label.unsqueeze(1).repeat(1, 8).reshape(-1)
            opt.zero_grad()
            out1,out2,score1, score2, p1,f1 = model(data_ori, pn_ori, data, pn,test=False)
            loss_entropy2 = nn.CrossEntropyLoss()(score1,label_score_L)+nn.CrossEntropyLoss()(score2,label_score_L)
            loss_entropy = nn.CrossEntropyLoss()(out1,label)+nn.CrossEntropyLoss()(out2,label)
            loss_CSL = CSL(p1)
            loss_CSL_all = CSL(f1,label)
            loss = loss_entropy+0.05*loss_entropy2+0.01*loss_CSL+0.01*loss_CSL_all
            loss.backward()
            opt.step()
            preds = out1.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_acc = metrics.accuracy_score(train_true, train_pred)
        avg_per_class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
        train_loss = train_loss * 1.0 / count
        outstr ='Source_train epoch %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (i,train_loss, train_acc, avg_per_class_acc)
        io.cprint(outstr)
        scheduler.step()
        if i > -1 :
            loss_t, acc_t, per_class_acc_t = val(model, target_test_loader, device)
            loss_t2, acc_t2, per_class_acc_t2 = val(model, target_test_loader2, device)
            outstr2 = 'Target_test epoch %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (i,loss_t, acc_t, per_class_acc_t)
            io.cprint(outstr2)
            outstr3 = 'Target_test_bg epoch %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (
            i, loss_t2, acc_t2, per_class_acc_t2)
            io.cprint(outstr3)
            # writer.add_scalar('train_acc:',acc,i)
            # writer.add_scalar('test_acc:', acc_bg, i)
            if acc_t >= best_test_acc:
                best_test_acc = acc_t
            if acc_t2 >= best_test_acc_bg:
                best_test_acc_bg = acc_t2
            if i >= 150:
                torch.save(model.state_dict(), 'checkpoint/%s/models/ori_model_%d.t7' % (args.exp_name, i))
            io.cprint('best_target_test_acc::::::::::::::::::::::%5f' % (best_test_acc))
            io.cprint('best_target_bg_test_acc::::::::::::::::::::::%5f' % (best_test_acc_bg))
            last_five_epoch.append(acc_t)
            last_five_epoch_bg.append(acc_t2)
    io.cprint('final 5 epoch acc: %5f' % (np.mean(np.array(last_five_epoch)[-5:-1])))
    io.cprint('final 5 epoch acc bg: %5f' % (np.mean(np.array(last_five_epoch_bg)[-5:-1])))
    writer.close()

def val(model,data_loader,device):
    test_loss = 0.0
    count = 0.0
    model.eval()
    test_pred = []
    test_true = []
    iter_test = iter(data_loader)
    for i in tqdm.trange(len(data_loader)):
        idx,data, pn, label = iter_test.next()
        data, label = data.to(device), label.to(device).squeeze()
        data = data.squeeze()
        batch_size = data.size()[0]
        with torch.no_grad():
            logits,_  = model(data)
        loss = nn.CrossEntropyLoss()(logits, label)
        preds = logits.max(dim=1)[1]
        count += batch_size
        test_loss += loss.item() * batch_size
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    test_loss = test_loss * 1.0 / count

    return test_loss, test_acc, avg_per_class_acc


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='test', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--src_dataset', type=str, default='modelnet_11', metavar='N',
                        choices=['modelnet,shapenet,scannet'])
    parser.add_argument('--trg_dataset', type=str, default='scanobjectnn_11', metavar='N',
                        choices=['modelnet,shapenet,scannet'])
    parser.add_argument('--trg_dataset2', type=str, default='scanobjectnn_bg_11', metavar='N',
                        choices=['modelnet,shapenet,scannet'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=160, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--mixup_params', type=float, default=1.0, help='a,b in beta distribution')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='./checkpoint/pointnet_6part/models/best_ori_model.t7',
                        metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()
    def seed_torch(seed=0):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.enabled = True
    io = IOStream('checkpoint/' + args.exp_name + '/run.log')
    io.cprint(str(args))
    num_class = 11 if args.src_dataset == 'modelnet_11' else 9
    swapax = args.src_dataset == 'modelnet_11'
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    seed_torch(111)
    start_time = time.time()
    train(args, io)
    end_time = time.time()
    print('training time%.6f' % ((end_time - start_time) / 60))

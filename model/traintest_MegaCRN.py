import sys
import os
import shutil
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
# from torchsummary import summary
import argparse
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from utils import StandardScaler, DataLoader, masked_mae_loss, masked_mape_loss, masked_mse_loss, masked_rmse_loss
from MegaCRN import MegaCRN
print(torch.version.cuda)
print('Is GPU available? {}\n'.format(torch.cuda.is_available()))

cros_entropy_loss = nn.MultiLabelSoftMarginLoss()

def print_model(model):
    param_count = 0
    logger.info('Trainable parameter list:')
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape, param.numel())
            param_count += param.numel()
    logger.info(f'In total: {param_count} trainable parameters.')
    return

def get_model():
    model = MegaCRN(num_nodes=args.num_nodes, input_dim=args.input_dim, output_dim=args.output_dim, horizon=args.horizon,
                    rnn_units=args.rnn_units, num_layers=args.num_rnn_layers, mem_num=args.mem_num, mem_dim=args.mem_dim,
                    cheb_k = args.max_diffusion_step, cl_decay_steps=args.cl_decay_steps, use_curriculum_learning=args.use_curriculum_learning).to(device)
    return model

def prepare_x_y(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :return1: x shape (seq_len, batch_size, num_sensor, input_dim)
              y shape (horizon, batch_size, num_sensor, input_dim)
    :return2: x: shape (seq_len, batch_size, num_sensor * input_dim)
              y: shape (horizon, batch_size, num_sensor * output_dim)
    """
    x0 = x[..., :args.input_dim]
    y0 = y[..., :args.output_dim]
    y1 = y[..., args.output_dim:]
    x0 = torch.from_numpy(x0).float()
    y0 = torch.from_numpy(y0).float()
    y1 = torch.from_numpy(y1).float()
    return x0.to(device), y0.to(device), y1.to(device) # x, y, y_cov

def evaluate(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter = data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        maes, mapes, mses = [], [], []
        l_3, m_3, r_3 = [], [], []
        l_6, m_6, r_6 = [], [], []
        l_12, m_12, r_12 = [], [], []
        for x, y in data_iter:
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg = model(x, ycov)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            # y_pred = output
            # y_true = y
            loss1 = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            # loss1 = cros_entropy_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            maes.append(masked_mae_loss(y_pred, y_true).item())
            mapes.append(masked_mape_loss(y_pred, y_true).item())
            mses.append(masked_mse_loss(y_pred, y_true).item())
            # Important for MegaCRN model to let T come first.
            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)
            l_3.append(masked_mae_loss(y_pred[2:3], y_true[2:3]).item())
            m_3.append(masked_mape_loss(y_pred[2:3], y_true[2:3]).item())
            r_3.append(masked_mse_loss(y_pred[2:3], y_true[2:3]).item())
            l_6.append(masked_mae_loss(y_pred[5:6], y_true[5:6]).item())
            m_6.append(masked_mape_loss(y_pred[5:6], y_true[5:6]).item())
            r_6.append(masked_mse_loss(y_pred[5:6], y_true[5:6]).item())
            l_12.append(masked_mae_loss(y_pred[11:12], y_true[11:12]).item())
            m_12.append(masked_mape_loss(y_pred[11:12], y_true[11:12]).item())
            r_12.append(masked_mse_loss(y_pred[11:12], y_true[11:12]).item())
            ys_true.append(y_true)
            ys_pred.append(y_pred)
        mean_loss = np.mean(losses)
        mean_mae, mean_mape, mean_rmse = np.mean(maes), np.mean(mapes), np.sqrt(np.mean(mses))
        l_3, m_3, r_3 = np.mean(l_3), np.mean(m_3), np.sqrt(np.mean(r_3))
        l_6, m_6, r_6 = np.mean(l_6), np.mean(m_6), np.sqrt(np.mean(r_6))
        l_12, m_12, r_12 = np.mean(l_12), np.mean(m_12), np.sqrt(np.mean(r_12))
        if mode == 'test':
            logger.info('Horizon overall: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(mean_mae, mean_mape, mean_rmse))
            logger.info('Horizon 3 day: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_3, m_3, r_3))
            logger.info('Horizon 6 day: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_6, m_6, r_6))
            logger.info('Horizon 12 day: mae: {:.4f}, mape: {:.4f}, rmse: {:.4f}'.format(l_12, m_12, r_12))
        return mean_loss, ys_true, ys_pred

def mzeMaeMetrics(y_pred, y_true):
    Y_test = y_true.transpose(1, 2).reshape((-1, args.num_nodes))
    Y_pred = y_pred.transpose(1, 2).reshape((-1, args.num_nodes))
    if args.num_class == 3:
        Y_pred[(0 < Y_pred) & (Y_pred <= 0.5)] = 0
        Y_pred[(0.5 < Y_pred) & (Y_pred <= 1.5)] = 1
        Y_pred[(1.5 < Y_pred)] = 2
    elif args.num_class == 4:
        Y_pred[(0 < Y_pred) & (Y_pred <= 0.5)] = 0
        Y_pred[(0.5 < Y_pred) & (Y_pred <= 1.5)] = 1
        Y_pred[(1.5 < Y_pred) & (Y_pred <= 2.5)] = 2
        Y_pred[(2.5 < Y_pred)] = 3
    elif args.num_class == 5:
        Y_pred[(0 < Y_pred) & (Y_pred <= 0.5)] = 0
        Y_pred[(0.5 < Y_pred) & (Y_pred <= 1.5)] = 1
        Y_pred[(1.5 < Y_pred) & (Y_pred <= 2.5)] = 2
        Y_pred[(2.5 < Y_pred) & (Y_pred <= 3.5)] = 3
        Y_pred[(3.5 < Y_pred)] = 4
    else:
        Y_pred[(0 < Y_pred) & (Y_pred <= 0.5)] = 0
        Y_pred[(1.5 < Y_pred)] = 1
    mze = ((Y_pred != Y_test).long().sum() / Y_test.ravel().shape[0]).item()
    mae = (abs(Y_pred - Y_test).sum() / Y_test.ravel().shape[0]).item()
    precision = precision_score(Y_test.cpu().detach().numpy().flatten().astype(int),
                                Y_pred.cpu().detach().numpy().flatten().astype(int),
                                average='macro')
    recall = recall_score(Y_test.cpu().detach().numpy().flatten().astype(int),
                          Y_pred.cpu().detach().numpy().flatten().astype(int),
                          average='macro')
    f1 = f1_score(Y_test.cpu().detach().numpy().flatten().astype(int),
                  Y_pred.cpu().detach().numpy().flatten().astype(int),
                  average='macro')
    return mze, mae, precision, recall, f1

def evaluate_scale(model, mode):
    with torch.no_grad():
        model = model.eval()
        data_iter = data[f'{mode}_loader'].get_iterator()
        losses = []
        ys_true, ys_pred = [], []
        mze, mae = [], []
        mze_1, mae_1, precision_1, recal_1, f1_1 = [], [], [], [], []
        mze_3, mae_3, precision_3, recal_3, f1_3  = [], [], [], [], []
        mze_5, mae_5, precision_5, recal_5, f1_5  = [], [], [], [], []
        mze_7, mae_7, precision_7, recal_7, f1_7  = [], [], [], [], []
        for x, y in data_iter:
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg = model(x, ycov)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            # y_pred = output
            # y_true = y
            loss1 = masked_mae_loss(y_pred, y_true)  # masked_mae_loss(y_pred, y_true)
            # loss1 = cros_entropy_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            # Followed the DCRNN TensorFlow Implementation
            mze_iter, mae_iter, precision_iter, recall_iter, f1_iter = mzeMaeMetrics(y_pred, y_true)
            mze.append(mze_iter)
            mae.append(mae_iter)
            # Important for MegaCRN model to let T come first.
            y_true, y_pred = y_true.permute(1, 0, 2, 3), y_pred.permute(1, 0, 2, 3)

            mze_iter1, mae_iter1, precision_iter1, recall_iter1, f1_iter1 = mzeMaeMetrics(y_pred[0:1], y_true[0:1])
            mze_1.append(mze_iter1)
            mae_1.append(mae_iter1)
            precision_1.append(precision_iter1)
            recal_1.append(recall_iter1)
            f1_1.append(f1_iter1)

            mze_iter3, mae_iter3, precision_iter3, recall_iter3, f1_iter3 = mzeMaeMetrics(y_pred[2:3], y_true[2:3])
            mze_3.append(mze_iter3)
            mae_3.append(mae_iter3)
            precision_3.append(precision_iter3)
            recal_3.append(recall_iter3)
            f1_3.append(f1_iter3)

            mze_iter5, mae_iter5, precision_iter5, recall_iter5, f1_iter5  = mzeMaeMetrics(y_pred[6:7], y_true[6:7])
            mze_5.append(mze_iter5)
            mae_5.append(mae_iter5)
            precision_5.append(precision_iter5)
            recal_5.append(recall_iter5)
            f1_5.append(f1_iter5)

            mze_iter7, mae_iter7, precision_iter7, recall_iter7, f1_iter7  = mzeMaeMetrics(y_pred[8:9], y_true[8:9])
            mze_7.append(mze_iter7)
            mae_7.append(mae_iter7)
            precision_7.append(precision_iter7)
            recal_7.append(recall_iter7)
            f1_7.append(f1_iter7)

            ys_true.append(y_true)
            ys_pred.append(y_pred)

        mean_loss = np.mean(losses)
        mean_mze, mean_mae = np.mean(mze), np.mean(mae)
        mze_1, mae_1, precision_1, recal_1, f1_1 = np.mean(mze_1), np.mean(mae_1), np.mean(precision_1), np.mean(recal_1), np.mean(f1_1)
        mze_3, mae_3, precision_3, recal_3, f1_3 = np.mean(mze_3), np.mean(mae_3), np.mean(precision_3), np.mean(recal_3), np.mean(f1_3)
        mze_5, mae_5, precision_5, recal_5, f1_5 = np.mean(mze_5), np.mean(mae_5), np.mean(precision_5), np.mean(recal_5), np.mean(f1_5)
        mze_7, mae_7, precision_7, recal_7, f1_7 = np.mean(mze_7), np.mean(mae_7), np.mean(precision_7), np.mean(recal_7), np.mean(f1_7)
        if mode == 'test':
            logger.info(
                'Horizon overall: mze: {:.4f}, mae: {:.4f}'.format(mean_mze, mean_mae))
            logger.info('Horizon 1 day: mze: {:.4f}, mae: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(mze_1, mae_1, precision_1, recal_1, f1_1))
            logger.info('Horizon 3 day: mze: {:.4f}, mae: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(mze_3, mae_3, precision_3, recal_3, f1_3))
            logger.info('Horizon 5 day: mze: {:.4f}, mae: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(mze_5, mae_5, precision_5, recal_5, f1_5))
            logger.info('Horizon 7 day: mze: {:.4f}, mae: {:.4f}, precision: {:.4f}, recall: {:.4f}, f1: {:.4f}'.format(mze_7, mae_7, precision_7, recal_7, f1_7))
        return mean_loss, ys_true, ys_pred

def traintest_model():
    model = get_model()
    print_model(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, eps=args.epsilon)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.steps, gamma=args.lr_decay_ratio)
    min_val_loss = float('inf')
    wait = 0
    batches_seen = 0
    for epoch_num in range(args.epochs):
        start_time = time.time()
        model = model.train()
        data_iter = data['train_loader'].get_iterator()
        losses = []
        for x, y in data_iter:
            optimizer.zero_grad()
            x, y, ycov = prepare_x_y(x, y)
            output, h_att, query, pos, neg = model(x, ycov, y, batches_seen)
            y_pred = scaler.inverse_transform(output)
            y_true = scaler.inverse_transform(y)
            # y_pred = output
            # y_true = y
            loss1 = masked_mae_loss(y_pred, y_true) # masked_mae_loss(y_pred, y_true)
            # loss1 = cros_entropy_loss(y_pred, y_true)
            separate_loss = nn.TripletMarginLoss(margin=1.0)
            compact_loss = nn.MSELoss()
            loss2 = separate_loss(query, pos.detach(), neg.detach())
            loss3 = compact_loss(query, pos.detach())
            loss = loss1 + args.lamb * loss2 + args.lamb1 * loss3
            losses.append(loss.item())
            batches_seen += 1
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm) # gradient clipping - this does it in place
            optimizer.step()
        train_loss = np.mean(losses)
        lr_scheduler.step()
        val_loss, _, _ = evaluate(model, 'val')
        # if (epoch_num % args.test_every_n_epochs) == args.test_every_n_epochs - 1:
        end_time2 = time.time()
        message = 'Epoch [{}/{}] ({}) train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.1f}s'.format(epoch_num + 1, 
                   args.epochs, batches_seen, train_loss, val_loss, optimizer.param_groups[0]['lr'], (end_time2 - start_time))
        logger.info(message)
        # test_loss, _, _ = evaluate_scale(model, 'test')

        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), modelpt_path)
            # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        elif val_loss >= min_val_loss:
            wait += 1
            if wait == args.patience:
                logger.info('Early stopping at epoch: %d' % epoch_num)
                break
        
        # if test_loss < min_val_loss:
        #     wait = 0
        #     min_val_loss = test_loss
        #     torch.save(model.state_dict(), modelpt_path)
        #     # logger.info('Val loss decrease from {:.4f} to {:.4f}, saving model to pt'.format(min_val_loss, val_loss))
        # elif test_loss >= min_val_loss:
        #     wait += 1
        #     if wait == args.patience:
        #         logger.info('Early stopping at epoch: %d' % epoch_num)
        #         break

    torch.save(model.state_dict(), modelpt_path)
    logger.info('=' * 35 + 'Best model performance' + '=' * 35)
    model = get_model()
    model.load_state_dict(torch.load(modelpt_path))
    # test_loss, _, _ = evaluate(model, 'test')
    test_loss, _, _ = evaluate_scale(model, 'test')

#########################################################################################    
parser = argparse.ArgumentParser()
parser.add_argument('--val_ratio', type=float, default=0.125, help='the ratio of validation data among the trainval ratio')
parser.add_argument('--seq_len', type=int, default=12, help='input sequence length') #12
parser.add_argument('--horizon', type=int, default=12, help='output sequence length') #12
parser.add_argument('--input_dim', type=int, default=1, help='number of input channel')
parser.add_argument('--output_dim', type=int, default=1, help='number of output channel')
parser.add_argument('--max_diffusion_step', type=int, default=3, help='max diffusion step or Cheb K') #3
parser.add_argument('--num_rnn_layers', type=int, default=1, help='number of rnn layers')
parser.add_argument('--rnn_units', type=int, default=64, help='number of rnn units') #64
parser.add_argument('--mem_num', type=int, default=20, help='number of meta-nodes/prototypes') #20
parser.add_argument('--mem_dim', type=int, default=64, help='dimension of meta-nodes/prototypes') #64
parser.add_argument("--loss", type=str, default='mask_mae_loss', help="mask_mae_loss")
parser.add_argument('--lamb', type=float, default= 0.01, help='lamb value for separate loss') #0.01
parser.add_argument('--lamb1', type=float, default= 0.01, help='lamb1 value for compact loss') #0.01
parser.add_argument("--epochs", type=int, default=100, help="number of epochs of training") #200
parser.add_argument("--patience", type=int, default=20, help="patience used for early stop")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches") #64

parser.add_argument("--lr", type=float, default=1e-2, help="base learning rate") #0.01

parser.add_argument("--steps", type=eval, default=[50, 100], help="steps")
parser.add_argument("--lr_decay_ratio", type=float, default=0.1, help="lr_decay_ratio") #0.1
parser.add_argument("--epsilon", type=float, default=1e-3, help="optimizer epsilon")
parser.add_argument("--max_grad_norm", type=int, default=5, help="max_grad_norm") #5
parser.add_argument("--use_curriculum_learning", type=eval, choices=[True, False], default='True', help="use_curriculum_learning") #True
parser.add_argument("--cl_decay_steps", type=int, default=2000, help="cl_decay_steps") #2000
parser.add_argument('--test_every_n_epochs', type=int, default=5, help='test_every_n_epochs')
parser.add_argument('--gpu', type=int, default=0, help='which gpu to use')
parser.add_argument('--seed', type=int, default=100, help='random seed.') #100

parser.add_argument('--trainval_ratio', type=float, default=0.5, help='the ratio of training and validation data among the total')
parser.add_argument('--dataset', type=str, choices=['Flu_11_12', 'Flu_13_14', 'China_Air',
                                                    'Argentina', 'Brazil', 'Chile', 'Colombia',
                                                    'Mexico', 'Paraguay', 'Uruguay', 'Venezuela'],
                    default='China_Air', help='which dataset to run')
# parser.add_argument('--num_nodes', type=int, default=56, help='num_nodes')
# parser.add_argument('--num_class', type=int, default=5, help='number of classes') # civil 3, flu 5, air 5
args = parser.parse_args()
        
if args.dataset == 'Flu_11_12':
    data_path = f'../data/{args.dataset}/2011-2012_flu_normalized_y.csv'
    args.num_nodes = 56
    args.num_class = 5
elif args.dataset == 'Flu_13_14':
    data_path = f'../data/{args.dataset}/2013-2014_flu_normalized_y.csv'
    args.num_nodes = 57
    args.num_class = 5
elif args.dataset == 'China_Air':
    data_path = f'../data/{args.dataset}/china_air_y.csv'
    args.num_nodes = 10
    args.num_class = 5
elif args.dataset == 'Argentina':
    data_path = f'../data/{args.dataset}/2013-2014_argentina_y.csv'
    args.num_nodes = 23
    args.num_class = 3
elif args.dataset == 'Brazil':
    data_path = f'../data/{args.dataset}/2013-2014_brazil_y.csv'
    args.num_nodes = 26
    args.num_class = 3
elif args.dataset == 'Brazil_Class':
    data_path = f'../data/{args.dataset}/2013-2014_brazil_y_class.csv'
    args.num_nodes = 11
    args.num_class = 5
elif args.dataset == 'Chile':
    data_path = f'../data/{args.dataset}/2013-2014_chile_y.csv'
    args.num_nodes = 14
    args.num_class = 3
elif args.dataset == 'Colombia_Class':
    data_path = f'../data/{args.dataset}/2013-2014_colombia_y_class.csv'
    args.num_nodes = 13
    args.num_class = 4
elif args.dataset == 'Mexico':
    data_path = f'../data/{args.dataset}/2013-2014_mexico_y.csv'
    args.num_nodes = 32
    args.num_class = 3
elif args.dataset == 'Mexico_Class':
    data_path = f'../data/{args.dataset}/2013-2014_mexico_y_class.csv'
    args.num_nodes = 12
    args.num_class = 5
elif args.dataset == 'Paraguay':
    data_path = f'../data/{args.dataset}/2013-2014_paraguay_y.csv'
    args.num_nodes = 18
    args.num_class = 3
elif args.dataset == 'Paraguay_Class':
    data_path = f'../data/{args.dataset}/2013-2014_paraguay_y_class.csv'
    args.num_nodes = 9
    args.num_class = 5
elif args.dataset == 'Uruguay':
    data_path = f'../data/{args.dataset}/2013-2014_uruguay_y.csv'
    args.num_nodes = 19
    args.num_class = 3
elif args.dataset == 'Venezuela':
    data_path = f'../data/{args.dataset}/2013-2014_venezuela_y.csv'
    args.num_nodes = 23
    args.num_class = 3
elif args.dataset == 'Venezuela_Class':
    data_path = f'../data/{args.dataset}/2013-2014_Venezuela_y_class.csv'
    args.num_nodes = 14
    args.num_class = 4
else:
    pass # including more datasets in the future    

model_name = 'MegaCRN'
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
path = f'../save/{args.dataset}_{model_name}_{timestring}'
logging_path = f'{path}/{model_name}_{timestring}_logging.txt'
score_path = f'{path}/{model_name}_{timestring}_scores.txt'
epochlog_path = f'{path}/{model_name}_{timestring}_epochlog.txt'
modelpt_path = f'{path}/{model_name}_{timestring}.pt'
if not os.path.exists(path): os.makedirs(path)
shutil.copy2(sys.argv[0], path)
shutil.copy2(f'{model_name}.py', path)
shutil.copy2('utils.py', path)
    
logger = logging.getLogger(__name__)
logger.setLevel(level = logging.INFO)
class MyFormatter(logging.Formatter):
    def format(self, record):
        spliter = ' '
        record.msg = str(record.msg) + spliter + spliter.join(map(str, record.args))
        record.args = tuple() # set empty to args
        return super().format(record)
formatter = MyFormatter()
handler = logging.FileHandler(logging_path, mode='a')
handler.setLevel(logging.INFO)
handler.setFormatter(formatter)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(formatter)
logger.addHandler(handler)
logger.addHandler(console)

logger.info('model', model_name)
logger.info('dataset', args.dataset)
logger.info('trainval_ratio', args.trainval_ratio)
logger.info('val_ratio', args.val_ratio)
logger.info('num_nodes', args.num_nodes)
logger.info('seq_len', args.seq_len)
logger.info('horizon', args.horizon)
logger.info('input_dim', args.input_dim)
logger.info('output_dim', args.output_dim)
logger.info('num_rnn_layers', args.num_rnn_layers)
logger.info('rnn_units', args.rnn_units)
logger.info('max_diffusion_step', args.max_diffusion_step)
logger.info('mem_num', args.mem_num)
logger.info('mem_dim', args.mem_dim)
logger.info('loss', args.loss)
logger.info('separate loss lamb', args.lamb)
logger.info('compact loss lamb1', args.lamb1)
logger.info('batch_size', args.batch_size)
logger.info('epochs', args.epochs)
logger.info('patience', args.patience)
logger.info('lr', args.lr)
logger.info('epsilon', args.epsilon)
logger.info('steps', args.steps)
logger.info('lr_decay_ratio', args.lr_decay_ratio)
logger.info('use_curriculum_learning', args.use_curriculum_learning)

cpu_num = 1
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
# device = torch.device("cuda:{}".format(args.gpu)) if torch.cuda.is_available() else torch.device("cpu")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Please comment the following three lines for running experiments multiple times.
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available(): torch.cuda.manual_seed(args.seed)
print(torch.random.initial_seed())
#####################################################################################################

data = {}
for category in ['train', 'val', 'test']:
    cat_data = np.load(os.path.join(f'../data/{args.dataset}', category + '.npz'))
    data['x_' + category] = cat_data['x']
    data['y_' + category] = cat_data['y']
scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
for category in ['train', 'val', 'test']:
    data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
    data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
data['train_loader'] = DataLoader(data['x_train'], data['y_train'], args.batch_size, shuffle=True)
data['val_loader'] = DataLoader(data['x_val'], data['y_val'], args.batch_size, shuffle=False)
data['test_loader'] = DataLoader(data['x_test'], data['y_test'], args.batch_size, shuffle=False)

def main():
    logger.info(args.dataset, 'training and testing started', time.ctime())
    logger.info('train xs.shape, ys.shape', data['x_train'].shape, data['y_train'].shape)
    logger.info('val xs.shape, ys.shape', data['x_val'].shape, data['y_val'].shape)
    logger.info('test xs.shape, ys.shape', data['x_test'].shape, data['y_test'].shape)
    traintest_model()
    logger.info(args.dataset, 'training and testing ended', time.ctime())
    
if __name__ == '__main__':
    main()


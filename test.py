
#
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import argparse
import torch
import cv2

#设置随机数种子
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

#
from U_net import UNet
from Data_loader import Data_loader
device = 'cuda' if torch.cuda.is_available() else 'cpu'

args = argparse.ArgumentParser()
#总共训练迭代次数
args.epochs = 20
#批次
args.batch_size = 1
#学习率
args.learning_rate = 0.0001
#图片缩放比例
args.scale = 0.5
#达到总数据的(0-100)，验证一次，百分比
args.validation_every = 10
#梯度修剪的最大梯度
args.max_gradient = 0.1
#
net = UNet(n_channels=3, n_classes=1, bilinear=True).to(device)
net.load_state_dict(torch.load('model/model_0_.pkl'))
print('parameter loaded')
loffF = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, weight_decay=1e-8, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=2)
# print('net = ', net)
#
args.imgs_dir = 'data/imgs/'
args.masks_dir = 'data/masks/'
scale = 0.5
data_loader = Data_loader(args.imgs_dir, args.masks_dir, args.scale)
#每次迭代的损失
train_loss_total_his = np.zeros((args.epochs), dtype=np.float32)
train_acc_total_his = np.zeros((args.epochs), dtype=np.float32)
test_loss_total_his = np.zeros((args.epochs), dtype=np.float32)
test_acc_total_his = np.zeros((args.epochs), dtype=np.float32)
#
def plot_res(train_loss_his, test_loss_his, ite):
    np.savetxt('result/train_loss_his_'+str(ite)+'_.txt', train_loss_his)
    np.savetxt('result/test_loss_his_'+str(ite)+'_.txt', test_loss_his)
    plt.plot(train_loss_his)
    plt.title('train_loss_his')
    plt.savefig('result/train_loss_his_'+str(ite)+'_.png')
    plt.clf()
    plt.plot(test_loss_his)
    plt.title('test_loss_his')
    plt.savefig('result/test_loss_his_'+str(ite)+'_.png')
    plt.clf()

def get_accuracy(SR,GT,threshold=0.5):

    #
    SR_show = torch.squeeze(SR, 0)
    SR_show = torch.squeeze(SR_show, 0)
    SR_show = torch.unsqueeze(SR_show, -1).cpu().data.numpy()
    #
    GT_show = torch.squeeze(GT, 0)
    GT_show = torch.squeeze(GT_show, 0)
    GT_show = torch.unsqueeze(GT_show, -1).cpu().data.numpy()
    #
    cv2.imshow('GT_show', GT_show)
    cv2.imshow('SR_show', SR_show)
    cv2.waitKey()


    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc

# 开始训练
for ite in range(args.epochs):
    net.train()  #
    print('----------epoch = ', ite, ' running')
    #重新开始循环
    data_loader.train_index = 0
    #进度条
    pbar_train = tqdm(total=data_loader.train_file_lens)
    train_loss_his = np.zeros((data_loader.train_file_lens), dtype=np.float32)
    train_acc_his = np.zeros((data_loader.train_file_lens), dtype=np.float32)
    test_loss_his = []
    test_acc_his = []
    for step_train in range(data_loader.train_file_lens):
        img, mask = data_loader.train_next()
        img = torch.unsqueeze(torch.FloatTensor(img),0).to(device)
        mask = torch.unsqueeze(torch.FloatTensor(mask),0).to(device)
        # print(img.shape, mask.shape)#torch.Size([1, 3, 640, 959]) torch.Size([1, 1, 640, 959])
        mask_pred = net(img)
        loss = loffF(mask_pred, mask)
        optimizer.zero_grad()
        loss.backward()
        train_loss_his[step_train] = loss.cpu().data.numpy()
        train_acc_his[step_train] = get_accuracy(mask_pred, mask)
        torch.nn.utils.clip_grad_value_(net.parameters(), args.max_gradient)
        optimizer.step()
        #一定要及时清理内存
        del img, mask, mask_pred
        pbar_train.update()
        #开始评估
        if step_train % (data_loader.train_file_lens // args.validation_every) == 0:
            print('start evaluation, step_train = ', step_train)
            # 重新开始循环
            data_loader.test_index = 0
            test_loss_his.append(0)
            test_acc_his.append(0)
            pbar_test = tqdm(total=data_loader.test_file_lens)
            for step_test in range(data_loader.test_file_lens):
                img, mask = data_loader.test_next()
                img = torch.unsqueeze(torch.FloatTensor(img), 0).to(device)
                mask = torch.unsqueeze(torch.FloatTensor(mask), 0).to(device)
                #在这里不计算梯度，可以节省一点内存
                with torch.no_grad(): mask_pred = net(img)
                loss = loffF(mask_pred, mask)
                test_loss_his[-1] += loss.cpu().data.numpy()
                test_acc_his[-1] += get_accuracy(mask_pred, mask)
                # 一定要及时清理内存
                del img, mask, mask_pred
                pbar_test.update()
            #取损失的平均
            test_loss_his[-1] /= data_loader.test_file_lens
            test_acc_his[-1] /= data_loader.test_file_lens
            #改变学习率
            scheduler.step(test_loss_his[-1])
            #绘图
            plot_res(train_loss_his[:step_train], test_loss_his, ite)
            pbar_test.close()
            print('evaluation', ' loss = ', test_loss_his[-1], ' acc = ', test_acc_his[-1])
            print('end evaluation')
    #保存模型
    torch.save(net.state_dict(), 'model/model_'+str(ite)+'_.pkl')
    pbar_train.close()
    train_loss_total_his[ite] = np.mean(train_loss_his)
    test_loss_total_his[ite] = np.mean(test_loss_his)
    np.savetxt('result/train_loss_total_his.txt', train_loss_total_his[:ite])
    np.savetxt('result/test_loss_total_his.txt', test_loss_total_his[:ite])
    print('model saved')



from __future__ import print_function
from torch.utils import data
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR
from models.losses import FocalLoss, MPSLoss, MHELoss
from dataset.dataset import BoDaiDataset
from utils import *
import yaml
from models.seresnet import se_resnet50
import torch.nn as nn


def train_epoch(trainloader, model, criterion, optimizer, config, epoch_cnt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.train()
    time_cost = []
    for i, batch in enumerate(trainloader):
        start_time = time.time()
        data_input, labels = batch
        ### image tensor list
        data_input = data_input.to(device)
        # tensor([ 73,  23,  66,  26,  48,  68, 125,  20,  64,  76, 121, 103,  35,  48,
        #  85, 113,  43,  86,  97,  66,   0,  77,  31, 127,  16,  30,  28,  64,
        #  60,  68,  40,   1], device='cuda:0')
        labels = labels.to(device).long()
        output = model(data_input)
        loss = criterion(output, labels)

        ### if without optimizer.zero_grad(), the gradient will add together
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_cost.append(time.time() - start_time)
        iters = (epoch_cnt - 1) * len(trainloader) + i + 1
        if iters % config['print_freq'] == 0:
            speed = 1.0 / (np.mean(time_cost))
            time_str = time.asctime(time.localtime(time.time()))
            learning_rates = [each['lr'] for each in optimizer.param_groups]
            print('{} train epoch {} iter {} {} iters/s loss {}'.format(time_str, epoch_cnt, iters, speed, loss.item()))
            print('learning rate: model %f' % (learning_rates[0]))

def train_epoch_with_two_criterion(trainloader, model, criterion1, criterion2, optimizer, config, epoch_cnt):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Sigmoid_fun = nn.Sigmoid()
    model.train()
    time_cost = []
    for i, batch in enumerate(trainloader):
        start_time = time.time()

        data_input, labels, bce_label = batch

        data_input = data_input.to(device)
        labels = labels.to(device).long()
        bce_label = bce_label.cuda()

        output = model(data_input)

        loss1 = criterion1(output, labels)
        loss2 = criterion2(Sigmoid_fun(output), bce_label)

        loss = loss1 + loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_cost.append(time.time() - start_time)

        iters = (epoch_cnt - 1) * len(trainloader) + i + 1

        if iters % config['print_freq'] == 0:
            speed = 1.0 / (np.mean(time_cost))
            time_str = time.asctime(time.localtime(time.time()))
            learning_rates = [each['lr'] for each in optimizer.param_groups]
            print('{} train epoch {} iter {} {} iters/s loss {}'.format(time_str, epoch_cnt, iters, speed, loss.item()))
            print('learning rate: model %f' % (learning_rates[0]))


from sklearn import metrics
import matplotlib.pyplot as plt
def val_epcho(valloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    y_true = []
    scores = []
    for i, batch in enumerate(valloader):
        data_input, labels = batch
        gt_label = int(labels)
        data_input = data_input.to(device)
        labels = labels.to(device).long()
        output = model(data_input)
        output = F.softmax(output, dim=1)
        index = output.cpu().data.numpy().argmax()
        score = float(output.cpu().data.numpy().max())
        scores.append(score)
        total += labels.size(0)
        correct += (index == labels).sum()

        if int(index) == gt_label:
            y_true.append(1)
        else:
            y_true.append(0)

    correct = int(correct)
    Accuracy_pre = correct / total
    print(100 * Accuracy_pre)
    print('Accuracy of the network on the test images: %.6f %%' % (100 * Accuracy_pre))

    # roc 曲线
    # fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    # print(thresholds)
    # plt.plot(fpr, tpr)
    # plt.show()


def val_epcho_with_two_criterion(valloader, model):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()
    correct = 0
    total = 0
    y_true = []
    scores = []
    for i, batch in enumerate(valloader):
        start_time = time.time()

        data_input, labels, _ = batch
        gt_label = int(labels)

        data_input = data_input.to(device)
        labels = labels.to(device).long()
        output = model(data_input)
        output = F.softmax(output, dim=1)
        index = output.cpu().data.numpy().argmax()
        score = float(output.cpu().data.numpy().max())
        scores.append(score)
        total += labels.size(0)
        correct += (index == labels).sum()

        if int(index) == gt_label:
            y_true.append(1)
        else:
            y_true.append(0)

    correct = int(correct)
    Accuracy_pre = correct / total
    print(100 * Accuracy_pre)
    print('Accuracy of the network on the test images: %.6f %%' % (100 * Accuracy_pre))

    # roc 曲线
    fpr, tpr, thresholds = metrics.roc_curve(y_true, scores)
    print(thresholds)
    plt.plot(fpr, tpr)
    plt.show()


def main():
    # choose device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    gpu_num = torch.cuda.device_count()
    print(device, gpu_num)

    # read configuration files
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)
    print(config)


    # choose pretrained model: se_resnet50
    model = se_resnet50(pretrained='imagenet')
    if config['input_shape'][1] == 128:
        # 128 bodai
        model.avg_pool = nn.AvgPool2d(4, stride=1)
    elif config['input_shape'][1] == 320:
        # 320 cow body
        model.avg_pool = nn.AvgPool2d(10, stride=1)

    train_dataset = BoDaiDataset(config['train_root'],
                                 config['class_label'],
                                 phase='train',
                                 input_shape=config['input_shape'])

    num_classes = train_dataset.num_classes
    print("data set length:", len(train_dataset))
    # exit()
    model.last_linear = nn.Linear(2048, num_classes)
    train_loader = data.DataLoader(train_dataset,
                                  batch_size=config['train_batch_size'],
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True)
    print('%d train iters per epoch'% (len(train_loader)))

    # val
    val_dataset = BoDaiDataset(config['val_root'], config['class_label'], phase='val', input_shape=config['input_shape'])
    val_loader = data.DataLoader(val_dataset,
                                  batch_size=1,
                                  shuffle=False,
                                  num_workers=8,
                                  pin_memory=True)

    if config['loss'] == 'focal':
        criterion = FocalLoss(num_classes, gamma=2)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)
    lr = config['lr']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.SGD([{'params': model.parameters()}],
                                lr=lr, weight_decay=weight_decay, momentum=0.9)
    epoch_cnt = 0
    adjust_learning_rate(optimizer, lr)
    scheduler = MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=0.1)  # , last_epoch=epoch_cnt
    for i in range(epoch_cnt + 1, config['epochs'] + 1):
        train_epoch(train_loader, model, criterion, optimizer, config, i)
        val_epcho(val_loader, model)
        scheduler.step()
        if i % config['save_freq'] == 0 or i == config['epochs']:
            save_checkpoint(model, optimizer, config['save_path'], i, gpu_num>1)


if __name__ == '__main__':
    main()
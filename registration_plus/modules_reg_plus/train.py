import torch
from torch.utils import data
import os
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import time
import numpy as np
import re

from dataset_reg_plus.dataset import Dataset_reg
from modules_reg_plus.utils_reg import load_network_structure, get_subtitle


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def save_checkpoint(model, optimizer, save_path, epoch):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    checkpoint_path = os.path.join(save_path, str(epoch))
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)

    model_name = os.path.join(checkpoint_path, 'model.pth')
    optimizer_name = os.path.join(checkpoint_path, 'optimizer.pth')

    # if use_multi_gpu:
    #     if model is not None:
    #         torch.save(model.module.state_dict(), model_name)
    # else:

    if model is not None:
        torch.save(model.state_dict(), model_name)

    if optimizer is not None:
        torch.save(optimizer.state_dict(), optimizer_name)


def train_epoch(trainloader, model, classifier, criterion, optimizer, config, save_path, epoch_cnt):
    global learning_rates
    model.train()
    time_cost = []
    loss_items = []
    for i, batch in enumerate(trainloader):
        start_time = time.time()
        data_input, labels = batch
        data_input = data_input.to(config['device'])
        labels = labels.to(config['device']).long()
        features = model(data_input)

        if config['phase']=='train':
            output = classifier(features)
        elif config['phase']=='finetune':
            output = classifier(features, labels)
        elif config['phase']=='triplet':
            output = classifier(features, labels)

        loss = criterion(output, labels)
        loss_items.append(loss.item())
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
            print('learning rate: model %f, classify %f' % (learning_rates[0], learning_rates[1]))

    # save model
    if epoch_cnt % config['save_freq'] == 0 or epoch_cnt == config['epochs']:
        save_checkpoint(model, optimizer, save_path, epoch_cnt)
        info = "TRAIN_epoch{}_lr{}_loss_{}.txt".format(epoch_cnt, learning_rates[0], np.mean(loss_items))
        f = open(os.path.join(save_path, str(epoch_cnt), info), 'w')
        f.close()


def train(config:dict):
    device = config['device']
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    train_dataset = Dataset_reg(config['train_root'], phase='train', input_shape=config['input_shape'])
    num_classes = train_dataset.num_classes
    print('num_classes: ', num_classes)

    trainloader = data.DataLoader(train_dataset,
                                  batch_size=config['train_batch_size'],
                                  shuffle=True,
                                  # num_workers=cpu_count(),
                                  num_workers=4 if config['phase']!='triplet' else 0,
                                  pin_memory=True)
    print('%d train iters per epoch' % (len(trainloader)))

    backbone, classifier, criterion = load_network_structure(config, single_model=False, num_classes=num_classes)
    classifier.to(device)
    backbone.to(device)
    criterion.to(device)
    print(backbone, '\n', classifier, '\n', criterion)

    lr = config['lr']
    weight_decay = config['weight_decay']
    optimizer = torch.optim.SGD([{'params': backbone.parameters()},
                                 {'params': classifier.parameters()}],
                                lr=lr, weight_decay=weight_decay, momentum=0.9)

    config['checkpoint_subtitle']= get_subtitle(config, train_dataset.transforms)
    epoch_cnt = 0
    if config['resume']:
        model_path = config['resume']
        backbone.load_state_dict(torch.load(model_path, map_location=device))
        epoches = re.findall('(\d+0)', model_path)
        epoch_cnt = int(epoches[-1])
        print('resume train model. model path:', model_path)
        print('resume epoch count:', epoch_cnt)

    elif config['phase']=='finetune':
        model_subtitle = config['checkpoint_subtitle'].replace('_finetune', '')
        model_path = os.path.join(config['save_path'], model_subtitle, str(config['epochs']), 'model.pth')
        print('load model path:', model_path)
        backbone.load_state_dict(torch.load(model_path, map_location=device))

    warmup_epochs = config['warmup_epochs']
    if warmup_epochs>0:
        print('start warmup')
        scheduler_w = LambdaLR(optimizer, lr_lambda=lambda epoch: float(epoch + 1) / warmup_epochs)
        for i in range(1, warmup_epochs + 1):
            train_epoch(trainloader, backbone, classifier, criterion, optimizer, config,
                        os.path.join(config['save_path'], config['checkpoint_subtitle']), i)
            scheduler_w.step()
        epoch_cnt += warmup_epochs

    adjust_learning_rate(optimizer, lr)
    scheduler = MultiStepLR(optimizer, milestones=config['lr_milestones'], gamma=0.1, last_epoch=epoch_cnt)
    for i in range(epoch_cnt + 1, config['epochs'] + 1):
        train_epoch(trainloader, backbone, classifier, criterion, optimizer, config,
                    os.path.join(config['save_path'], config['checkpoint_subtitle']), i)
        scheduler.step()

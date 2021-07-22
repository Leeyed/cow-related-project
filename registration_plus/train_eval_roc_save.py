from __future__ import print_function
import yaml
import torch
from modules_reg_plus.train import train
from modules_reg_plus.eval import eval_cow_body_and_save


if __name__ == '__main__':
    with open('cfg.yaml', 'r') as fd:
        config = yaml.load(fd, Loader=yaml.FullLoader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config['device'] = device
    config['phase']='train'

    for key, val in config.items():
        print(key, val)

    # train part
    # config['phase']='train'
    # train(config=config)
    # config['phase']='finetune'
    # train(config=config)

    # eval part
    # config['phase'] = 'train'
    # eval_cow_body_and_save(config=config)
    config['phase'] = 'finetune'
    eval_cow_body_and_save(config=config)


    # config['phase']='triplet'
    # train(config=config)
    # eval_cow_body_and_save(config=config)
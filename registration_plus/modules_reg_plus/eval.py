import torch
import os
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pandas as pd

from dataset_reg_plus.dataset import Dataset_reg
from modules_reg_plus.utils_reg import load_network_structure, get_subtitle, get_dirs_all_data, save_to_final_result
from modules_reg_plus.eval_subtask import get_ref_test_feature, gen_sample_pair, get_th_from_roc, cal_sim_pairs


def get_acc_and_draw_roc(config, labels: list, predicts: list, thresholds:list, name: str):
    fpr, tpr, _ = roc_curve(labels, predicts)
    plt.figure(figsize=(15, 10))
    plt.plot(fpr, tpr)
    plt.xticks(rotation=90)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    path = os.path.join(config['save_path'], config['checkpoint_subtitle'])
    plt.savefig(os.path.join(path, name + '.jpg'))
    auc_score = auc(fpr, tpr)

    labels, predicts = np.array(labels), np.array(predicts)
    ans = [auc_score]
    for th in thresholds:
        f_index_th = np.where(predicts>=th)
        _labels = labels[f_index_th]
        _predicts = predicts[f_index_th]

        f_index_lab = np.where(_labels==1)
        ans.append(th)
        ans.append(len(f_index_lab[0])/(len(f_index_th[0])+0.000001))
        ans.append(len(f_index_lab[0]))
    return ans


@torch.no_grad()
def get_data_and_predict_cow(config, model):
    ref_feature_dict, test_feature_dict = get_ref_test_feature(config, model)

    ref_imgs = list(ref_feature_dict.keys())
    ref_labels = [img.split(os.sep)[-2] for img in ref_imgs]
    ref_labels = list(set(ref_labels))
    sorted_ref_imgs = []
    for each_label in ref_labels:
        for path in ref_imgs:
            if path.split('/')[-2] == each_label:
                sorted_ref_imgs.append(path)
    sorted_ref_feature = []
    for each_pic in sorted_ref_imgs:
        # sorted_ref_feature.append(torch.tensor(ref_feature_dict[each_pic]).to(config['device']))
        sorted_ref_feature.append(ref_feature_dict[each_pic].clone().detach().to(config['device']))
    sorted_ref_feature_tensor = torch.stack(sorted_ref_feature)

    detail = open(os.path.join(config['save_path'], config['checkpoint_subtitle'], 'error_detail.txt'), 'w')

    each_label_error = {}
    max_labels, max_predicts = [], []
    test_list = get_dirs_all_data(config['test_dir'])
    for each_test in test_list:
        #  [(test_path, ref_path, 最大值), (均值最大的label, 均值), (label, 中位数最大值)]
        predict_data = get_max_mean_median_similarity_from_ref(each_test, test_feature_dict[each_test],
                                                               sorted_ref_feature_tensor, ref_labels, sorted_ref_imgs)
        gt_label = each_test.split('/')[-2]
        if gt_label in each_label_error:
            each_label_error[gt_label][0] += 1  # 该嵌件数量加1
        else:
            # label 总数， 预测错误的数量
            each_label_error[gt_label] = [1, 0]

        ref_label = predict_data[0][1].split('/')[-2]
        gt_label = each_test.split('/')[-2]
        if gt_label == ref_label:
            max_labels.append(1)
        else:
            txt_line = "%s,%s,%f\n" % (predict_data[0][0], predict_data[0][1], predict_data[0][2])
            detail.write(txt_line)
            each_label_error[gt_label][1] += 1
            max_labels.append(0)
        max_predicts.append(predict_data[0][2])
    detail.close()

    csv_data = {'label': [], 'max_acc': [], 'num': []}
    for key, value in each_label_error.items():
        # csv 数据写入
        csv_data['label'].append(key)
        csv_data['max_acc'].append((value[0] - value[1]) / value[0])
        csv_data['num'].append(value[0])

    acc_num = len(np.where(np.array(max_labels)==1)[0])
    max_acc = acc_num / len(test_list)
    print(20 * '*****')
    print("总的预测图片张数为: ", len(test_list))
    print("预测正确的图片张数为: ", acc_num)
    print("准确率为: ", max_acc)
    csv_data['label'].append('all_labels')
    csv_data['max_acc'].append(max_acc)

    csv_data['num'].append(len(test_list))

    # 保存csv文件
    csv_result = os.path.join(config['save_path'], config['checkpoint_subtitle'], 'csv_result.txt')
    df = pd.DataFrame(csv_data)
    df.to_csv(csv_result, index=False, sep=',')

    gen_sample_pair(config)
    # cal_sim_pairs(config, model)
    thresholds = get_th_from_roc(config, model)
    max_acc_data = get_acc_and_draw_roc(config, max_labels, max_predicts, thresholds, 'max')
    # mean_acc_data = get_acc_and_draw_roc(mean_labels, mean_predicts, subtitle, 'mean')
    # median_acc_data = get_acc_and_draw_roc(median_labels, median_predicts, subtitle, 'median')
    acc_data = [max_acc]
    acc_data.extend(max_acc_data)
    acc_data.append(len(test_list))
    return acc_data


@torch.no_grad()
def get_max_mean_median_similarity_from_ref(each_test, test_feature, ref_feature_matrix, ref_labels: list,
                                            ref_imgs: list):
    '''
    用于计算一张图片, 与对应嵌件的参考图片 以及所有的参考图片的进行对比, 求出相似度的[(最大值, path1, path2),(最小值, path1, path2),均值, 所有参考图片相似度均值最大的]
    :param each_test:  image url
    :param feature1:  当前图片的feature vector
    :param ref_feature_matrix:  所有ref中的图片vector    ref_feature_matrix 顺序 与 ref_imgs 顺序一致
    :param ref_labels:  所有ref的label
    :param ref_imgs:  所有 ref iamge url
    :return:
    '''
    all_data = []
    # test_feature 扩张成 test_feature_matrix
    test_feature_matrix = test_feature.expand_as(ref_feature_matrix)
    predicts = torch.cosine_similarity(test_feature_matrix, ref_feature_matrix).cpu().numpy().reshape(-1)

    max_id = np.argmax(predicts)
    all_data.append((each_test, ref_imgs[max_id], predicts[max_id]))

    each_label_num = len(predicts) // len(ref_labels)
    predicts = predicts.reshape((len(ref_labels), each_label_num))
    means = predicts.mean(axis=0)
    max_mean_id = np.argmax(means)
    all_data.append((means[max_mean_id], ref_labels[max_mean_id]))

    medians = np.median(predicts, axis=1)
    max_median_id = np.argmax(medians)
    all_data.append((medians[max_median_id], ref_labels[max_median_id]))

    return all_data


@torch.no_grad()
def eval_cow_body_and_save(config: dict):
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    input_shape = config['input_shape']
    train_dataset = Dataset_reg(config['train_root'], phase='train', input_shape=input_shape)
    config['checkpoint_subtitle'] = get_subtitle(config, train_dataset.transforms)

    model_path = os.path.join(config['save_path'], config['checkpoint_subtitle'], str(config['epochs']), 'model.pth')
    print('model path', model_path)
    # model_path = '/home/liusheng/deepLearningProjects/projects/registration_plus/checkpoints/model.pth'
    # exit()
    model = load_network_structure(config, single_model=True)
    print('load model', model_path)
    model_path = '/home/liusheng/deepLearningProjects/projects/Cow_track_demo/trained_weight/XzMask125Sameway_resnetface20_rot180HorVer_192_finetune_acc787.pth'
    model.load_state_dict(torch.load(model_path, map_location=config['device']))
    model.to(config['device'])

    print('****************** start eval ******************')
    model.eval()
    # acc_data = [auc, th1, acc1, num1, th2, acc2, num2, th2, acc2, num2]
    acc_data = get_data_and_predict_cow(config, model)
    save_to_final_result(config, acc_data, model_path)

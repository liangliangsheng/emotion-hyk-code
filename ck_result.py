from core_code import load
from core_code import networks
from core_code import util
import argparse
import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

torch.cuda.set_device(0)
DEVICE_CUDA = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE_CPU = torch.device("cpu")
cate2label = {'CK+': {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                      'Angry': 1, 'Disgust': 2, 'Fear': 3, 'Happy': 0, 'Contempt': 5, 'Sad': 4, 'Surprise': 6},

              'AFEW': {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Neutral', 6: 'Surprise',
                       'Angry': 1, 'Disgust': 2, 'Fear': 3, 'Happy': 0, 'Neutral': 5, 'Sad': 4, 'Surprise': 6}}


def generate_true_result(prime_path, true_result_path):
    res = []
    with open(prime_path, 'r') as imf:
        imf = imf.readlines()
    for line in imf:
        if 'fold' in line:
            pass
        else:
            file, label = line.strip().split(' ')
            label = cate2label['CK+'][label]
            res.append(file + ' ' + str(label))
    with open(true_result_path, 'w') as fp:
        for line in res:
            fp.write(line)
            fp.write('\n')


def generate_pred_result(model_path, pred_result_path, aggregate_mode='last', patch_size=3):
    # 加载模型
    model = networks.resnet18_at(aggregate_mode=aggregate_mode, patch_size=patch_size)
    _, _, model = util.load_model(model, model_path, True)

    # 加载数据
    video_root = './data/face/ck_face'
    video_list = './data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
    points_name = '16pts.list'
    batch = 64
    num_workers = 4

    all_loader = load.ck_faces_hyk_result(video_root, video_list, points_name, num_workers, batch)
    acc_frame, true_video, video_length = val(all_loader, model, pred_result_path)
    print(acc_frame.avg, true_video, video_length)


def val(all_loader, model, pred_result_path):
    losses = util.AverageMeter()
    top_frame = util.AverageMeter()
    model.eval()

    # [batch] / [batch] / [batch]
    pred_label_list = []
    true_label_list = []
    video_index_list = []

    res = []
    with torch.no_grad():
        for i, (input_first, input_second, input_third, target, ponits_three, video_index) in enumerate(all_loader):
            true_label = target.to(DEVICE_CUDA)
            input_var = torch.stack([input_first, input_second, input_third], dim=4).to(DEVICE_CUDA)
            ponits_var = ponits_three.to(DEVICE_CUDA)

            # [batch,7]
            pred_score = model(input_var, ponits_var)
            loss = F.cross_entropy(pred_score, true_label)

            (_, pred_label) = torch.max(pred_score.detach(), dim=1)
            acc_iter = (pred_label == true_label).sum(dim=0).float() / pred_label.shape[0]
            losses.update(loss.item(), input_var.size(0))
            top_frame.update(acc_iter, input_var.size(0))

            pred_label_list.append(pred_label)
            true_label_list.append(true_label)
            video_index_list.append(video_index)

            true_label.to(DEVICE_CPU)
            input_var.to(DEVICE_CPU)
            ponits_var.to(DEVICE_CPU)

        '''----- 计算帧正确率-----'''
        true_num = 0
        pred_label_list = torch.cat(pred_label_list, dim=0).cpu()
        true_label_list = torch.cat(true_label_list, dim=0).cpu()
        video_index_list = torch.cat(video_index_list, dim=0)
        length = max(video_index_list) + 1
        for i in range(length):
            index = torch.nonzero(video_index_list == i).squeeze()
            value = torch.gather(pred_label_list, dim=0, index=index)
            label = true_label_list[index[0]]
            pred_label = np.argmax(np.bincount(value.numpy()))
            res.append(pred_label)
            if label == pred_label:
                true_num += 1

        with open(pred_result_path, 'w') as fp:
            for line in res:
                fp.write(str(line))
                fp.write('\n')
        top_video = true_num / length.float()

    return top_frame, true_num, length


def check(true_result_path, pred_result_path):
    with open(true_result_path, 'r') as f:
        true_result = f.readlines()
    with open(pred_result_path, 'r') as f:
        pred_result = f.readlines()
    length = len(true_result)
    equal_count = 0
    for i in range(length):
        if true_result[i].strip().split(' ')[1] == pred_result[i].strip():
            equal_count += 1
    print(equal_count, length)


if __name__ == '__main__':
    # path = './data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
    # target_path = './result/ck+_true_label'
    # generate_true_result(path, target_path)

    generate_pred_result('./model/ck_hyk_last_test.pth.tar', './result/ck+_pred_label')

    # check('./result/ck+_true_label','./result/ck+_pred_label')

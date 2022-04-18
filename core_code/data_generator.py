# coding=utf-8
import os
import pdb
import random

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


# data generator for afew
class VideoDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None, csv=False):
        self.imgs_first, self.index = load_imgs_total_frame(video_root, video_list, rectify_label)
        self.transform = transform

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


#
class TripleImageDataset(data.Dataset):
    def __init__(self, video_root, video_list, rectify_label=None, transform=None):

        self.imgs_first, self.imgs_second, self.imgs_third, self.index = load_imgs_tsn(video_root, video_list,
                                                                                       rectify_label)
        self.transform = transform

    def __getitem__(self, index):

        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)
        return img_first, img_second, img_third, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


def load_imgs_tsn(video_root, video_list, rectify_label):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()

    with open(video_list, 'r') as imf:
        index = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video
            num_per_part = int(img_count) // 3

            if int(img_count) > 3:
                for i in range(img_count):
                    random_select_first = random.randint(0, num_per_part)
                    random_select_second = random.randint(num_per_part, num_per_part * 2)
                    random_select_third = random.randint(2 * num_per_part, len(img_lists) - 1)

                    img_path_first = os.path.join(video_path, img_lists[random_select_first])
                    img_path_second = os.path.join(video_path, img_lists[random_select_second])
                    img_path_third = os.path.join(video_path, img_lists[random_select_third])

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            else:
                for j in range(len(img_lists)):
                    img_path_first = os.path.join(video_path, img_lists[j])
                    img_path_second = os.path.join(video_path, random.choice(img_lists))
                    img_path_third = os.path.join(video_path, random.choice(img_lists))

                    imgs_first.append((img_path_first, label))
                    imgs_second.append((img_path_second, label))
                    imgs_third.append((img_path_third, label))

            ###  return video frame index  #####
            index.append(np.ones(img_count) * id)  # id: 0 : 379
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, imgs_second, imgs_third, index


def load_imgs_total_frame(video_root, video_list, rectify_label):
    imgs_first = list()

    with open(video_list, 'r') as imf:
        index = []
        video_names = []
        for id, line in enumerate(imf):

            video_label = line.strip().split()

            video_name = video_label[0]  # name of video
            label = rectify_label[video_label[1]]  # label of video

            video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
            ###  for sampling triple imgs in the single video_path  ####

            img_lists = os.listdir(video_path)
            img_lists.sort()  # sort files by ascending
            img_count = len(img_lists)  # number of frames in video

            for frame in img_lists:
                # pdb.set_trace()
                imgs_first.append((os.path.join(video_path, frame), label))
            ###  return video frame index  #####
            video_names.append(video_name)
            index.append(np.ones(img_count) * id)
        index = np.concatenate(index, axis=0)
        # index = index.astype(int)
    return imgs_first, index


'''------------------------------------- ck_plus baseline ------------------------------------'''


# data generator for ck_plus baseline
class TenfoldCkBaseLineDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', rectify_label=None, transform=None, fold=1, run_type='train'):
        self.imgs_first, self.index = load_ck_baseline_imgs(video_root, video_list, rectify_label, fold,
                                                            run_type)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert('RGB')
        if self.transform is not None:
            img_first = self.transform(img_first)

        return img_first, target_first, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


# data generator for ck_plus baseline
def load_ck_baseline_imgs(video_root, video_list, rectify_label, fold, run_type):
    imgs_first = list()
    new_imf = list()

    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:
        imf = imf.readlines()
    if run_type == 'train':
        fold_ = list(range(1, 11))
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9, 10] -> [2,3,4,5,6,7,8,9,10]

        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold
            for index, item in enumerate(
                    imf):  # 0, '1-fold\t31\n' in {[0, '1-fold\t31\n'], [1, 'S037/006 Happy\n'], ...}
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    if run_type == 'test':
        fold_ = fold
        fold_str = str(fold_) + '-fold'
        for index, item in enumerate(imf):
            if fold_str in item:
                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])

    index = []
    for id, line in enumerate(new_imf):

        video_label = line.strip().split()

        video_name = video_label[0]  # name of video
        try:
            label = rectify_label[video_label[1]]  # label of video
        except:
            pdb.set_trace()
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending

        img_lists = img_lists[- int(round(len(img_lists))):]

        img_count = len(img_lists)  # number of frames in video
        for frame in img_lists:
            imgs_first.append((os.path.join(video_path, frame), label))
        index.append(np.ones(img_count) * id)

    index = np.concatenate(index, axis=0)
    return imgs_first, index


'''------------------------------------- ck_plus hyk ------------------------------------'''


class TenfoldCkHykDataset(data.Dataset):
    def __init__(self, video_root='', video_list='', points_name='', rectify_label=None, transform=None, fold=1,
                 run_type='train', mode='test'):

        self.imgs_first, self.imgs_second, self.imgs_third, self.ponits_three, self.index = load_ck_hyk_imgs(video_root,
                                                                                                             video_list,
                                                                                                             points_name,
                                                                                                             rectify_label,
                                                                                                             fold,
                                                                                                             run_type,
                                                                                                             mode)

        self.transform = transform
        self.video_root = video_root

    def __getitem__(self, index):
        path_first, target_first = self.imgs_first[index]
        img_first = Image.open(path_first).convert("RGB")
        if self.transform is not None:
            img_first = self.transform(img_first)

        path_second, target_second = self.imgs_second[index]
        img_second = Image.open(path_second).convert("RGB")
        if self.transform is not None:
            img_second = self.transform(img_second)

        path_third, target_third = self.imgs_third[index]
        img_third = Image.open(path_third).convert("RGB")
        if self.transform is not None:
            img_third = self.transform(img_third)

        ponits_three = torch.tensor(self.ponits_three[index], requires_grad=False)
        return img_first, img_second, img_third, target_first, ponits_three, self.index[index]

    def __len__(self):
        return len(self.imgs_first)


def load_ck_hyk_imgs(video_root, video_list, points_name, rectify_label, fold, run_type, mode):
    imgs_first = list()
    imgs_second = list()
    imgs_third = list()
    ponits_three = list()
    new_imf = list()
    ''' Make ten-fold list '''
    with open(video_list, 'r') as imf:
        imf = imf.readlines()
    if run_type == 'train':
        fold_ = list(range(1, 11))
        fold_.remove(fold)  # [1,2,3,4,5,6,7,8,9,10] -> [2,3,4,5,6,7,8,9,10]
        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold
            for index, item in enumerate(imf):
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    if run_type == 'test':
        fold_ = fold
        fold_str = str(fold_) + '-fold'
        for index, item in enumerate(imf):
            if fold_str in item:
                for j in range(index + 1, index + int(item.split()[1]) + 1):
                    new_imf.append(imf[j])
                break

    if run_type == 'all':
        fold_ = list(range(1, 11))
        for i in fold_:
            fold_str = str(i) + '-fold'  # 1-fold
            for index, item in enumerate(imf):
                if fold_str in item:  # 1-fold in '1-fold\t31\n'
                    for j in range(index + 1, index + int(item.split()[1]) + 1):  # (0 + 1, 0 + 31 + 1 )
                        new_imf.append(imf[j])  # imf[2] = 'S042/006 Happy\n'

    # 测试时加快
    if mode == 'test':
        new_imf = new_imf[0:8]

    ''' Make triple-image list '''
    index = []
    for id, line in enumerate(new_imf):
        video_label = line.strip().split()
        video_name = video_label[0]  # name of video
        label = rectify_label[video_label[1]]  # label of video
        video_path = os.path.join(video_root, video_name)  # video_path is the path of each video
        points_path = os.path.join(video_path, points_name)  # video_path is the path of each video

        img_lists = os.listdir(video_path)
        img_lists.sort()  # sort files by ascending

        # 去除list文件
        for i, item in enumerate(img_lists):
            if item.endswith('.list'):
                continue
            else:
                img_lists = img_lists[i:]
                break
        # img_lists = img_lists[- int(round(len(img_lists))):]
        img_count = len(img_lists)  # number of frames in video
        num_per_part = int(img_count) // 3

        # 读取list文件点数
        points = []
        with open(points_path) as fp:
            lines = fp.readlines()
            for temp in lines:
                points.append([int(item) for item in temp.split(' ')])
        # 结果
        for i in range(num_per_part):
            # for i in range(img_count):
            # pdb.set_trace()
            random_select_first = random.randint(0, num_per_part - 1)
            random_select_second = random.randint(num_per_part, 2 * num_per_part - 1)
            random_select_third = random.randint(2 * num_per_part, img_count - 1)

            img_path_first = os.path.join(video_path, img_lists[random_select_first])
            img_path_second = os.path.join(video_path, img_lists[random_select_second])
            img_path_third = os.path.join(video_path, img_lists[random_select_third])

            imgs_first.append((img_path_first, label))
            imgs_second.append((img_path_second, label))
            imgs_third.append((img_path_third, label))
            ponits_three.append(
                [points[random_select_first], points[random_select_second], points[random_select_third]])
        index.append(np.ones(num_per_part, int) * id)  # id: 0 : 379
        # index.append(np.ones(img_count, int) * id)  # id: 0 : 379
    index = np.concatenate(index, axis=0)
    # index = index.astype(int)
    # pdb.set_trace()
    return imgs_first, imgs_second, imgs_third, ponits_three, index

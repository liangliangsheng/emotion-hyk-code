from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from core_code import data_generator

cate2label = {'CK+': {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Contempt', 6: 'Surprise',
                      'Angry': 1, 'Disgust': 2, 'Fear': 3, 'Happy': 0, 'Contempt': 5, 'Sad': 4, 'Surprise': 6},

              'AFEW': {0: 'Happy', 1: 'Angry', 2: 'Disgust', 3: 'Fear', 4: 'Sad', 5: 'Neutral', 6: 'Surprise',
                       'Angry': 1, 'Disgust': 2, 'Fear': 3, 'Happy': 0, 'Neutral': 5, 'Sad': 4, 'Surprise': 6}}


def ck_faces_baseline(video_root, video_list, fold, batch_train, batch_eval):
    train_dataset = data_generator.TenfoldCkBaseLineDataset(
        video_root=video_root,
        video_list=video_list,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose(
            [transforms.Resize(224), transforms.ToTensor()]),
        fold=fold,
        run_type='train'
    )

    val_dataset = data_generator.TenfoldCkBaseLineDataset(
        video_root=video_root,
        video_list=video_list,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        fold=fold,
        run_type='test'
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_train, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_eval, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader


def ck_faces_hyk(video_root, video_list, points_name, fold, num_workers, batch_train, batch_eval, mode):
    train_dataset = data_generator.TenfoldCkHykDataset(
        video_root=video_root,
        video_list=video_list,
        points_name=points_name,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose([
            transforms.Resize(224), transforms.ToTensor()]),
        fold=fold,
        run_type='train',
        mode=mode
    )

    val_dataset = data_generator.TenfoldCkHykDataset(
        video_root=video_root,
        video_list=video_list,
        points_name=points_name,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        fold=fold,
        run_type='test',
        mode=mode
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_train, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_eval, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def ck_faces_hyk_result(video_root, video_list, points_name, num_workers, batch, ):
    all_dataset = data_generator.TenfoldCkHykDataset(
        video_root=video_root,
        video_list=video_list,
        points_name=points_name,
        rectify_label=cate2label['CK+'],
        transform=transforms.Compose([
            transforms.Resize(224), transforms.ToTensor()]),
        fold=0,
        run_type='all',
        mode='true'
    )

    all_loader = torch.utils.data.DataLoader(
        all_dataset, batch_size=batch, shuffle=False, num_workers=num_workers, pin_memory=True)

    return all_loader


def afew_faces_baseline(root_train, list_train, batch_train, root_eval, list_eval, batch_eval):
    train_dataset = data_generator.VideoDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose(
            [transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_train, shuffle=True,
        num_workers=8, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_eval, shuffle=False,
        num_workers=8, pin_memory=True)
    return train_loader, val_loader


def afew_faces_hyk(root_train, list_train, points_name, num_workers, batch_train, root_eval, list_eval, batch_eval):
    train_dataset = data_generator.TripleImageDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose(
            [transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
    )

    val_dataset = data_generator.VideoDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label['AFEW'],
        transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]),
        csv=False)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_train, shuffle=True,
        num_workers=8, pin_memory=True, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_eval, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

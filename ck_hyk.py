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

parser = argparse.ArgumentParser(description='PyTorch Frame Attention Network Training')
parser.add_argument('-e', '--epochs', default=60, type=int, help='epochs number')
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='initial learning rate')
parser.add_argument('-f', '--fold', default=10, type=int, help='which fold used for ck+ test')
parser.add_argument('-aggr', '--aggregate_mode', default='last', type=str, choices=('last', 'average', 'max'))
parser.add_argument('-re', '--resume', type=str, default=False)
parser.add_argument('-b', '--batch_train', type=int, default=4)
parser.add_argument('-w', '--num_workers', type=int, default=8, )
parser.add_argument('-s', '--save_suffix', type=str, default='test', help='save model file name suffix')
parser.add_argument('-m', '--train_mode', type=str, default='test', choices=('test', 'true'))
parser.add_argument('-p', '--patch_size', type=int, default=3)
args = parser.parse_args()


def main():
    start_date = util.time_now()
    logger = util.Logger('./log/', 'ck_hyk', start_date, args.aggregate_mode)
    logger.print(
        'The aggregate mode is {:}, patch size is {:} ,learning rate: {:}'.format(args.aggregate_mode, args.patch_size,
                                                                                  args.learning_rate))
    # 加载checkpoint
    save_path = './model/ck_hyk_' + args.aggregate_mode + '_' + args.save_suffix + '.pth.tar'
    model = networks.resnet18_at(aggregate_mode=args.aggregate_mode, patch_size=args.patch_size)
    best_acc_frame = 0
    current_epoch = 0
    if args.resume:
        current_epoch, best_acc_video, model = util.load_model(model, save_path, True)
        logger.print('train from ck_hyk_' + args.save_suffix)
    else:
        _parameterDir = './pretrain_model/Resnet18_FER+_pytorch.pth.tar'
        model = util.load_model(model, _parameterDir, False)
        logger.print('train from begin')
    # 优化器
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters(
    )), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=20, gamma=0.2)
    cudnn.benchmark = True
    # 加载数据
    video_root = './data/face/ck_face'
    video_list = './data/txt/CK+_10-fold_sample_IDascendorder_step10.txt'
    points_name = '16pts.list'
    batch_train = args.batch_train
    batch_eval = 64
    num_workers = args.num_workers
    train_loader_set = []
    val_loader_set = []
    for i in range(1, args.fold + 1):
        train_loader, val_loader = load.ck_faces_hyk(video_root, video_list, points_name, i, num_workers, batch_train,
                                                     batch_eval, args.train_mode)
        train_loader_set.append(train_loader)
        val_loader_set.append(val_loader)

    for epoch in range(current_epoch, args.epochs):
        logger.print(
            '---------------------- epoch:{} train begin learning_rate:{} '
            '--------------------------'.format(epoch, optimizer.param_groups[0]['lr']))

        acc_video_sum = 0
        acc_video_count = 0
        acc_frame_sum = 0
        acc_frame_count = 0

        for i in range(1, args.fold + 1):
            logger.print('>>> fold{} begin <<<'.format(i))
            train_loader = train_loader_set[i - 1]
            val_loader = val_loader_set[i - 1]
            train(train_loader, model, optimizer, epoch, logger, i)
            acc_frame, true_video, video_length = val(val_loader, model, epoch, logger, i)

            acc_video_sum += true_video
            acc_video_count += video_length
            acc_frame_sum += acc_frame.sum
            acc_frame_count += acc_frame.count
            logger.print('>>> fold{} end <<<'.format(i))

        acc_video = acc_video_sum / float(acc_video_count)
        acc_frame = acc_frame_sum / float(acc_frame_count)

        logger.print('epoch:{}\ttotal_acc_video:{}\ttotal_acc_frame{}'.format(epoch, acc_video, acc_frame))
        if acc_frame > best_acc_frame:
            logger.print('better acc video')
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'accuracy_frame': acc_frame,
                'accuracy_video': acc_video,
            }, save_path=save_path)
            best_acc_frame = acc_frame
        logger.print('---------------------- epoch{} end --------------------------'.format(epoch))

    lr_scheduler.step()


def train(train_loader, model, optimizer, epoch, logger, fold):
    losses = util.AverageMeter()
    top_frame = util.AverageMeter()

    # [batch] / [batch] / [batch]
    pred_label_list = []
    true_label_list = []
    video_index_list = []

    model.train()
    for i, (input_first, input_second, input_third, target, ponits_three, video_index) in enumerate(train_loader):
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

        # 更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_label_list.append(pred_label)
        true_label_list.append(true_label)
        video_index_list.append(video_index)

        # 每10个batch 记录损失和帧正确率
        if i % 10 == 0:
            logger.print(
                'epoch:[{}] fold:[{}] batch:[{}/{}]\t'
                'Loss:{loss.val:.4f}({loss.avg:.4f})\t\t'
                'Frame:{top_frame.val:.3f}({top_frame.avg:.3f})\t'.format(
                    epoch, fold, i, len(train_loader), loss=losses, top_frame=top_frame))

        true_label.to(DEVICE_CPU)
        input_var.to(DEVICE_CPU)
        ponits_var.to(DEVICE_CPU)

    '''----- 计算帧正确率-----'''
    true_num = 0
    with torch.no_grad():
        pred_label_list = torch.cat(pred_label_list, dim=0).cpu()
        true_label_list = torch.cat(true_label_list, dim=0).cpu()
        video_index_list = torch.cat(video_index_list, dim=0)
        length = max(video_index_list) + 1
        for i in range(length):
            index = torch.nonzero(video_index_list == i).squeeze()
            value = torch.gather(pred_label_list, dim=0, index=index)
            label = true_label_list[index[0]]
            if label == np.argmax(np.bincount(value.numpy())):
                true_num += 1
    top_video = true_num / length.float()

    logger.print('epoch:[{}] fold:[{}] train finish\t'
                 'Loss:{loss.avg:.4f}\t\t'
                 'Frame:{top_frame.avg:.3f}\t\t'
                 'Video:{top_video:.3f}\t\t'.format(
        epoch, fold, loss=losses, top_frame=top_frame, top_video=top_video))


def val(val_loader, model, epoch, logger, fold):
    losses = util.AverageMeter()
    top_frame = util.AverageMeter()
    model.eval()

    # [batch] / [batch] / [batch]
    pred_label_list = []
    true_label_list = []
    video_index_list = []
    with torch.no_grad():
        for i, (input_first, input_second, input_third, target, ponits_three, video_index) in enumerate(val_loader):
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
            if label == np.argmax(np.bincount(value.numpy())):
                true_num += 1
        top_video = true_num / length.float()

        logger.print('epoch:[{}] fold:[{}] val finish\t'
                     'Loss:{loss.avg:.4f}\t\t'
                     'Frame:{top_frame.avg:.3f}\t\t'
                     'Video:{top_video:.3f}\t\t'.format(
            epoch, fold, loss=losses, top_frame=top_frame, top_video=top_video))

    return top_frame, true_num, length


if __name__ == '__main__':
    main()

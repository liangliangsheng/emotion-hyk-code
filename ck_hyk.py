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
parser.add_argument('-re', '--resume', type=str, default=True)
parser.add_argument('-b', '--batch_train', type=int, default=8, )
parser.add_argument('-w', '--num_workers', type=int, default=8, )
parser.add_argument('-s', '--save_suffix', type=str, default='test', help='save model file name suffix')
args = parser.parse_args()


def main():
    start_date = util.time_now()
    logger = util.Logger('./log/', 'ck_hyk', start_date, args.aggregate_mode)
    logger.print('The aggregate mode is {:}, learning rate: {:}'.format(args.aggregate_mode, args.learning_rate))
    # 加载checkpoint
    save_path = './model/ck_hyk_' + args.aggregate_mode + '_' + args.save_suffix
    model = networks.resnet18_at(aggregate_mode=args.aggregate_mode)
    best_acc_video = 0
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
    train_loader, val_loader = load.ck_faces_hyk(video_root, video_list, points_name, 10, num_workers, batch_train,
                                                 batch_eval)
    for epoch in range(current_epoch, args.epochs):
        train(train_loader, model, optimizer, epoch, logger, optimizer.param_groups[0]['lr'])
        acc_frame, acc_video = val(val_loader, model, epoch, logger)

        if acc_video > best_acc_video:
            logger.print('better acc video')
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'accuracy_frame': acc_frame,
                'accuracy_video': acc_video,
            }, save_path=save_path)
            best_acc_video = acc_video

        lr_scheduler.step()


def train(train_loader, model, optimizer, epoch, logger, learning_rate):
    logger.print(
        '---------------------- epoch:{} train begin learning_rate:{} --------------------------'.format(epoch,
                                                                                                         learning_rate))

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

        # 每200个batch 记录损失和帧正确率
        if i % 200 == 0:
            logger.print(
                '[{}] batch:[{}/{}]\t'
                'Loss:{loss.val:.4f}({loss.avg:.4f})\t'
                'Frame:{top_frame.val:.3f}({top_frame.avg:.3f})\t'.format(
                    epoch, i, len(train_loader), loss=losses, top_frame=top_frame))

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

    logger.print('[{}] train finish\t'
                 'Loss:{loss.avg:.4f}\t'
                 'Frame:{top_frame.avg:.3f}\t'
                 'Video:{top_video:.3f}\t'.format(
        epoch, loss=losses, top_frame=top_frame, top_video=top_video))


def val(val_loader, model, epoch, logger):
    logger.print('---------------------- epoch{epoch} val begin --------------------------'.format(epoch=epoch))

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

        logger.print('[{}] val finish\t'
                     'Loss:{loss.avg:.4f}\t'
                     'Frame:{top_frame.avg:.3f}\t'
                     'Video:{top_video:.3f}\t'.format(
            epoch, loss=losses, top_frame=top_frame, top_video=top_video))

    return top_frame.avg, top_video


# top_video = util.AverageMeter()
#
#     # switch to evaluate mode
#     model.eval()
#     output_store_fc = []
#     output_alpha = []
#     target_store = []
#     index_vector = []
#     with torch.no_grad():
#         for i, (input_var, target, video_index) in enumerate(val_loader):
#             # compute output
#             target = target.to(DEVICE_CUDA)
#             input_var = input_var.to(DEVICE_CUDA)
#             ''' model & full_model'''
#             f, alphas = model(input_var, phrase='eval')
#
#             output_store_fc.append(f)
#             output_alpha.append(alphas)
#             target_store.append(target)
#             index_vector.append(video_index)
#
#         index_vector = torch.cat(index_vector, dim=0)  # [256] ... [256]  --->  [21570]
#         index_matrix = []
#         for i in range(int(max(index_vector)) + 1):
#             index_matrix.append(index_vector == i)
#
#         index_matrix = torch.stack(index_matrix, dim=0).to(DEVICE_CUDA).float()  # [21570]  --->  [380, 21570]
#         output_store_fc = torch.cat(output_store_fc, dim=0)  # [256,7] ... [256,7]  --->  [21570, 7]
#         output_alpha = torch.cat(output_alpha, dim=0)  # [256,1] ... [256,1]  --->  [21570, 1]
#         target_store = torch.cat(target_store, dim=0).float()  # [256] ... [256]  --->  [21570]
#         ''' keywords: mean_fc ; weight_sourcefc; sum_alpha; weightmean_sourcefc '''
#         weight_sourcefc = output_store_fc.mul(output_alpha)  # [21570,512] * [21570,1] --->[21570,512]
#         sum_alpha = index_matrix.mm(output_alpha)  # [380,21570] * [21570,1] -> [380,1]
#         weightmean_sourcefc = index_matrix.mm(weight_sourcefc).div(sum_alpha)
#         target_vector = index_matrix.mm(target_store.unsqueeze(1)).squeeze(1).div(
#             index_matrix.sum(1)).long()  # [380,21570] * [21570,1] -> [380,1] / sum([21570,1]) -> [380]
#         if at_type == 'self-attention':
#             pred_score = model(vm=weightmean_sourcefc, phrase='eval', AT_level='pred')
#         if at_type == 'self_relation-attention':
#             pred_score = model(vectors=output_store_fc, vm=weightmean_sourcefc, alphas_from1=output_alpha,
#                                index_matrix=index_matrix, phrase='eval', AT_level='second_level')
#
#         acc_video = util.accuracy(pred_score.cpu(), target_vector.cpu(), topk=(1,))
#         top_video.update(acc_video[0], i + 1)
#         logger.print(' *Acc@Video {top_video.avg:.3f} '.format(top_video=top_video))
#
#         return top_video.avg


if __name__ == '__main__':
    main()

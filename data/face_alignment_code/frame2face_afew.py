#  coding:utf-8
import os
from data.face_alignment_code.face_align_cuda_afew import face_align_cuda


def not_thread_main(frame_dir, face_dir, less_ten):
    less_ten_video = []
    for category in os.listdir(frame_dir):
        category_dir = os.path.join(frame_dir, category)
        for frame_file in os.listdir(category_dir):
            # if frame_file == '000046280':
            #     continue
            frame_root_folder = os.path.join(category_dir, frame_file)
            face_root_folder = frame_root_folder.replace(frame_dir, face_dir)
            if os.path.isdir(frame_root_folder):
                makefile(face_root_folder)
            print('deal {:}/{:}'.format(category, frame_file))
            result = face_align_cuda(frame_root_folder, face_root_folder, False)
            if result is not None:
                less_ten_video.append(result)
    with open(less_ten, 'w') as f:
        for item, length in less_ten_video:
            f.write(item + ' ' + str(length))
            f.write('\n')


def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


if __name__ == '__main__':
    frame_dir_train_afew = '../frame/train_afew'
    face_dir_train_afew = '../face/train_afew'
    frame_dir_val_afew = '../frame/val_afew'
    face_dir_val_afew = '../face/val_afew'
    train_afew_less_ten = '../face/train_afew/less_ten_txt'
    # main(frame_dir_train_afew, face_dir_train_afew, n_thread=20)
    # main(frame_dir_val_afew, face_dir_val_afew, n_thread=20)
    not_thread_main(frame_dir_train_afew, face_dir_train_afew, train_afew_less_ten)

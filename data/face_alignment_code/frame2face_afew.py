#  coding:utf-8
import os
import threading
from data.face_alignment_code.face_align_cuda_afew import face_align_cuda


def main(frame_dir, face_dir, n_thread):
    threads = []
    # function
    func_path = 'face_align_cuda_afew.py'
    # Model
    for category in os.listdir(frame_dir):
        category_dir = os.path.join(frame_dir, category)
        for frame_file in os.listdir(category_dir):
            frame_root_folder = os.path.join(category_dir, frame_file)
            face_root_folder = frame_root_folder.replace(frame_dir, face_dir)

            if os.path.isdir(frame_root_folder):
                makefile(face_root_folder)
                threads.append(threadFun(frame2face, (frame_root_folder, face_root_folder)))
        # break
    # threads = threads[1:2]
    # 测试时只对一个文件夹进行检测
    run_threads(threads, n_thread)
    print('all is over')


def makefile(file_dir):
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)


def run_threads(threads, n_thread):
    used_thread = []
    for num, new_thread in enumerate(threads):
        print('thread index: {:}'.format(num), end=' \t')
        new_thread.start()
        used_thread.append(new_thread)

        if num % n_thread == 0:
            for old_thread in used_thread:
                old_thread.join()
            used_thread = []


class threadFun(threading.Thread):
    def __init__(self, func, args):
        super(threadFun, self).__init__()
        self.fun = func
        self.args = args

    def run(self):
        self.fun(*self.args)


def frame2face(frame_root_folder, face_root_folder):
    # linux_command = 'python {:} {:} {:} {:} {:} {:}'.format(
    #     func_path, predictor_path, image_root_folder, save_root_folder, cnn_face_detector, gpu_id)
    # print('{:}'.format(image_root_folder))
    # subprocess.getstatusoutput(linux_command)
    face_align_cuda(frame_root_folder, face_root_folder)


if __name__ == '__main__':
    frame_dir_train_afew = '../frame/train_afew'
    face_dir_train_afew = '../face/train_afew'
    frame_dir_val_afew = '../frame/val_afew'
    face_dir_val_afew = '../face/val_afew'

    main(frame_dir_train_afew, face_dir_train_afew, n_thread=20)
    # main(frame_dir_val_afew, face_dir_val_afew, n_thread=20)

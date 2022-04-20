# coding=utf-8
import os
import dlib
import cv2
import numpy as np
from imutils import face_utils
from data.face_alignment_code.pts68_n import change68to16

detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor("./lib/shape_predictor_68_face_landmarks.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1('./lib/mmod_human_face_detector.dat')
predictor5 = dlib.shape_predictor("./lib/shape_predictor_5_face_landmarks.dat")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif',
]


def face_align_cuda(frame_root_folder, face_root_folder, cnn_flag=True):
    face_landmarks_list = []
    image_list = []
    landmark68_path = os.path.join(face_root_folder, '68pts.list')
    landmark16_path = os.path.join(face_root_folder, '16pts.list')
    for name in os.listdir(frame_root_folder):
        img_path = os.path.join(frame_root_folder, name)
        save_path = img_path.replace(frame_root_folder, face_root_folder)
        # 人脸对齐
        result = face_detector(img_path, False, 0.4)
        if result is not None:
            image, landmark = result
            # 存图片
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            image_list.append((save_path, image))
            # 存关键点
            face_landmarks_str = ''
            for item in landmark:
                face_landmarks_str += ' ' + str(item)
            face_landmarks_list.append(face_landmarks_str)

    # 小于10张 重新用cnn处理
    if cnn_flag:
        if len(face_landmarks_list) < 8:
            face_landmarks_list = []
            image_list = []

            dir_image_list = os.listdir(frame_root_folder)
            dir_image_diff = len(dir_image_list) // 8
            for (i, name) in enumerate(dir_image_list):
                if i % dir_image_diff != 0:
                    continue
                img_path = os.path.join(frame_root_folder, name)
                save_path = img_path.replace(frame_root_folder, face_root_folder)
                # 人脸对齐
                result = face_detector(img_path, True, 0.6)
                if result is not None:
                    image, landmark = result
                    # 存图片
                    if not os.path.exists(os.path.dirname(save_path)):
                        os.makedirs(os.path.dirname(save_path))
                    image_list.append((save_path, image))
                    # 存关键点
                    face_landmarks_str = ''
                    for item in landmark:
                        face_landmarks_str += ' ' + str(item)
                    face_landmarks_list.append(face_landmarks_str)

    # 保存图片
    for (save_path, face) in image_list:
        cv2.imwrite(save_path, face)
    # 保存68pts
    with open(landmark68_path, 'w') as fp:
        for face_landmarks_str in face_landmarks_list:
            fp.write(face_landmarks_str)
            fp.write('\n')
    # 保存16pts
    change68to16(landmark68_path, landmark16_path)
    if len(face_landmarks_list) < 10:
        return frame_root_folder[20:],len(face_landmarks_list)
    return None


def face_detector(face_file_path, cnn_flag=False, padding=0.5):
    # Load the image using OpenCV
    bgr_img = cv2.imread(face_file_path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # 人脸对齐
    dets = detector(img, 1)
    if len(dets) == 0:
        if cnn_flag:
            dets = apply_cnn_detection(img)
            if len(dets) == 0:
                return None
        else:
            return None

    faces = dlib.full_object_detections()
    faces.append(predictor68(img, dets[0]))

    image_rgb1 = dlib.get_face_chip(img, faces[0], size=224, padding=padding - 0.05)
    image_rgb2 = dlib.get_face_chip(img, faces[0], size=224, padding=padding)
    image_rgb3 = dlib.get_face_chip(img, faces[0], size=224, padding=padding + 0.05)
    image_rgb = image_rgb1

    # landmark
    dets = detector(image_rgb, 1)
    if len(dets) == 0:
        image_rgb = image_rgb2
        dets = detector(image_rgb, 1)
        if len(dets) == 0:
            image_rgb = image_rgb3
            dets = detector(image_rgb, 1)
            if len(dets) == 0:
                if cnn_flag:
                    dets = apply_cnn_detection(image_rgb)
                    if len(dets) == 0:
                        return None
                else:
                    return None

    face_landmarks = face_utils.shape_to_np(predictor68(image_rgb, dets[0])).flatten().tolist()
    if len(face_landmarks) != 136:
        return None

    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), face_landmarks


def LinearEqual(image):
    lut = np.zeros(256, dtype=image.dtype)
    hist = cv2.calcHist([image], [0], None, [256], [0.0, 255.0])
    minBinNo, maxBinNo = 0, 255

    for binNo, binValue in enumerate(hist):
        if binValue != 0:
            minBinNo = binNo
            break
    for binNo, binValue in enumerate(reversed(hist)):
        if binValue != 0:
            maxBinNo = 255 - binNo
            break
    for i, v in enumerate(lut):
        if i < minBinNo:
            lut[i] = 0
        elif i > maxBinNo:
            lut[i] = 255
        else:
            lut[i] = int(255.0 * (i - minBinNo) / (maxBinNo - minBinNo) + 0.5)  # why plus 0.5
    return cv2.LUT(image, lut)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def apply_cnn_detection(img):
    cnn_dets = cnn_face_detector(img, 1)
    dets = dlib.rectangles()
    dets.extend([d.rect for d in cnn_dets])
    return dets

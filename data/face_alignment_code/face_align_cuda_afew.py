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


def face_align_cuda(frame_root_folder, face_root_folder):
    face_landmarks_list = []
    landmark68_path = os.path.join(face_root_folder, '68pts.list')
    landmark16_path = os.path.join(face_root_folder, '16pts.list')
    for root, dirs, files in os.walk(frame_root_folder, followlinks=False):
        for name in files:
            if is_image_file(name):
                img_path = os.path.join(root, name)
                save_path = img_path.replace(frame_root_folder, face_root_folder)
                # 人脸对齐
                image_rgb = face_alignment(img_path)
                if image_rgb is not None:
                    # 关键点检测
                    face_landmarks = face_landmark(image_rgb)
                    if face_landmarks is not None:
                        if len(face_landmarks) == 136:
                            # 存图片
                            if not os.path.exists(os.path.dirname(save_path)):
                                os.makedirs(os.path.dirname(save_path))
                            cv2.imwrite(save_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
                            # 存关键点
                            face_landmarks_str = ''
                            for item in face_landmarks:
                                face_landmarks_str += ' ' + str(item)
                            face_landmarks_list.append(face_landmarks_str)
                        else:
                            print(save_path + 'landmark error')

    # 保存68pts
    with open(landmark68_path, 'w') as fp:
        for face_landmarks_str in face_landmarks_list:
            fp.write(face_landmarks_str)
            fp.write('\n')
    # 保存16pts
    change68to16(landmark68_path, landmark16_path)
    # check16(landmark16_path)


def face_alignment(face_file_path):
    # Load the image using OpenCV
    bgr_img = cv2.imread(face_file_path)
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    ''' traditional method '''
    dets = detector(img, 1)
    if len(dets) == 0:
        # first use cnn detector
        dets = apply_cnn_detection(img)
        if len(dets) == 0:
            ''' Linear '''
            img = LinearEqual(img)
            dets = apply_cnn_detection(img)
            if len(dets) == 0:
                return None

    faces = dlib.full_object_detections()
    faces.append(predictor5(img, dets[0]))
    # faces.append(predictor68(img, dets[0]))
    image_rgb = dlib.get_face_chip(img, faces[0], size=224, padding=0.4)
    return image_rgb


def face_landmark(image_rgb):
    dets = detector(image_rgb, 1)
    if len(dets) == 0:
        # first use cnn detector
        dets = apply_cnn_detection(image_rgb)
        # if len(dets) == 0:
        #     ''' Linear '''
        #     img = LinearEqual(image_rgb)
        #     dets = apply_cnn_detection(img)
        if len(dets) == 0:
            return None

    face_landmarks = face_utils.shape_to_np(predictor68(image_rgb, dets[0])).flatten().tolist()
    return face_landmarks


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

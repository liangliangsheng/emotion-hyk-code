# coding=utf-8
import os
import dlib
import cv2
import numpy as np
from imutils import face_utils
from data.face_alignment_code.pts68_n import change68to16, check16

detector = dlib.get_frontal_face_detector()
predictor68 = dlib.shape_predictor("./lib/shape_predictor_68_face_landmarks.dat")
cnn_face_detector = dlib.cnn_face_detection_model_v1('./lib/mmod_human_face_detector.dat')
predictor5 = dlib.shape_predictor("./lib/shape_predictor_5_face_landmarks.dat")

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.gif',
]


def face_align_cuda(frame_root_folder, face_root_folder, flag_landmark=True, flag_align=True):
    face_landmarks_list = []
    landmark68_path = os.path.join(face_root_folder, '68pts.list')
    landmark16_path = os.path.join(face_root_folder, '16pts.list')
    for root, dirs, files in os.walk(frame_root_folder, followlinks=False):
        for name in files:
            if is_image_file(name):
                img_path = os.path.join(root, name)
                save_path = img_path.replace(frame_root_folder, face_root_folder)
                # 人脸对齐
                if flag_align:
                    face = face_alignment(img_path)
                    if face is not None:
                        if not os.path.exists(os.path.dirname(save_path)):
                            os.makedirs(os.path.dirname(save_path))
                        cv2.imwrite(save_path, face)

                # landmark
                if flag_landmark:
                    image = cv2.imread(save_path)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = detector(gray)
                    for face in faces:
                        face_landmarks = predictor68(gray, face)
                        face_landmarks = face_utils.shape_to_np(face_landmarks).flatten().tolist()
                        face_landmarks_str = ''
                        for item in face_landmarks:
                            face_landmarks_str += ' ' + str(item)
                        face_landmarks_list.append(face_landmarks_str)

    if flag_landmark:
        # 保存68pts
        with open(landmark68_path, 'w') as fp:
            for face_landmarks_str in face_landmarks_list:
                fp.write(face_landmarks_str)
                fp.write('\n')
    # 保存26pts
    # change68to16(landmark68_path, landmark16_path)
    check16(landmark16_path)


def face_alignment(face_file_path):
    # Load the image using OpenCV
    bgr_img = cv2.imread(face_file_path)
    if bgr_img is None:
        print("Sorry, we could not load '{}' as an image".format(face_file_path))
        exit()

    # Convert to RGB since dlib uses RGB images
    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
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
                ''' clahe '''
                img = claheColor(img)
                dets = apply_cnn_detection(img)
                if len(dets) == 0:
                    # ''' Histogram_equalization '''
                    img = hisEqulColor(img)
                    dets = apply_cnn_detection(img)
                    if len(dets) == 0:
                        return None

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()

    for detection in dets:
        faces.append(predictor5(img, detection))
    image = dlib.get_face_chip(img, faces[0], size=224, padding=0.25)
    cv_bgr_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return cv_bgr_img


def claheColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.predictor5lit(ycrcb)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe.apply(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def hisEqulColor(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.predictor5lit(ycrcb)
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


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

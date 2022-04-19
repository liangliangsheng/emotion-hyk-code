import matplotlib.pyplot as plt
import os
import cv2


def show(dir_path, width):
    image_list = []
    landmark_list = []
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if '.list' not in file_name:
            image_list.append(cv2.imread(file_path))

    with open(os.path.join(dir_path, '68pts.list')) as fp:
        for line in fp:
            landmark_list.append(line.strip().split(' '))
    for (i, line) in enumerate(landmark_list):
        image = image_list[i]
        for n in range(0, 68):
            x = int(line[n * 2])
            y = int(line[n * 2 + 1])
            cv2.circle(image, (x, y), 1, (0, 255, 0), 1)
        cv2.imshow("Face Landmarks", image)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    path = '../data/face/train_afew/Angry/000223480'
    show(path, 10)

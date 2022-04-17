import cv2
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")


def landmark68(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        face_landmarks = predictor(gray, face)
        face_landmarks = face_utils.shape_to_np(face_landmarks).flatten().tolist()
        # for n in range(0, 68):
        #     x = face_landmarks.part(n).x
        #     y = face_landmarks.part(n).y
        #     cv2.circle(image, (x, y), 1, (0, 255, 255), 1)
    cv2.imshow("Face Landmarks", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


def show(image_path, landmark_path, n):
    landmark_list = []
    with open(landmark_path) as fp:
        for line in fp:
            landmark_list.append(line.strip().split(' '))
            break
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for n in range(0, n):
        x = int(landmark_list[0][n * 2])
        y = int(landmark_list[0][n * 2 + 1])
        cv2.circle(image, (x, y), 1, (0, 0, 0), 1)
    cv2.imshow("Face Landmarks", image)
    key = cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    image_path = './S130_007_00000002.png'
    landmark_path = './16pts.list'
    show(image_path, landmark_path, 16)

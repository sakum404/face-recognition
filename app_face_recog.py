import cv2
import numpy as np
from PIL import Image
import os
import csv
from datetime import datetime
import time

def check_password():
    expected_password = "sakum"  # Замените на ваш реальный пароль
    entered_password = input("Введите пароль: ")

    if entered_password == expected_password:
        print("Пароль верный. Запуск программы.")
        return True
    else:
        print("Неверный пароль. Программа завершена.")
        return False


def capture_faces():
    cascade_path = 'haarcascade_frontalface_default.xml'

    clf = cv2.CascadeClassifier(cascade_path)
    cam = cv2.VideoCapture(0)

    # For each person, enter one numeric face id
    face_id = input('\n Введите ID лица  ==>  ')

    print("\n [INFO] Идентификация ...")
    # Initialize individual sampling face count
    count = 0

    while (True):

        ret, frame = cam.read()
        # img = cv2.flip(img, -1)  # flip video image vertically
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            count += 1

            # Save the captured image into the datasets folder
            cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y + h, x:x + w])

            cv2.imshow('image', frame)

        k = cv2.waitKey(100) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break
        elif count >= 30:  # Take 30 face sample and stop video
            break

    # Do a bit of cleanup
    print("\n [INFO] Закрытие")
    cam.release()
    cv2.destroyAllWindows()


def train_recognizer(path='dataset'):
    path = 'dataset'

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

    # function to get the images and label data
    def getImagesAndLabels(path):

        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        ids = []

        for imagePath in imagePaths:

            PIL_img = Image.open(imagePath).convert('L')  # convert it to grayscale
            img_numpy = np.array(PIL_img, 'uint8')

            id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = detector.detectMultiScale(img_numpy)

            for (x, y, w, h) in faces:
                faceSamples.append(img_numpy[y:y + h, x:x + w])
                ids.append(id)

        return faceSamples, ids

    print("\n [INFO] Загрузка ...")
    faces, ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))

    # Save the model into trainer/trainer.yml
    recognizer.write('trainer/trainer.yml')  # recognizer.save() worked on Mac, but not on Pi

    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Закрытие".format(len(np.unique(ids))))


def log_face_capture(face_id, guest=False):
    log_file = 'face_log.csv' if not guest else "guest_log.csv"
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Проверяем, был ли пользователь уже записан в файл
    with open(log_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            if row and row[1] == str(face_id):
                return

    # Если пользователь не был найден в файле, добавляем информацию
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([current_time, face_id])
    print(f"Пользователь {face_id} успешно записан в файл.")

def recognize_faces():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('trainer/trainer.yml')
    cascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath);

    font = cv2.FONT_HERSHEY_SIMPLEX

    # iniciate id counter
    id = 0

    # names related to ids: example ==> Marcelo: id=1,  etc
    names = ['None', 'Muhammed', 'Nazerke', 'Ilza', 'Z', 'W']

    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    while True:

        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

            # Check if confidence is less than 100 ==> "0" is a perfect match
            if (confidence < 100):
                id = names[id]
                confidence = "  {0}%".format(round(100 - confidence))
                log_face_capture(id)  # Добавьте логирование при идентификации
            else:
                id = "задержать"
                confidence = "  {0}%".format(round(100 - confidence))

            cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
            cv2.putText(img, str(confidence), (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

        cv2.imshow('camera', img)

        k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
        if k == 27:
            break

    # Do a bit of cleanup
    print("\n [INFO] Закрытие")
    cam.release()
    cv2.destroyAllWindows()

def remove_user_images(user_id):
    path = 'dataset'
    images_to_remove = [f for f in os.listdir(path) if f.startswith(f"User.{user_id}.")]

    for image_name in images_to_remove:
        image_path = os.path.join(path, image_name)
        os.remove(image_path)

    print(f"Removed {len(images_to_remove)} images for User {user_id}")

def main():
    cascade_path = 'haarcascade_frontalface_default.xml'
    clf = cv2.CascadeClassifier(cascade_path)

    while True:
        print("\n1. Захватить лица\n2. Обновить данные\n3. Распознать лица\n4. Удалить пользователя\n5. Выход")
        choice = input("Введите номер выбора: ")

        if choice == '1':
            capture_faces()
        elif choice == '2':
            train_recognizer()
        elif choice == '3':
            recognize_faces()
        elif choice == '4':
            user_id_to_remove = input("Введите идентификатор пользователя для удаления: ")
            remove_user_images(user_id_to_remove)
        elif choice == '5':
            break
        else:
            print("Неверный выбор. Пожалуйста, введите допустимую опцию.")

if __name__ == "__main__":
    if check_password():
        start_time = time.time()

        while True:
            elapsed_time = time.time() - start_time

            if elapsed_time >= 60: # Проверка каждую минуту
                log_face_capture('None', guest=True) # Запись неидентифицированного пользователя в guest.csv
                start_time = time.time()

            main()
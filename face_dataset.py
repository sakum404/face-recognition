import cv2
import os


def face_capture():
    cascade_path = 'haarcascade_frontalface_default.xml'

    clf = cv2.CascadeClassifier(cascade_path)
    cam = cv2.VideoCapture(0)

    # For each person, enter one numeric face id
    face_id = input('\n enter user id end press <return> ==>  ')

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


def main():
    face_capture()


if __name__ == '__main__':
    main()
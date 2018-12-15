import cv2

# print(cv2.__version__)

imagePath = "abba.png"
cascPath = "HaarCascadeFace.xml"

# opencv cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# live webcam feed
video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    # gray
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the image - detectMultiScale detects objects & returns array
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(30, 30),
        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
    )

    # number of faces detected from cascade
    print "Found {0} faces!".format(len(faces))

    # draws a rectangle around faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # displays face with rectangle
    cv2.imshow('Video', frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# end
video_capture.release()
cv2.destroyAllWindows()

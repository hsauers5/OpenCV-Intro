import cv2

# print(cv2.__version__)

imagePath = "mycats.jpg"
cascPath = "CatCascade.xml"

# opencv cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# reads image from image file
image = cv2.imread(imagePath)

# gray
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect faces in the image - detectMultiScale detects objects & returns array
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=15,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# number of faces detected from cascade
print "Found {0} cats!".format(len(faces))

# draws a rectangle around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# shows faces
cv2.imshow("Cats found", image)
cv2.waitKey(0)

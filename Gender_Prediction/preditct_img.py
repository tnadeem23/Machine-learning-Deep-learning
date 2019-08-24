import cv2
import time
import numpy as np

image = cv2.imread('predictimg1.jpg')

mean_val = (78.4263377603, 87.7689143744, 114.895847746)
genders = ['Male', 'Female']

def build_model():

	gender_model = cv2.dnn.readNetFromCaffe(
		'data/gender.prototxt',
		'data/gender.caffemodel')

	return(gender_model)

def predict_gender(gender_model):
	font = cv2.FONT_HERSHEY_SIMPLEX

	face_cascade = cv2.CascadeClassifier('data/haarcascade_face.xml')

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	if(len(faces)>0):
		print("Found {} faces".format(str(len(faces))))

	for (x, y, w, h )in faces:
		cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

		# Get Face
		face_img = image[y:y+h, h:h+w].copy()
		blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), mean_val, swapRB=False)

		#Predict Gender
		gender_model.setInput(blob)
		prediction = gender_model.forward()
		output = genders[prediction[0].argmax()]
		print("Gender : " + output)


		overlay_text = "%s" % (output)
		cv2.putText(image, overlay_text, (x, y), font, 2, (255, 0, 0), 3, cv2.LINE_AA)


	cv2.imwrite("Output_image.jpg", image)
	im = cv2.imread('Output_image.jpg')
	imS = cv2.resize(im, (1024, 768))
	cv2.imshow("output", imS)
	cv2.waitKey()

if __name__ == "__main__":
	gender_model = build_model()

	predict_gender(gender_model)
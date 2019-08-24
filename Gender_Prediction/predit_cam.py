import cv2
import time
import numpy as np

vid = cv2.VideoCapture(0)

vid.set(3, 480)
vid.set(4, 640)

mean_val = (78.4263377603, 87.7689143744, 114.895847746)
genders = ['Male', 'Female']

def build_model():

	gender_model = cv2.dnn.readNetFromCaffe(
		'data/gender.prototxt',
		'data/gender.caffemodel')

	return(gender_model)

def predict_gender(gender_model):
	font = cv2.FONT_HERSHEY_SIMPLEX

	while True:

		ret, image = vid.read()
		cv2.imwrite("Output_image1.jpg", image)

		face_cascade = cv2.CascadeClassifier('data/haarcascade_face.xml')

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)

		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h )in faces:
			cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)


			face_img = image[y:y+h, h:h+w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), mean_val, swapRB=False)


			gender_model.setInput(blob)
			prediction = gender_model.forward()
			output = genders[prediction[0].argmax()]
			print("Gender : " + output)


			text_opt = "%s" % (output)
			cv2.putText(image, text_opt, (x, y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)


		cv2.imshow('Output', image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	gender_model = build_model()

	predict_gender(gender_model)
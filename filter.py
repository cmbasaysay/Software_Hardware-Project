#import necessary packages
import face_recognition
import numpy as np
import cv2

#this sets a variable for the image filter
elf = cv2.imread('ck4.png',-1) 

#this opens the camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 30)

#this is responsible in overlaying the filter to the face detected in the video
class Filter:

	def overlay(self, frame, elf, pos=(0,0), scale = 1): 

		elf = cv2.resize(elf, (0,0), fx=scale, fy=scale)
		h, w, _ = elf.shape
		rows, cols, _ = frame.shape
		y, x = pos[0], pos[1]

		for i in range(h):
			for j in range(w):
				if x + i >= rows or y + j >= cols:
					continue

				alpha = float(elf[i][j][3] / 255.0)
				frame[x + i][y + j] = alpha * elf[i][j][:3] + (1 - alpha) * frame[x + i][y + j]

		return frame

while True:

	filter_ops = Filter()

	ret, image = cap.read()

    #this will detect facial landmarks from the video
	image_frame = image[:, :, ::-1]
	detected_face = face_recognition.face_locations(image_frame)
	
	#initialize coordinates
	faces_detected = [(0,0,0,0)]

    #if a face is detected, the following will be executed
	if detected_face != []:

		faces_detected = [[detected_face[0][3], detected_face[0][0], abs(detected_face[0][3] - detected_face[0][1]) + 150, abs(detected_face[0][0] - detected_face[0][2])]]

		for (x, y, w, h) in faces_detected:
            
            #this adjusts the coordinates of the filter
			x -= 60
			w -= 40
			y -= 70			
			h -= 50
            
			elf_symin = int(y - 3 * h / 5)
			elf_symax = int(y + 8 * h / 5)

			sh_elf = elf_symax - elf_symin

			elf_sxmin = int(x - 3 * w / 5)
			elf_sxmax = int(x * w / 5)

			sw_elf = elf_sxmax - elf_sxmin

			face_frame = image[elf_symin:elf_symax, x:x+w]
			
			#the filter is resized to fit the face
			elf_resized= cv2.resize(elf, (w, sh_elf),interpolation=cv2.INTER_CUBIC)

            #function is called
			filter_ops.overlay(face_frame, elf_resized)

    #video stream is shown
	cv2.imshow('I am an elf!', image)

    #if s is pressed, the application will stop
	if cv2.waitKey(30) & 0xFF == ord("s"):
		break
		
#cleanup
cap.release()
cv2.destroyAllWindows()
	


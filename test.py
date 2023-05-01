import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0);

sahil_image = face_recognition.load_image_file("photos/sahil.jpg")
sahil_encoding = face_recognition.face_encodings(sahil_image)[0]

aman_image = face_recognition.load_image_file("photos/aman.jpg")
aman_encoding = face_recognition.face_encodings(aman_image)[0]

shreyansh_image = face_recognition.load_image_file("photos/shreyansh.jpg")
shreyansh_encoding = face_recognition.face_encodings(shreyansh_image)[0]

amit_rathi_sir_image = face_recognition.load_image_file("photos/amit.jpg")
amit_rathi_sir_encoding = face_recognition.face_encodings(amit_rathi_sir_image)[0]

navaljeet_sir_image = face_recognition.load_image_file("photos/navaljeet.jpg")
navaljeet_sir_encoding = face_recognition.face_encodings(navaljeet_sir_image)[0]

nkjain_sir_image = face_recognition.load_image_file("photos/nkjain.jpg")
nkjain_sir_encoding = face_recognition.face_encodings(nkjain_sir_image)[0]



known_face_encoding = [

	sahil_encoding,
	shreyansh_encoding,
	aman_encoding,
	amit_rathi_sir_encoding,
	navaljeet_sir_encoding,
	nkjain_sir_encoding

]

known_face_names = [
	"Sahil Gupta",
	"Shreyansh Jaiswal",
	"Aman Tripathi",
	"Dr. Amit Rathi",
	"Dr. Navaljeet Singh Arora",
	"Dr. Neelesh Kumar Jain"
]

students = known_face_names.copy()

face_locations = []
face_encodings = []
face_names = []
s=True


now = datetime.now()
current_date = now.strftime("%Y-%m-%d")



f = open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame = video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("Attendance System",frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
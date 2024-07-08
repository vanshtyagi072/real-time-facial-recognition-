import cv2
import face_recognition
known_face_encodings = []
known_face_names = []
image_paths=[]
for i in range(1,15):
    a = str(i)
    p = f"C://Users//Admin//Desktop//vansh intern 2.0//image dataset//person{a}.jpg"
    image_paths.append(p)
    print(image_paths)
    pass

names = []

# Read data from the file and populate the list
with open("dataset.txt", "r") as file:
    for line in file:
        # Assuming each line contains "personX = name", where X is the number
        person_data = line.strip().split(" = ")
        if len(person_data) == 2:  # Ensure the line is formatted correctly
            names.append(person_data[1])

# Print the list of person names
print(names)  
    
    
    
#adding comment

#names = ["vansh", "govind", "narendra"]

# Lists to store the loaded encodings and names
known_face_encodings = []
known_face_names = []

# Load images and encodings in a loop
for path, name in zip(image_paths, names):
    # Load image
    image = face_recognition.load_image_file(path)
    
    # Compute face encoding
    encoding = face_recognition.face_encodings(image)[0]
    
    # Append encoding and name to their respective lists
    known_face_encodings.append(encoding)
    known_face_names.append(name)



video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame,face_locations)

    for(top,right,bottom,left), face_encodings in zip(face_locations,face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings,face_encodings)
        name = "unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left,top),(right,bottom),(0,0,255),2)
        cv2.putText(frame,name,(left,top-10),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
    cv2.imshow("video",frame)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break 


video_capture.release() 
cv2.destroyAllWindows()           

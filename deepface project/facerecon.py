import os
import cv2
import numpy as np
import pyttsx3
import datetime
from deepface import DeepFace

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Directory to store face dataset
data_dir = "face_dataset"
os.makedirs(data_dir, exist_ok=True)

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Step 1: Create Face Dataset
def create_face_dataset(name):
    person_dir = os.path.join(data_dir, name)
    os.makedirs(person_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    print("Capturing images. Press 'q' to stop.")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            count += 1
            face_img = frame[y:y + h, x:x + w]
            face_path = os.path.join(person_dir, f"{name}_{count}.jpg")
            cv2.imwrite(face_path, face_img)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Capture Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Saved {count} images for {name}.")
    speak(f"Face data for {name} captured successfully.")

# Step 2: Train Face Dataset
def train_face_dataset():
    embeddings = {}
    for person in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person)
        if os.path.isdir(person_dir):
            embeddings[person] = []
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                try:
                    embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embeddings[person].append(embedding)
                except Exception as e:
                    print(f"Failed to process {img_path}: {e}")
    return embeddings

# Step 3: Recognize Faces
def recognize_faces(embeddings):
    cap = cv2.VideoCapture(0)
    last_emotion = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error!")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)
        face_count = len(faces)
        
        for (x, y, w, h) in faces:
            face_img = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            try:
                analysis = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                if isinstance(analysis, list):
                    analysis = analysis[0]

                age = analysis["age"]
                gender = max(analysis["gender"], key=analysis["gender"].get)
                emotion = max(analysis["emotion"], key=analysis["emotion"].get)

                if emotion != last_emotion:
                    last_emotion = emotion
                    face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    
                    match = "Unknown"
                    max_similarity = -1

                    for person, person_embeddings in embeddings.items():
                        for embed in person_embeddings:
                            similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                            if similarity > max_similarity:
                                max_similarity = similarity
                                match = person

                    label = f"{match} ({max_similarity:.2f})" if max_similarity > 0.75 else "Unknown"
                    display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"
                    speak(f"{label}, age {int(age)}, gender {gender}, emotion {emotion}")

                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception as e:
                print(f"Error recognizing face: {e}")
        
        # Display live time and face count
        current_time = datetime.datetime.now().strftime("%H:%M:%S")
        cv2.putText(frame, f"Time: {current_time}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Faces detected: {face_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Recognize Faces", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Menu Loop
while True:
    print("\n1. Create Face Dataset\n2. Train Face Dataset\n3. Recognize Faces\n4. Exit")
    choice = input("Enter your choice: ")
    
    if choice == "1":
        name = input("Enter the name of the person: ")
        create_face_dataset(name)
    elif choice == "2":
        embeddings = train_face_dataset()
        np.save("embeddings.npy", embeddings)
        print("Embeddings saved.")
    elif choice == "3":
        if os.path.exists("embeddings.npy"):
            embeddings = np.load("embeddings.npy", allow_pickle=True).item()
            recognize_faces(embeddings)
        else:
            print("No trained data found. Train dataset first.")
    elif choice == "4":
        print("Exiting...")
        break
    else:
        print("Invalid choice. Try again.")

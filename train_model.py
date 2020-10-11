from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", help="path to serialized db of facial embeddings",
                default=r'C:\Users\K. SELVAM\Desktop\opencv-face-recognition\output\embeddings.pickle')
ap.add_argument("-r", "--recognizer", help="path to output model trained to recognize faces",
                default=r'C:\Users\K. SELVAM\Desktop\opencv-face-recognition\output\recognizer.pickle')
ap.add_argument("-l", "--le", help="path to output label encoder",
                default=r'C:\Users\K. SELVAM\Desktop\opencv-face-recognition\output\le.pickle')
args = vars(ap.parse_args())

print("[INFO] loading face embeddings...")
data = pickle.loads(open(args["embeddings"], "rb").read())

print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()

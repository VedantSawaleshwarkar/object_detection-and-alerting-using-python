# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle
import numpy as np

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# ensure at least two classes are present for SVC
unique_classes = np.unique(labels)
if unique_classes.shape[0] < 2:
    print("[ERROR] Training requires at least TWO distinct persons/classes in 'dataset/'.")
    print("        Found only one class: ", np.unique(data["names"]))
    print("[HINT] Add images under 'dataset/<another_person>/' (20-30 images), then re-run:")
    print("       1) extract_embeddings.py")
    print("       2) train_model.py")
    raise SystemExit(1)

# L2-normalize embeddings to improve class separation
embeddings = np.asarray(data["embeddings"], dtype=np.float32)
norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
embeddings = embeddings / norms

# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=1.0, kernel="linear", probability=True, class_weight="balanced")
recognizer.fit(embeddings, labels)

# write the actual face recognition model to disk
f = open("output/recognizer", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()
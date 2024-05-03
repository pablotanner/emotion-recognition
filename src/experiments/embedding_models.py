"""
Models Trained with embeddings from DeepFace, Facenet, Facenet512, OpenFace, DeepID, ArcFace, and SFace
"""
from src.data_processing.embedding_loader import EmbeddingLoader
from src.model_training.data_splitter import DataSplitter
from src.model_training.emotion_recognition_models import SVM

embedding_loader = EmbeddingLoader("../../data")


y = embedding_loader.emotions

X_dict = embedding_loader.embeddings

vgg_split = DataSplitter(X_dict["VGG-Face"], y, test_size=0.2)
facenet_split = DataSplitter(X_dict["Facenet"], y, test_size=0.2)
facenet512_split = DataSplitter(X_dict["Facenet512"], y, test_size=0.2)
openface_split = DataSplitter(X_dict["OpenFace"], y, test_size=0.2)
deepface_split = DataSplitter(X_dict["DeepFace"], y, test_size=0.2)
deepid_split = DataSplitter(X_dict["DeepID"], y, test_size=0.2)
arcface_split = DataSplitter(X_dict["ArcFace"], y, test_size=0.2)
sface_split = DataSplitter(X_dict["SFace"], y, test_size=0.2)

X_train_vgg, X_test_vgg, y_train_vgg, y_test_vgg = vgg_split.split_data()
X_train_facenet, X_test_facenet, y_train_facenet, y_test_facenet = facenet_split.split_data()
X_train_facenet512, X_test_facenet512, y_train_facenet512, y_test_facenet512 = facenet512_split.split_data()
X_train_openface, X_test_openface, y_train_openface, y_test_openface = openface_split.split_data()
X_train_deepface, X_test_deepface, y_train_deepface, y_test_deepface = deepface_split.split_data()
X_train_deepid, X_test_deepid, y_train_deepid, y_test_deepid = deepid_split.split_data()
X_train_arcface, X_test_arcface, y_train_arcface, y_test_arcface = arcface_split.split_data()
X_train_sface, X_test_sface, y_train_sface, y_test_sface = sface_split.split_data()

KERNEL = 'rbf'


# SVM
svm_vgg = SVM(kernel=KERNEL)
svm_vgg.train(X_train_vgg, y_train_vgg)
print("VGG-Face")
svm_vgg.evaluate(X_test_vgg, y_test_vgg)

svm_facenet = SVM(kernel=KERNEL)
svm_facenet.train(X_train_facenet, y_train_facenet)
print("Facenet")
svm_facenet.evaluate(X_test_facenet, y_test_facenet)

svm_facenet512 = SVM(kernel=KERNEL)
svm_facenet512.train(X_train_facenet512, y_train_facenet512)
print("Facenet512")
svm_facenet512.evaluate(X_test_facenet512, y_test_facenet512)

svm_openface = SVM(kernel=KERNEL)
svm_openface.train(X_train_openface, y_train_openface)
print("OpenFace")
svm_openface.evaluate(X_test_openface, y_test_openface)

svm_deepface = SVM(kernel=KERNEL)
svm_deepface.train(X_train_deepface, y_train_deepface)
print("DeepFace")
svm_deepface.evaluate(X_test_deepface, y_test_deepface)

svm_deepid = SVM(kernel=KERNEL)
svm_deepid.train(X_train_deepid, y_train_deepid)
print("DeepID")
svm_deepid.evaluate(X_test_deepid, y_test_deepid)

svm_arcface = SVM(kernel=KERNEL)
svm_arcface.train(X_train_arcface, y_train_arcface)
print("ArcFace")
svm_arcface.evaluate(X_test_arcface, y_test_arcface)

svm_sface = SVM(kernel=KERNEL)
svm_sface.train(X_train_sface, y_train_sface)
print("SFace")
svm_sface.evaluate(X_test_sface, y_test_sface)



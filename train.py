import matplotlib
matplotlib.use("Agg")

from helper.learning_rate_scheduler import StepDecay ,PolynomialDecay
from helper.resnet import ResNet
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--schedule", type=str, default="",
	help="learning rate schedule method")
ap.add_argument("-e", "--epochs", type=int, default=100,
	help="# of epochs to train for")
ap.add_argument("-l", "--lr-plot", type=str, default="output/lr.png",
	help="path to output learning rate plot")
ap.add_argument("-t", "--train-plot", type=str, default="output/training.png",
	help="path to output training plot")
args = vars(ap.parse_args())


epochs = args["epochs"]
callbacks = []
schedule = None

if args["schedule"] == "step":
    print("[INFO] using step based learning rate decay")
    schedule = StepDecay(initAlpha = 1e-1 ,factor = 0.25 ,dropEvery = 15)

elif args["schedule"] == "linear":
    print("[INFO] using 'linear' learning rate decay...")
    schedule = PolynomialDecay(maxEpochs = epochs ,initAlpha = 1e-1 ,power = 1)

elif args["schedule"] == "poly":
    print("[INFO] using 'polynomial' learning rate decay...")
    schedule = PolynomialDecay(maxEpochs = epochs ,initAlpha = 1e-1 ,power = 5)

if schedule is not None:
    callbacks = [LearningRateScheduler(schedule)]

print("[INFO] loading CIFAR-10 data...")
((trainX ,trainY) ,(testX ,testY)) = cifar10.load_data()
trainX = trainX.astype("float32") / 255.0
testX = testX.astype("float32") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

labelNames = ["airplane", "automobile", "bird", "cat", "deer",
	"dog", "frog", "horse", "ship", "truck"]

decay = 0.0

if args["schedule"] == "standard":
    print("[INFO] using 'keras standard' learning rate decay...")
    decay = 1e-1 / float(epochs)

elif schedule is None:
	print("[INFO] no learning rate schedule being used")

opt = SGD(learning_rate = 1e-1 ,momentum=0.9 ,decay = decay)
model = ResNet.build(32 ,32 ,3 ,10 ,(9 ,9 ,9) ,(64 ,64 ,128 ,256) ,reg = 0.0005)
model.compile(loss = "categorical_crossentropy" ,optimizer=opt ,metrics = ["accuracy"])
H = model.fit(x = trainX ,y = trainY ,validation_data = (testX ,testY) ,epochs = epochs ,batch_size = 128 ,callbacks = callbacks ,verbose = True)

print("[INFO evaluating network")
prediction = model.predict(testX ,batch_size = 128)
prediction = prediction.argmax(axis = 1)

print(classification_report(testX.argmax(axis = 1) ,prediction ,target_names = labelNames))

N = np.arange(0 ,epochs)
plt.style.use("ggplot")
plt.figure()
plt.plot(N ,H.history["loss"] ,label = "Train Loss")
plt.plot(N ,H.history["val_loss"] ,label = "validation loss")
plt.plot(N ,H.history["accuracy"] ,label = "Train Accuracy")
plt.plot(N ,H.history["val_accuracy"] ,label = "validation accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.title("Training performance")
plt.legend()
plt.savefig(args["train_plot"])

if schedule is not None:
    schedule.plot(epochs = N)
    plt.savefig(args["lr_plot"])

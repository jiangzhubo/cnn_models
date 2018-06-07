
import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.mobilenet import MobileNet
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from keras.layers import Input, Dense, Dropout, Flatten, GlobalAveragePooling2D,Conv2D, MaxPooling2D, Conv2DTranspose, merge,Flatten, concatenate,BatchNormalization,Activation, Dropout,  Dense,UpSampling2D,Input,Concatenate,add
from keras.models import Sequential, Model,load_model
from sklearn.model_selection import train_test_split,KFold
from keras.optimizers import Adam
import tensorflow as tf
import random
import argparse
import cv2
from keras.preprocessing.image import ImageDataGenerator
np.random.seed(7)
import os
from sklearn.metrics import confusion_matrix
import pickle
import matplotlib.pyplot as plt
import itertools
from skimage.color import rgb2gray
from  skimage.transform import resize
from collections import Counter
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D, merge,Flatten, concatenate,BatchNormalization,Activation, Dropout,  Dense,UpSampling2D,Input,Concatenate,add
from keras import backend as K
import keras
import imutils
import peakutils
from time import clock
from skimage.morphology import disk, closing, opening
import datetime
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression as LR

from sklearn.metrics import accuracy_score
def Confusion_Matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
               
def load_data( data_path, img_rows = 224,img_cols = 224): 
	print('data path:', data_path)
	class_names = []
	data = []
	labels = []
	label_index = 0
	data_new = []
	label_new = []

	for label_name in os.listdir(data_path):
        	print(label_name)
		class_names.append(label_name)
        	for image_name in os.listdir(data_path+'/'+label_name):        
           		image = cv2.imread(data_path+'/'+label_name + '/' +image_name)
              		image = cv2.resize(image, (img_rows, img_cols))
     	                data.append(image)
           		labels.append(label_index)
		label_index = label_index + 1 
	length_data = range(len(data))
 #   	random.shuffle(length_data)
#	import pdb;pdb.set_trace()
 #   	data_new = [data[idx] for idx in length_data]
  #  	label_new = [label_name[idx] for idx in length_data]
    	labels = keras.utils.to_categorical(labels,len(class_names))
	print('class_number:', len(class_names), 'class names:', str(class_names))
	data = np.array(data)
	data = data.astype('float32')
	
	return data, labels,len(class_names), class_names
def Normailzation(X_train, X_test, y_train, y_test):
 	print('**********************************************')
	print('Normailzation')
	print('**********************************************')
	X_test_original = X_test
	mean_image = np.mean(X_train, axis = 0)
    	std_image  = np.std(X_train,  axis = 0) 
	X_train -= mean_image
    	X_test  -= mean_image
    	X_train /= std_image
    	X_test  /= std_image
	return X_train, X_test, y_train, y_test, X_test_original
		


def get_optimizer(optimizer = 'SGD'):
	print('optimizer:',optimizer)
	if optimizer == 'SGD':
		optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
	elif optimizer == 'RMSprop':
		optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
	elif optimizer == 'Adagrad':
		optimizer = keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
	elif optimizer == 'Adadelta':
		optimizer = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
	elif optimizer == 'Adam':
		optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
	elif optimizer == 'Adamax':
		optimizer = keras.optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
	elif optimizer == 'Nadam':
		optimizer = keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
	return optimizer
def plot_loss_accuracy(history, output_file,model_name,k_folds):
	# summarize history for accuracy
        plt.figure()
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title(model_name+' fold:'+ str(k_folds) + ' model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
	plt.savefig(pdf, format = 'pdf')	


def plot_confusion_matrix(test_softmax_output,Ytest,class_names,score_test, model_name, pdf, k_folds):
        print('**********************************************')
        print('plot confusion matrix')
        print('**********************************************')
      
        cnf_matrix = confusion_matrix(np.argmax(Ytest,axis = 1),np.argmax(test_softmax_output,axis = 1))
        plt.figure()
        plt.title('Model : '+ model_name + 'fold:'+ str(k_folds) +' Test Acc:'+str(score_test[1]))
        Confusion_Matrix(cnf_matrix, classes=class_names,
                                  title=model_name)
        plt.savefig(pdf, format='pdf')
        plt.figure()
        plt.title('Normalized Model : '+ model_name)
        Confusion_Matrix(cnf_matrix, classes=class_names,
                                  normalize=True,
                                  title=model_name)
        plt.savefig(pdf, format='pdf')
 

def get_models(input_shape, num_classes, model_name, weights='imagenet'):
    print('Input_shape:',input_shape, 'num_classes:',num_classes)
    model_names = []
    models = []
    input_tensor = Input(shape=input_shape)

    if model_name == 'xception':
        print('xception')
        xception = Xception(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=xception.output_shape[1:]))
        top_model.add(Dense(num_classes, activation='softmax'))
        model_xception = Model(inputs=xception.input, outputs=top_model(xception.output))
        return model_xception
    elif model_name == 'vgg16':
        vgg16 = VGG16(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(num_classes, activation='softmax'))
        model_vgg16 = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))
        return model_vgg16
    elif model_name == 'vgg19':
        vgg19 = VGG19(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        last = Flatten()(vgg19.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_vgg19 = Model(inputs=vgg19.input, outputs=predictions)
        return model_vgg19
    elif model_name == 'resnet50':
        print('resnet50')
        resnet50 = ResNet50(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        last = Flatten()(resnet50.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_resnet50 = Model(inputs=resnet50.input, outputs=predictions)
        return model_resnet50

    elif model_name == 'inception_v3':
        input_tensor = Input(shape=input_shape)
        inception_v3 = InceptionV3(include_top=False, weights=weights,
                                input_tensor=input_tensor) 
        last = GlobalAveragePooling2D()(inception_v3.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_inception_v3 = Model(inputs=inception_v3.input, outputs=predictions)
        model_inception_v3.summary()
        return model_inception_v3

    elif model_name=='inception_resnet_v2':
        inception_resnet_v2 = InceptionResNetV2(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        last = Flatten()(inception_resnet_v2.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_inception_resnet_v2 = Model(inputs=inception_resnet_v2.input, outputs=predictions)
        return model_inception_resnet_v2
    elif model_name=='mobile_net':
        mobile_net = MobileNet(include_top=False, weights=weights,
                      input_tensor=input_tensor)
        last = Flatten()(mobile_net.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_mobile_net = Model(inputs=mobile_net.input, outputs=predictions)
        return model_mobile_net

    elif model_name=='dense_net_121':
        dense_net_121 = DenseNet121(include_top=False, weights=weights)
        last = GlobalAveragePooling2D(name='avg_pool')(dense_net_121.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_dense_net_121 = Model(inputs=dense_net_121.input, outputs=predictions)
        return model_dense_net_121
    elif model_name=='dense_net_201':
        dense_net_201 = DenseNet201(include_top=False, weights=weights)
        last = GlobalAveragePooling2D(name='avg_pool')(dense_net_201.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_dense_net_201 = Model(inputs=dense_net_201.input, outputs=predictions)
        return model_dense_net_201
    elif model_name=='dense_net_169':
        dense_net_169 = DenseNet169(include_top=False, weights=weights)
        last = GlobalAveragePooling2D(name='avg_pool')(dense_net_169.output)
        predictions = Dense(num_classes, activation="softmax")(last)
        model_dense_net_169 = Model(inputs=dense_net_169.input, outputs=predictions)
        return model_dense_net_169
    return None

if __name__ =='__main__':
	
	parser = argparse.ArgumentParser()
    	parser.add_argument('--fine_tune_with_imagenet', default=True)
    	parser.add_argument('--train_from_scratch', default=False)
    	parser.add_argument('--test_only', default=False)
	parser.add_argument('--data_path', default='/data/eko.ai-project/view_classification_data_model/echo_modality_a4c_a5c_without_split_clean/')
	parser.add_argument('--train_num_epochs', default=1)
        parser.add_argument('--train_batch_size', default=32)
	parser.add_argument('--k_folds', default=2)
	parser.add_argument('--model_save_path', default='/data/eko.ai-project/Prototype_Run_Classified/')
	parser.add_argument('--GPUs', default=None)#None means using GPU as memory grows, int means use specific GPUs
	parser.add_argument('--save_all_predictions', default=False)
	args = parser.parse_args()
        fine_tune_with_imagenet = args.fine_tune_with_imagenet
	train_from_scratch = args.train_from_scratch
	test_only = args.test_only
	data_path = args.data_path
	train_num_epochs = args.train_num_epochs
	train_batch_size = args.train_batch_size	
	k_folds = args.k_folds
	model_save_path = args.model_save_path
	GPUs = args.GPUs
	save_all_predictions = args.save_all_predictions	
        # chOOSE the way of using GPU
	if GPUs == None:		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth=True
		sess = tf.Session(config=config)
		K.set_session(sess)
	elif GPUs != None:
		os.environ["CUDA_VISIBLE_DEVICES"]= str(GPUs)

# training part 
  


   
	pdf = PdfPages(model_save_path+'/'+'results.pdf')
	class_names = ['aa']
	
#	model_names = ['xception','inception_v3','inception_resnet_v2','dense_net_121','dense_net_201','dense_net_169','vgg16','vgg19','resnet50']
	model_names = ['resnet50']
	for model_name in model_names:
   	    images, labels,num_classes, class_names = load_data(data_path, img_rows = 224,img_cols = 224)	
	    kf = KFold(n_splits = k_folds)
	    print('k-fold:',k_folds )
	    cross_validation_accuracy = 0 
   	    for train_index, test_index in kf.split(images):
                Xtrain, Xtest = images[train_index], images[test_index]
                Ytrain, Ytest = labels[train_index], labels[test_index]
		
		print('**********************************************')
                print('training data:', Xtrain.shape, 'test data: ', Xtest.shape)
                print('**********************************************')
		X_train, X_test, y_train, y_test, X_test_original = Normailzation(Xtrain, Xtest, Ytrain, Ytest)
		# building training model	
		input_shape = Xtrain.shape[1:]
 		model = get_models(input_shape, num_classes, model_name, weights='imagenet')	
		keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=1e-6)	
		optimizer = get_optimizer('SGD')# Choice : SGD, RMSprop, Adadelta, Adam, Adamax, Nadam
		model.compile(optimizer= optimizer, loss='categorical_crossentropy',
                          metrics=['accuracy'])
                earlyStopping = EarlyStopping(monitor='val_acc', patience=20, verbose=1,
                                          mode='max')
                ckpt = ModelCheckpoint(model_save_path+model_name+'.hdf5', save_best_only=True,
                                    monitor='val_acc', mode='max',verbose = 1 )
		csv_logger = CSVLogger('training.log',)
                reduce_lr_loss = ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                                               patience=7, verbose=1, epsilon=1e-4,
                                               mode='max')
		print('**********************************************')
		print('Start training')
                print('**********************************************')		
		hist = model.fit(Xtrain, Ytrain, epochs = train_num_epochs, batch_size = train_batch_size,
				 shuffle=True, validation_data =(Xtest , Ytest), 
				callbacks=[ckpt,reduce_lr_loss,csv_logger],verbose=2)
		test_softmax_output = model.predict(Xtest)		
		test_predictions = np.argmax(test_softmax_output, axis = 1)		
		Ground_truth  =  np.argmax(Ytest, axis = 1)
		score_test = model.evaluate(Xtest, Ytest, verbose=1)
		cross_validation_accuracy += score_test[1]	
		plot_loss_accuracy(hist, pdf, model_name, k_folds)
		plot_confusion_matrix(test_softmax_output, Ytest, class_names, score_test, str(model_name), pdf,k_folds)
		if save_all_predictions == True:		
		    print('**********************************************')
                    print('Saving all the wrong predictions')
                    print('**********************************************')
		
	 
	            for ex in range(0,Ytest.shape[0]):
                     
			if  test_predictions[ex] != Ground_truth[ex]: 
                     		plt.figure() 
                        	plt.imshow(X_test_original[ex,:,:,:])
				plt.suptitle('True:'+class_names[Ground_truth[ex]]+'Pred : ' + class_names[test_predictions[ex]] + '  Prob : ' + str(np.max(test_softmax_output[ex])) )
	                plt.savefig(pdf, format='pdf')

	    print('**********************************************')
            print(model_name,'cross_validation accuracy; ',(cross_validation_accuracy/k_folds))
            print('**********************************************') 
	pdf.close()
                        
                      


############################################
# This file contains the following classes:
# 1. NNRegressionModel: Contains CNN, GoogleNet, and ResNet models for regression.
# 2. loadDataset: Loads the dataset for classification and regression.
# 3. predict: Predicts the power of the EEG signals using the ensemble method.
# 4. getInputPSD: Gets the power spectral density of the EEG signals.
# 5. normalizedPSDFeature: Normalizes the power spectral density feature.
# 6. yToOrigin: Converts the predicted value to the original scale.
# 7. evaluateRegression: metrics for evaluating the regression model.
############################################

from sklearn.metrics import r2_score
import numpy as np
import keras
from keras.models import Sequential
from keras import models
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
import matplotlib.pyplot as plt
import os
import pandas as pd
import ast

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def getInputPSD( psds):
    rightChs=['T6', 'O2', 'P4', 'T5', 'P3', 'O1']
    leftChs=['T5', 'O1', 'P3', 'T6', 'P4', 'O2']
    # p,f=psds.get_data(picks=rightChs, fmin=3, fmax=15,return_freqs=True)
    # print (f.tolist())
    rightPower=psds.get_data(picks=rightChs, fmin=3, fmax=15)*1e12
    leftPower=psds.get_data(picks=leftChs, fmin=3, fmax=15)*1e12
    rightPower=np.round(10*np.log10(rightPower), 2)
    leftPower=np.round(10*np.log10(leftPower), 2)

    return rightPower, leftPower

def normalizedPSDFeature(feature):
    feature=feature/feature.max()
    feature=feature.reshape(1,feature.shape[0],feature.shape[1],1)
    return feature

    
def yToOrigin(y):
    return y*8+4

def evaluateRegression(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1)
    acc06 = np.mean(np.abs(yToOrigin(predictions) - yToOrigin(y_test)) < 0.6)
    acc12 = np.mean(np.abs(yToOrigin(predictions) - yToOrigin(y_test)) < 1.2)
    # calculate error, MAE, R2
    error = np.abs(yToOrigin(predictions) - yToOrigin(y_test))
    mae = np.mean(error)
    r2 = r2_score(yToOrigin(y_test), yToOrigin(predictions))
    rmse = np.sqrt(np.mean(error**2))

    return  rmse, mae, r2, acc06, acc12

class NNRegressionModel:
    def __init__(self, model_name, model_folder,  X_train,  X_test, y_train, y_test,
                 epochs = 200, batch_size = 16, validation_split = 0.05, verbose = 1):
        self.model_name = model_name 
        self.model_folder = model_folder       
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.input_shape = X_train.shape[1:]
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation_split = validation_split
        self.verbose = verbose
        self.model = self.createModel()
    model_names =[ 'CNN', 'GoogleNet', 'ResNet']

    def createModel(self):
        if self.model_name == self.model_names[0]:
            model = self.modelCNNRegresion()
        elif self.model_name == self.model_names[1]:
            model = self.modelGoogleNet()
        elif self.model_name == self.model_names[2]:
            model = self.modelResNet()
        else:
            print('Model not found, only {} are available.'.format(self.model_names))
            return None
        return model
    
    def train(self):
        model=self.model
        temp_folder = './temp_model/'
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)
        # compile the keras model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MSE'])
        # save the best model, name /temp_model/best_model_{epoch}.h5
        model_checkpoint = keras.callbacks.ModelCheckpoint(temp_folder+'best_model_{epoch}.h5', monitor='loss', save_best_only=True)
        # fit the keras model on the dataset
        result = model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, validation_split=self.validation_split, callbacks=[model_checkpoint], verbose=self.verbose) #, class_weight=class_weight)
        # plot loss
        if self.verbose == 1:
            plt.plot(result.history['loss'], label='loss')
            plt.plot(result.history['val_loss'], label='val_loss')

            plt.legend()
            plt.show()

        def load_best_model():
            # load the best model
            # read the best model in /temp_model and sort by creation time, get the latest one
            files = os.listdir(temp_folder)
            files = [f for f in files if f.endswith('.h5')]
            files.sort(key=lambda x: os.path.getmtime(temp_folder+x), reverse=True)
            best_model = files[0]
            best_epoch = best_model.split('_')[-1].split('.')[0]
            print(best_model)
            model = keras.models.load_model(temp_folder+best_model)
            return model, best_epoch
        def clean_temp_model():
            # delete all files in temp_model
            for the_file in os.listdir(temp_folder):
                file_path = os.path.join(temp_folder, the_file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                except Exception as e:
                    print(e)
        
        model, best_epoch = load_best_model()
        rmse, mae, r2, acc06, acc12 = evaluateRegression(model, self.X_test, self.y_test)
        model.save(self.model_folder+'/{}_mae{}_epoch{}.h5'.format(self.model_name, round(mae, 3), best_epoch))
        clean_temp_model()
        del model
        return rmse, mae, r2, acc06, acc12
    
    def modelCNNRegresion(self):
        # define the keras model
        regularizer = keras.regularizers.l2(0.01)
        model = Sequential()
        model.add(Conv2D(64, (3, 3), activation='relu', input_shape=self.input_shape, padding='same'))
        model.add(Conv2D(128, (1, 3), activation='relu'))
        model.add(MaxPooling2D((1, 2)))
        model.add(Conv2D(256, (1, 3), activation='relu'))
        model.add(MaxPooling2D((1, 2)))
        model.add(Conv2D(512, (1, 3), activation='relu'))        
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(512, (3, 3), activation='relu', kernel_regularizer=regularizer))
        model.add(Flatten())
        model.add(Dense(64, activation='relu'))
        # drop
        model.add(keras.layers.Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))

        # compile the keras model
        model.compile(loss='mean_squared_error', optimizer='adam', metrics=['MSE'])
        return model

    def modelGoogleNet(self):
        def inception_module(x, filters):
            # 1x1 convolution
            conv1x1_1 = layers.Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)
            # 1x1 convolution followed by 3x3 convolution
            conv1x1_2 = layers.Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
            conv3x3 = layers.Conv2D(filters[2], (3, 3), padding='same', activation='relu')(conv1x1_2)
            # 1x1 convolution followed by 5x5 convolution
            conv1x1_3 = layers.Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
            conv5x5 = layers.Conv2D(filters[4], (5, 5), padding='same', activation='relu')(conv1x1_3)
            # 3x3 max pooling followed by 1x1 convolution
            maxpool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
            conv1x1_4 = layers.Conv2D(filters[5], (1, 1), padding='same', activation='relu')(maxpool)
            # Concatenate the outputs along the channel axis
            inception = layers.concatenate([conv1x1_1, conv3x3, conv5x5, conv1x1_4], axis=-1)
            return inception

        def googlenet_model(input_shape):
            inputs = keras.Input(shape=input_shape)
            # Initial convolutional layer
            x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
            x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            # First Inception module
            x = inception_module(x, [64, 128, 128, 32, 32, 32])
            # Additional Inception modules
            x = inception_module(x, [128, 192, 192, 96, 96, 96])
            x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
            x = inception_module(x, [192, 208, 208, 80, 80, 80])
            x = inception_module(x, [160, 224, 224, 64, 64, 64])
            x = inception_module(x, [128, 256, 256, 64, 64, 64])

            # Global average pooling and dense layers
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dense(128, activation='relu')(x)
            # add dropout and batch normalization
            # x = layers.Dropout(0.2)(x)
            # x = layers.BatchNormalization()(x)
            output = layers.Dense(1, activation='sigmoid')(x)
            model = models.Model(inputs, output, name='googlenet_model')
            return model        
        model = googlenet_model(self.input_shape)
        return model

    def modelResNet(self):
        def residual_block(x, filters, conv_num):
            # Shortcut
            s = layers.Conv2D(filters, 1, padding='same', strides=2, name='shortcut'+str(conv_num))(x)
            s = layers.BatchNormalization(name='shortcut_bn'+str(conv_num))(s)
            # Residual
            x = layers.Conv2D(filters, 3, padding='same', strides=2, name='conv1'+str(conv_num))(x)
            x = layers.BatchNormalization(name='conv1_bn'+str(conv_num))(x)
            x = layers.Activation('relu', name='conv1_relu'+str(conv_num))(x)
            x = layers.Conv2D(filters, 3, padding='same', name='conv2'+str(conv_num))(x)
            x = layers.BatchNormalization(name='conv2_bn'+str(conv_num))(x)
            x = layers.add([x, s])
            x = layers.Activation('relu', name='res'+str(conv_num))(x)
            return x

        def resnet_model(input_shape):
            inputs = keras.Input(shape=input_shape)
            x = layers.Conv2D(32, 3, padding='same', strides=2, name='conv1')(inputs)
            x = layers.BatchNormalization(name='conv1_bn')(x)
            x = layers.Activation('relu', name='conv1_relu')(x)
            x = layers.Conv2D(64, 3, padding='same', name='conv2')(x)
            x = layers.BatchNormalization(name='conv2_bn')(x)
            x = layers.Activation('relu', name='conv2_relu')(x)
            x = residual_block(x, 128, 3)
            x = residual_block(x, 256, 4)
            x = residual_block(x, 512, 5)
            x = layers.GlobalAveragePooling2D(name='pool')(x)
            # drop out
            # x = layers.Dropout(0.2)(x)
            
            output = layers.Dense(1, activation='sigmoid', name='fc')(x)
            model = models.Model(inputs, output, name='resnet_model')
            return model

        model = resnet_model(self.input_shape)
        return model
    
class loadDataset:        
    typesList = ['classification', 'regression']
    def __init__(self):
        pass
    def load(self, labelFile, type , class_num=3):
        self.class_num = class_num

        # read the csv file
        dfAll= pd.read_csv(labelFile)
        # classification
        if type==self.typesList[0]:
            # df right_peak 0->0, 99->1, others->2
            dfAll['right_peak'] = dfAll['right_peak'].apply(lambda x: 0 if x==0 else 1 if x==99 else 2)
            # left_peak 0->0, 99->1, others->2
            dfAll['left_peak'] = dfAll['left_peak'].apply(lambda x: 0 if x==0 else 1 if x==99 else 2)
            # one hot encoding
        # regression data
        elif type==self.typesList[1]:
            # remove the rows with peak=99 or 0
            dfAll = dfAll[dfAll['right_peak']!=0]
            dfAll = dfAll[dfAll['right_peak']!=99]
            dfAll = dfAll[dfAll['left_peak']!=0]
            dfAll = dfAll[dfAll['left_peak']!=99]
            dfAll['right_peak'] = dfAll['right_peak'].apply(lambda x: (float(x)-4)/8)
            dfAll['left_peak'] = dfAll['left_peak'].apply(lambda x: (float(x)-4)/8)   
        
        return self.process(dfAll)   

    def loadKfold(self, labelFile, Kindex):
        # read the csv file
        dfAll= pd.read_csv(labelFile)    
        # add a column isTrain, 1 for train, 0 for test, set the kfold=Kindex as test
        dfAll['isTrain'] = 1
        dfAll.loc[dfAll['kfold']==Kindex, 'isTrain'] = 0
        # remove the rows with peak=99 or 0
        dfAll = dfAll[dfAll['right_peak']!=0]
        dfAll = dfAll[dfAll['right_peak']!=99]
        dfAll = dfAll[dfAll['left_peak']!=0]
        dfAll = dfAll[dfAll['left_peak']!=99]
        dfAll['right_peak'] = dfAll['right_peak'].apply(lambda x: (float(x)-4)/8)
        dfAll['left_peak'] = dfAll['left_peak'].apply(lambda x: (float(x)-4)/8)   
        return self.process(dfAll)     
        


    def process(self, dfAll):
        
        # train data = column split=train
        df_train = dfAll[dfAll['isTrain']==1]
        # test data = column split=test
        df_test = dfAll[dfAll['isTrain']==0]

        for col in ['rightPower', 'leftPower']:
            for df in [df_train, df_test]:
                for i in range(df.shape[0]):
                    string = df[col].iloc[i]
                    arr=ast.literal_eval(string)
                    arr=normalizedPSDFeature(np.array(arr))
                    df.loc[df.index[i], col] = arr

        X_train_right = np.array(df_train['rightPower'].tolist())
        X_train_left = np.array(df_train['leftPower'].tolist())
        X_train = np.concatenate((X_train_right, X_train_left), axis=0)
        X_train=X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], 1)

        X_test_right = np.array(df_test['rightPower'].tolist())
        X_test_left = np.array(df_test['leftPower'].tolist())
        X_test = np.concatenate((X_test_right, X_test_left), axis=0)
        print ("X_test shape", X_test.shape)
        X_test=X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], 1)

        y_train_right = df_train['right_peak'].values
        y_train_left = df_train['left_peak'].values
        y_train = np.concatenate((y_train_right, y_train_left), axis=0)

        y_test_right = df_test['right_peak'].values
        y_test_left = df_test['left_peak'].values
        y_test = np.concatenate((y_test_right, y_test_left), axis=0)

        if type==self.typesList[0]:
            # one hot encoding
            y_train = np.eye(self.class_num)[y_train]
            y_test = np.eye(self.class_num)[y_test]              

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
class loadSmallDataset:        
    typesList = ['classification', 'regression']
    def __init__(self):
        pass
    def load(self, labelFile, type , datasetRatio, class_num=3):
        self.class_num = class_num

        # read the csv file
        dfAll= pd.read_csv(labelFile)
        # classification
        if type==self.typesList[0]:
            # df right_peak 0->0, 99->1, others->2
            dfAll['right_peak'] = dfAll['right_peak'].apply(lambda x: 0 if x==0 else 1 if x==99 else 2)
            # left_peak 0->0, 99->1, others->2
            dfAll['left_peak'] = dfAll['left_peak'].apply(lambda x: 0 if x==0 else 1 if x==99 else 2)
            # one hot encoding
        # regression data
        elif type==self.typesList[1]:
            # remove the rows with peak=99 or 0
            dfAll = dfAll[dfAll['right_peak']!=0]
            dfAll = dfAll[dfAll['right_peak']!=99]
            dfAll = dfAll[dfAll['left_peak']!=0]
            dfAll = dfAll[dfAll['left_peak']!=99]


            dfAll['right_peak'] = dfAll['right_peak'].apply(lambda x: (float(x)-4)/8)
            dfAll['left_peak'] = dfAll['left_peak'].apply(lambda x: (float(x)-4)/8)        


        # train data = column split=train
        df_train = dfAll[dfAll['isTrain']==1]

        # get a small dataset with datasetRatio, and maintain the distribution of the right_peak
        df_train = df_train.groupby('right_peak').apply(
            lambda x: x.sample(n=max(1, int(len(x) * datasetRatio)), replace=False)
        ).reset_index(drop=True)
        
        df_test = dfAll[dfAll['isTrain']==0]
        # get a small dataset with datasetRatio, and maintain the distribution of the right_peak
        df_test = df_test.groupby('right_peak').apply(
            lambda x: x.sample(n=max(1, int(len(x) * datasetRatio)), replace=False)
        ).reset_index(drop=True)

        for col in ['rightPower', 'leftPower']:
            for df in [df_train, df_test]:
                for i in range(df.shape[0]):
                    string = df[col].iloc[i]
                    arr=ast.literal_eval(string)
                    arr=normalizedPSDFeature(np.array(arr))
                    df.loc[df.index[i], col] = arr

        X_train_right = np.array(df_train['rightPower'].tolist())
        X_train_left = np.array(df_train['leftPower'].tolist())
        X_train = np.concatenate((X_train_right, X_train_left), axis=0)
        X_train=X_train.reshape(X_train.shape[0], X_train.shape[2], X_train.shape[3], 1)

        X_test_right = np.array(df_test['rightPower'].tolist())
        X_test_left = np.array(df_test['leftPower'].tolist())
        X_test = np.concatenate((X_test_right, X_test_left), axis=0)
        X_test=X_test.reshape(X_test.shape[0], X_test.shape[2], X_test.shape[3], 1)

        y_train_right = df_train['right_peak'].values
        y_train_left = df_train['left_peak'].values
        y_train = np.concatenate((y_train_right, y_train_left), axis=0)

        y_test_right = df_test['right_peak'].values
        y_test_left = df_test['left_peak'].values
        y_test = np.concatenate((y_test_right, y_test_left), axis=0)

        if type==self.typesList[0]:
            # one hot encoding
            y_train = np.eye(self.class_num)[y_train]
            y_test = np.eye(self.class_num)[y_test]              

        print(X_train.shape, y_train.shape)
        print(X_test.shape, y_test.shape)
        return X_train, X_test, y_train, y_test
    
class predict:
    def __init__(self, model_folder, model_names=['CNN','GoogleNet','ResNet']):
        self.model_folder = model_folder
        self.model_names = model_names
        self.modelsArray = []
        # get all .h5 files ./
        files = os.listdir(self.model_folder)
        files = [f for f in files if f.endswith('.h5')]
        self.models = []
        for name in model_names:
            for file in files:
                if name in file:
                    self.models.append(keras.models.load_model(self.model_folder+'/'+file, compile=False))
        
    # predict power
    def predictPower(self, power):       

        def calculate(model):            
            preds = model.predict(power)
            preds = preds.reshape(-1)
            return preds[0]

        predictions = []
        for model in self.models:
            prediction = calculate(model)
            predictions.append(prediction)

        # average the predictions
        predictions = np.array(predictions)
        predictions = np.mean(predictions)
        return yToOrigin( predictions)    
    
    def ensemble(self, psds):
        rightPower, leftPower = getInputPSD(psds)
        rightPower=normalizedPSDFeature(rightPower)
        leftPower=normalizedPSDFeature(leftPower)
        rightPreds = self.predictPower(rightPower)
        leftPreds = self.predictPower(leftPower)
        return np.round(rightPreds, 1), np.round(leftPreds, 1)


import numpy as np
import EvaluationMetricGenerator as t1
import LogisticRegression as t2
import pydot

from tensorflow import keras

#Function A:The following functions are used to implement several different kinds of neural networks
#==========================================================================================================#
def ArtificialNeuralNetworks(X_train,y_train,X_validation,y_validation):
    ANN_model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[600,1]),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(300, activation="relu"),
        keras.layers.Dense(100, activation="relu"),
        keras.layers.Dense(2, activation="softmax")
    ])

    ANN_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = ANN_model.fit(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))
    #CNN_model.evaluate(X_validation, y_validation)
    return ANN_model

def ConvolutionalNeuralNetworks(X_train,y_train,X_validation,y_validation):
    CNN_model = keras.models.Sequential([
        keras.layers.Conv2D(filters=8, kernel_size=[3,3], activation="relu", input_shape=[30, 20, 1]),

        keras.layers.MaxPooling2D(strides=(2,2)),

        keras.layers.Conv2D(filters=16, kernel_size=[3,3], activation="relu"),

        keras.layers.MaxPooling2D(strides=(2,2)),

        keras.layers.Flatten(),
        keras.layers.Dense(units=8, activation='relu'),
        keras.layers.Dense(units=2, activation='softmax')
    ])

    CNN_model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])
    history = CNN_model.fit(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))
    #CNN_model.evaluate(X_validation, y_validation)
    return CNN_model

def RecurrentNeuralNetworks(X_train,y_train,X_validation,y_validation):
    RNN_model = keras.models.Sequential([
        keras.layers.SimpleRNN(32, return_sequences=True, input_shape=[30, 20]),
        keras.layers.SimpleRNN(32, return_sequences=False),
        keras.layers.Dense(units=1, activation='sigmoid')
    ])

    RNN_model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
    history = RNN_model.fit(X_train, y_train, epochs=10, validation_data=(X_validation, y_validation))
    #CNN_model.evaluate(X_validation, y_validation)
    return RNN_model


if __name__ == '__main__':

    #Part A: The following code reads the data X_train, y_train, X_validation, y_validation in vector format required for various neural network training. (This part takes about 16min)
    #==========================================================================================================#

    X_train= np.loadtxt("TrainData_File/X_train_task2.txt",delimiter=',')
    y_train= np.loadtxt("TrainData_File/y_train_task2.txt",delimiter=',')
    X_validation = np.loadtxt("TrainData_File/X_validation_task2.txt",delimiter=',')
    y_validation = np.loadtxt("TrainData_File/y_validation_task2.txt",delimiter=',')

    X_train_ANN = X_train.reshape((-1,600,1))
    y_train_ANN = y_train.reshape((-1,1))
    X_validation_ANN = X_validation.reshape((-1,600,1))
    y_validation_ANN = y_validation.reshape((-1,1))

    X_train_CNN = X_train.reshape((-1,30,20,1))
    y_train_CNN = y_train.reshape((-1,1,1))
    X_validation_CNN = X_validation.reshape((-1,30,20,1))
    y_validation_CNN = y_validation.reshape((-1,1,1))

    X_train_RNN = X_train.reshape((-1,30,20))
    y_train_RNN = y_train.reshape((-1,1))
    X_validation_RNN = X_validation.reshape((-1,30,20))
    y_validation_RNN = y_validation.reshape((-1,1))

    #Part B: The following code is used to train three neural network models and save them (this part takes about 1h46min)
    #==================================================== ===================================================== =======#
    ANN_model = ArtificialNeuralNetworks(X_train_ANN,y_train_ANN,X_validation_ANN,y_validation_ANN)
    CNN_model = ConvolutionalNeuralNetworks(X_train_CNN,y_train_CNN,X_validation_CNN,y_validation_CNN)
    RNN_model = RecurrentNeuralNetworks(X_train_RNN,y_train_RNN,X_validation_RNN,y_validation_RNN)

    ANN_model.save("TrainData_File/task4_ANN_model.h5")
    CNN_model.save("TrainData_File/task4_CNN_model.h5")
    RNN_model.save("TrainData_File/task4_RNN_model.h5")

    #Part C: The following code uses the three trained neural networks to predict the validation data (this part takes about 3 minutes)
    #==================================================== ===================================================== =======#
    ANN_model = keras.models.load_model("TrainData_File/task4_ANN_model.h5")
    CNN_model = keras.models.load_model("TrainData_File/task4_CNN_model.h5")
    RNN_model = keras.models.load_model("TrainData_File/task4_RNN_model.h5")

    # keras.utils.plot_model(ANN_model, to_file='ANNmodel.pdf', show_shapes=True)
    # keras.utils.plot_model(CNN_model, to_file='CNNmodel.pdf', show_shapes=True)
    # keras.utils.plot_model(RNN_model, to_file='RNNmodel.pdf', show_shapes=True)

    ANN_model.summary()
    CNN_model.summary()
    RNN_model.summary()

    y_validation_prediction = ANN_model.predict(X_validation_ANN)[:,1]
    y_validation_prediction = CNN_model.predict(X_validation_CNN)[:,1]
    y_validation_prediction = RNN_model.predict(X_validation_RNN)

    #Part D: The following code sorts and evaluates the validation data based on the correlation prediction results (this part takes about 6min)
    #================================================== ===================================================== ========#

    #Initialize all_data containing all data records, and query_data containing all queries
    all_data, query_data = t2.InitializeData(y_validation_prediction)
    
    #According to the prediction correlation based on LambdaMARTModel in all_data, return the sorted all_data, namely LambdaMART_Df
    NN_Df = t2.GenerateSortedDataFrame(all_data,'NN')

    #Save the sorting results to the NN.txt file according to the format requirements
    order = ['qid', 'A', 'pid', 'rank', 'score', 'algoname']
    NN_Df_record = NN_Df[order]
    NN_Df_record = NN_Df_record.query('rank<=100')
    NN_Df_record.to_csv('NN.txt',sep='\t',index=False,header=False)

    #Calculate AP and NDCG corresponding to each query
    query_data = t1.CalculateAveragePrecision(query_data,NN_Df)
    query_data = t1.CalculateNDCG(query_data,NN_Df)

    print("The MAP of Neural Network is")
    print(query_data['ave_precision'].mean())

    print("The NDCG of Neural Network is")
    print(query_data['NDCG'].mean())


# Information-Retrieval-Engine-Based-on-BM25-and-LambdaMART

In this system, a number of information retrieval models to solve the problem of paragraph retrieval were established, and the retrieval effect was evaluated. The following are instructions on how to run the relevant code and how to process input and output data

### 1. Solution and output file:

In task1.py, metrics for evaluating information retrieval models: mean mean precision (MAP) and normalized discounted cumulative gain (NDCG) are established, and the performance of the BM25 retrieval model built in previous work is evaluated.
In task2.py, based on the Word2Vec model, word embedding vectors of training data and validation data are constructed, and a logistic regression model is trained to predict the relevance before query and paragraph. The retrieval and sorting results are stored in LR.txt in the root directory.
In task3.py, the LambdaMART model is used to build and evaluate a similar query ranking model. The retrieval and sorting results are stored in LM.txt in the root directory.
In task4.py, three different types of neural networks are constructed and their query performance is compared and analyzed. Among them, the retrieval and sorting results of the RNN with the best performance are stored in NN.txt in the root directory.

In the Report of Assignment2.pdf, the whole process and output results of the above solutions are described, and the analysis and discussion are carried out.

### 2. Input data
For task1.py, task2.py, task3.py, task4.py, the input data test-queries.tsv, candidate_passages_top1000.tsv, train_data.tsv and validation_data.tsv need to be stored in the same folder with the same path as them, to make it work properly.

### 3. Output data
In the solution of this assignment, part A (word embedding) of task2.py takes the longest time (about 30+h), part A of task3.py (generates XGBoost training and validation files) and part B of task4.py The part (training the neural network) also takes a long time (about 1h+), so this solution chooses to store the generated data to save time, all the stored data will be output to the TrainData_File folder (the folder is currently empty) .
Below are all the files that may be stored in this folder.


X_train_task2.txt
X_validation_task2.txt
y_train_task2.txt
y_validation_task2.txt
task2_b.txt
task2_W.txt
task3_train_data.txt
task3_validation_data.txt
task4_ANN_model.h5
task4_CNN_model.h5
task4_RNN_model.h5

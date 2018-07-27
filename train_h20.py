from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import pickle
import numpy as np
import os

def get_data_statistics(num_minis,std_scaler):
    ''' Calculates the mean and standard deviation of given data.

    Arguments :
        num_minis  -- Int.
                      The number of mini-batches.
        std_scaler -- sklearn.preprocessing.StandardScaler Object

    Returns :
        std_scaler -- The StandardScaler Object with the learned
                      mean and standard deviation.
    '''

    path_to_data = './Data_training/Fast_H20/'
    file_type = '.pkl'
    
    for i in range(num_minis):

        # For train
        path = path_to_data + 'train_' + str(i) + '_' + file_type
        with open(path,'rb') as f :
            x = pickle.load(f)

        x = np.reshape(x, newshape = (-1,14))

        # Partial fitting the mini-batch using standard scaler
        std_scaler.partial_fit(x)
        
        print("{} is Done".format(i))

    print("Mean : {}, Std Dev : {}".format(std_scaler.mean_, pow(std_scaler.var_,0.5) ))
    return std_scaler

def normalize(number,std_scaler) :
    ''' Transforms the given data with the calculated mean and std-dev.

    Arguments :
        number -- Int.
                  The mini-batch number.
        std_scaler -- sklearn.preprocessing.StandardScaler Object.
                      The learned standard scaler for the training
                      data.
    Returns :
        x - The normalized data input.
        y - The corresponding ground truth labels.
    '''

    path_to_data = './Data_training/Fast_H20/'
    file_type = '.pkl'
    
    # For train
    path = path_to_data + 'train_' + str(number) + '_' + file_type
    with open(path,'rb') as f :
        x = pickle.load(f)

    # For test
    path = path_to_data + 'test_' + str(number) + '_' + file_type
    with open(path,'rb') as f :
        y = pickle.load(f)    

    x = np.reshape(x , newshape = (-1,14))
    y = np.reshape(y, newshape = (112*112*100))
    
    x = std_scaler.transform(x)
    
    return x,y

def train(std_scaler,sgd_svm,sgd_lr,rfc,num_minis,current_batch = 0) :
    ''' Trains three machine learning models on the trainig data.

    Description :
        Trains a stochastic gradient descent support vector machine ("sgd_svm"),
        a stochastic gradient descent logistic regression ("sgd_lr") and a
        random forest classifier on the training data.

    Arguments :

        std_scaler    -- sklearn.preprocessing.StandardScaler Object.
                         The learned standard scaler for the training
                         data.
        sgd_svm       -- sklearn.linear_model.SGDClassifier Object.
                         A support vector machine to be trained on the
                         mini-batches.
        sgd_lr        -- sklearn.linear_model.SGDClassifier Object.
                         A logistic regression model to be trained on the
                         mini-batches.
        rfc           -- sklearn.ensemble.RandomForestClassifier Object.
                         A random forest classifer to be trained on the mini-batches.
        current_batch -- Int. (Default = 0)
                         Integer meant for partially trained models. If models have been trained for "x"
                         mini-batches then assign "current_batch" to "x+1" to continue training.

    Returns :
        sgd_svm -- A trained support vector machine.
        sgd_lr  -- A trained logistic regression model.
        rfc     -- A trained random forest classifier.
    '''

    for i in range(current_batch,num_minis) :

        # Normalize the data
        X,Y = normalize(i,std_scaler)

        # Partially fit the svm,lr and rfc
        sgd_svm.partial_fit(X,Y,classes = [0,1])
        sgd_lr.partial_fit(X,Y,classes = [0,1])
        rfc.fit(X,Y)
        rfc.n_estimators += 3

        # Checking training accuracy
        pred_sgd_svm = sgd_svm.predict(X)
        pred_sgd_lr = sgd_lr.predict(X)
        pred_rfc = rfc.predict(X)

        if i%10 == 0:
            save_classifiers(sgd_svm,sgd_lr,rfc)            

        print("{} : ".format(i))
        print("Scores --> sgd_svm : {}, sgd_lr : {}, rfc : {}".format(accuracy_score(Y,pred_sgd_svm),accuracy_score(Y,pred_sgd_lr),accuracy_score(Y,pred_rfc)))

    return sgd_svm,sgd_lr,rfc

def save_classifiers(sgd_svm,sgd_lr,rfc):
    ''' Save the trained support vector machine("sgd_svm"),logistic regression model("sgd_lr") and random forest classifier model("rfc").'''

    path = os.getcwd()
    
    with open( os.path.join(path,'Parameters/Fast_H20/sgd_svm.pkl') ,'wb') as f :
        pickle.dump(sgd_svm,f)

    with open( os.path.join(path,'Parameters/Fast_H20/sgd_lr.pkl') ,'wb') as f :
        pickle.dump(sgd_lr,f)

    joblib_file = os.path.join(path,'Parameters/Fast_H20/rfc.pkl')
    joblib.dump(rfc,joblib_file)

def save_std_scaler(std_scaler) :
    ''' Save the already fit Standard_Scaler Object ("std_scaler").''' 
    path = os.getcwd()

    with open( os.path.join(path,'Parameters/Fast_H20/std_scaler.pkl') ,'wb') as f :
        pickle.dump(std_scaler,f)

def get_std_scaler():
    ''' Get the saved Standard_Scaler Object ("std_scaler").'''

    path = os.getcwd()

    with open( os.path.join(path,'Parameters/Fast_H20/std_scaler.pkl') ,'rb') as f :
        std_scaler = pickle.load(f)
    
    return std_scaler
    
def main(num_minis,current_batch = 0,statistics_pre_computed = False,classifiers_partial_computed = False) :

    path = os.getcwd()

    # If the mean and std-dev is not calculated, then first compute those values.
    if not statistics_pre_computed :
        std_scaler = StandardScaler()
        std_scaler = get_data_statistics(num_minis,std_scaler)
        save_std_scaler(std_scaler)
    else:
        std_scaler = get_std_scaler()

    if not classifiers_partial_computed:
        # If the classifiers have just been instantiated.
        sgd_svm = SGDClassifier(loss = 'hinge',max_iter = 10)
        sgd_lr = SGDClassifier(loss = 'log',max_iter = 10)
        rfc = RandomForestClassifier(n_estimators = 5,max_depth=15,warm_start = True)
        sgd_svm,sgd_lr,rfc = train(std_scaler,sgd_svm,sgd_lr,rfc,num_minis)

    else :
        # If the classifiers are already partially trained
        with open( os.path.join(path,'Parameters/Fast_H20/sgd_svm.pkl') ,'rb') as f :
            sgd_svm = pickle.load(f)
    
        with open( os.path.join(path,'Parameters/Fast_H20/sgd_lr.pkl') ,'rb') as f :
            sgd_lr = pickle.load(f)
    
        joblib_file = os.path.join(path,'Parameters/Fast_H20/rfc.pkl')
        rfc = joblib.load(joblib_file)

        sgd_svm,sgd_lr,rfc = train(std_scaler,sgd_svm,sgd_lr,rfc,num_minis,current_batch)

    save_classifiers(sgd_svm,sgd_lr,rfc)

if __name__ == '__main__' :

    # If the mean and standard deviation have already been calculated, then put this as true.
    statistics_pre_computed = False
    
    # If the classifiers have been trained on some minibatches (till current_batch -1), then put this as true and put value of current_batch. 
    classifiers_partial_computed = False
    current_batch = 0
    
    main(100,current_batch,False,False)






















    

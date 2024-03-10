# For K-Nearest Neighbors (KNN)
from sklearn.neighbors import KNeighborsClassifier

# For Logistic Regression (LR)
from sklearn.linear_model import LogisticRegression

# For Support Vector Machine (SVM)
from sklearn.svm import SVC

# For Random Forest (RF)
from sklearn.ensemble import RandomForestClassifier

# For Decision Tree (DT)
from sklearn.tree import DecisionTreeClassifier

# For Naive Bayes (NB)
from sklearn.naive_bayes import MultinomialNB

# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import accuracy_score, make_scorer
# from model import X_train_features , Y_train

class EmailClassifierFactory:
    def create_classifier(self, algorithm_id):
        if algorithm_id == 1:
            print("You use the LR algorithm \n")
            return LogisticRegression()
        elif algorithm_id == 2:
            print("You use the SVM algorithm \n")
            return SVC()
        elif algorithm_id == 3:
            print("You use the RF algorithm \n")
            return RandomForestClassifier()
        elif algorithm_id == 4:
            print("You use the DT algorithm \n")
            return DecisionTreeClassifier()
        elif algorithm_id == 5:
            print("You use the NB algorithm \n")
            return MultinomialNB()
        else:
            raise ValueError("Invalid algorithm ID")
        

       
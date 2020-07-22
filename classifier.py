import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
train=pd.read_csv("Training.csv")
test=pd.read_csv("Test.csv")

columns=['X1','X2','X3','X4','X5','X6','X7','X8']

#fitting the model
from sklearn import svm
X=train[columns]
y=train.Y
clf=svm.SVC(kernel="rbf",C=50)
clf=clf.fit(X,y)
pred=clf.predict(test[columns])
#preparing the csv file
submission_df={"0":pred}
submission=pd.DataFrame(submission_df)
submission.to_csv("prediction.csv",index=False,header=False)

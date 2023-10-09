# Implementation-of-Linear-Regression-Using-Gradient-Descent

# AIM:

To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Equipments Required:

    Hardware – PCs
    Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm
# Steps involved

1.Data Preparation: The first step is to prepare the data for the model. This involves cleaning the data, handling missing values and outliers, and transforming the data into a suitable format for the model.

2.Split the data: Split the data into training and testing sets. The training set is used to fit the model, while the testing set is used to evaluate the model's performance.

3.Define the model: The next step is to define the logistic regression model. This involves selecting the appropriate features, specifying the regularization parameter, and defining the loss function.

4.Train the model: Train the model using the training data. This involves minimizing the loss function by adjusting the model's parameters.

5.Evaluate the model: Evaluate the model's performance using the testing data. This involves calculating the model's accuracy, precision, recall, and F1 score.

6.Tune the model: If the model's performance is not satisfactory, you can tune the model by adjusting the regularization parameter, selecting different features, or using a different algorithm.

7.Predict new data: Once the model is trained and tuned, you can use it to predict new data. This involves applying the model to the new data and obtaining the predicted outcomes.

8.Interpret the results: Finally, you can interpret the model's results to gain insight into the relationship between the input variables and the output variable. This can help you understand the factors that influence the outcome and make informed decisions based on the results.
# Program:
```
import pandas as pd
data=pd.read_csv("/content/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
# Output:
1.Placement Data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/8a616a19-f88d-4701-842e-ec39f92cca80)

2.Salary Data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/302eaa95-851f-41f2-a17d-fb9191cf40a4)

3. Checking the null function()
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/99f359dd-8eaf-448f-9f06-25797429226b)

4.Data Duplicate
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/e6f329f7-c4cc-4ed7-8d48-8b385e896a8c)

5.Print Data
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/0047f6d0-bce1-42b3-8f03-baad0fe1aafc)

6.Data Status
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/bb30bffc-1ae0-461a-a49c-ffaa72a2d86a)

7.y_prediction array
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/91050a54-405b-438d-9042-7ec1ac739a3d)

8.Accuracy value
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/038e7496-21c5-4276-9794-60bc262fd93d)

9.Confusion matrix
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/ab8aefbb-8974-4f29-b6ac-1985cbacf218)

10.Classification Report
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/24011a87-e892-4e6d-a931-1d79d6211fe6)

11.Prediction of LR
![image](https://github.com/AGALYARAMESHKUMAR/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119394395/39fd1e28-b5f1-4e67-bf6a-112068a26910)

# Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

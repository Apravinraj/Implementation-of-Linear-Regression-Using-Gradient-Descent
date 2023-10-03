# Implementation-of-Linear-Regression-Using-Gradient-Descent
## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
Hardware – PCs
Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
Start the program.
import numpy as np.
Give the header to the data.
Find the profit of population.
Plot the required graph for both for Gradient Descent Graph and Prediction Graph.
End the program.
## Program:
```

#Program to implement the linear regression using gradient descent.
#Developed by: S AFSAR JUMAIL
#RegisterNumber: 212222240004

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data=pd.read_csv("/content/ex1.txt",header=None)

plt.scatter(data[0],data[1])
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of city (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Predication")

def computeCost(x,y,theta):
  m=len(y)
  h=x.dot(theta)
  square_err=(h-y)**2
  return 1/(2*m)*np.sum(square_err)
  
data_n=data.values
m=data_n[:,0].size
x=np.append(np.ones((m,1)),data_n[:,0].reshape(m,1),axis=1)
y=data_n[:,1].reshape(m,1)
theta=np.zeros((2,1))
computeCost(x,y,theta)

def gradientDescent(x,y,theta,alpha,num_iters):
  m=len(y)
  J_history=[]
  for i in range(num_iters):
    predictions=x.dot(theta)
    error=np.dot(x.transpose(),(predictions-y))
    descent=alpha*1/m*error
    theta-=descent
    J_history.append(computeCost(x,y,theta))
  return theta,J_history
  
theta,J_history = gradientDescent(x,y,theta,0.01,1500)
print("h(x) ="+str(round(theta[0,0],2))+" + "+str(round(theta[1,0],2))+"x1")

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")


plt.scatter(data[0],data[1])
x_value=[x for x in range(25)]
y_value=[y*theta[1]+theta[0] for y in x_value]
plt.plot(x_value,y_value,color='r')
plt.xticks(np.arange(5,30,step=5))
plt.yticks(np.arange(-5,30,step=5))
plt.xlabel("Population of City (10,000s)")
plt.ylabel("Profit ($10,000")
plt.title("Profit Prediction")

def predict(x,theta):
  predictions= np.dot(theta.transpose(),x)
  return predictions[0]
  
predict1=predict(np.array([1,3.5]),theta)*1000
print("For population = 35,000, we predict a profit of $"+str(round(predict1,0)))

predict2=predict(np.array([1,7]),theta)*1000
print("For population = 70,000, we predict a profit of $"+str(round(predict2,0)))


```
## Output:
![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/dc7f2dfe-0104-4017-a4ea-3ea01888895a)

![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/e403cfef-9188-4cc9-89e9-949c297acb81)

![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/65691062-33de-4af1-95ea-0d609ee3a4fb)

![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/6414249e-2f9b-4b91-bd18-eb77a5ef0568)
![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/d0b4c1a0-d083-4fc8-8788-71283d951e98)
![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/bbde385e-debf-43bb-9753-1dccc9915315)
![image](https://github.com/Afsarjumail/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/118343395/2bec26b6-3f3a-41b6-91e7-83eea17fa668)

## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.

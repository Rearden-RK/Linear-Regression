import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#We have used the Student Study Hour available from source: https://www.kaggle.com/datasets/aditeloo/student-study-hour-v2
data = pd.read_csv(r"C:\Users\reade\Downloads\Student Study Hour V2.csv")

# Plotting the data
# plt.scatter(data.Hours, data.Scores)
# plt.show()

#loss function taking parameters m - weights, b - bias and data - actual data
def loss_function(m, b, data):

	total_error = 0

	for i in range(len(data)):
		x = points.iloc[i].Hours # inputs
		y = points.iloc[i].Scores # outputs

		total_error += (y - (m * x + b)) ** 2 # loss equation
	total_error = total_error / float(len(data))

def gradient_descent(m_curr, b_curr, data, learning_rate):
	m_grad = 0
	b_grad = 0

	n = len(data)

	for i in range(n):
		x = data.iloc[i].Hours
		y = data.iloc[i].Scores

		m_grad += -(2/n) * x * (y - (m_curr * x + b_curr))
		b_grad += -(2/n) * (y - (m_curr * x + b_curr))

	m = m_curr - m_grad * learning_rate
	b = b_curr - b_grad * learning_rate
	return m,b

m = 0
b = 0

lr = 0.0001
epochs = 500


for i in range(epochs):
	if (i%50 == 0):
		print(f"Epoch: {i}")
	m, b = gradient_descent(m, b, data, lr)

print(m, b)

# Define the x values
x_values = list(range(1,10))

# Create the y values using the model
y_values = [m*x + b for x in x_values]

# Plot the line first
plt.plot(x_values, y_values, color = "red")

# Then plot the scatter
plt.scatter(data.Hours, data.Scores, color = "black")

plt.show()

# import the python libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # MATLAB-like way of plotting
from matplotlib.widgets import Cursor
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# declared variables to use
plotCount: int = 0  # use for track slider movement to clear previous plot
line_1 = (0, 0)  # stores the last slider vertical line plot
# read the csv file containing the dataset
df = pd.read_csv('nba_rookie_data.csv')
# display the data head
print(df.head())
# remove null values from dataframe
df.dropna()
# Display the first five rows
print(df.head())

########################################
# Function to train model
########################################
def train_model(X, y):
    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1 / 3)
    # build the model
    model = GaussianNB()
    # train the model
    model.fit(X_train, y_train)
    # get the model accuracy score
    accuracy = round(model.score(X_test, y_test), 2) * 100
    # store the models performance text
    result_text = 'Our model accuracy is ' + str(round(accuracy)) + '%' + '\n' \
                  + 'Number of mislabeled points out of a total %d points : %d' % (X_test.shape[0],
                                                                                   (y_test != model.predict(
                                                                                       X_test)).sum())
    # return the trained model and performance
    return model, result_text


######################################################
# Building the model using one feature
######################################################
# select the index location of the X feature
X = df.iloc[:, [1]].values
# select the index location of the y feature
y = df.iloc[:, -1].values
# call train model function
model, result_text = train_model(X, y)


# Display the model performance on the console
print(result_text)
fig = plt.figure(figsize=(15, 7))
# display figure suptitle
fig.suptitle('Gaussian Naive Bayes By Umolu John Chukwuemeka (2065655)', color='blue', fontsize=16, fontweight='bold')
# create box to contain the model performance text using the matplotlib.patch.Patch properties
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
# set the figure title
fig.canvas.manager.set_window_title('Coursework Task 3: Gaussian Naive Bayes By Umolu John Chukwuemeka (2065655)')
ax1 = fig.add_subplot(121)
x_range = len(X)
Xs = [i for i in range(x_range)]
Ys = [model.predict_proba([[value]])[0][1] for value in range(x_range)]
# Plot the scatter plot
plt.scatter(X, y)
# Plot the sigmoid
plt.plot(Xs, Ys, color='red')
# set the x-axis label
ax1.set_xlabel('x-axis (Games Played)')
# set the y-axis label
ax1.set_ylabel('y-axis (Target_5Yrs)')
# set the minimum and maximum range for x-axis
ax1.set_xlim(0, np.max(np.array(X)))
# set the minimum and maximum range for y-axis
ax1.set_ylim(-0.2, np.max(np.array(y)) + 0.2)
# display model performance at the bottom of the figure
plot_text1 = fig.text(x=ax1.get_position().x0 - 0.06, y=-0.27, s=result_text, fontsize=10,
                      transform=ax1.transAxes, wrap=True, bbox=props)

######################################################
# Adding more features to the model
######################################################
# select the index location of the X feature
X1 = df.iloc[:, [1, 14]].values
# select the index location of the y feature
y1 = df.iloc[:, -1].values
# call train model function
model1, result_text1 = train_model(X1, y1)
# creating new figure
ax2 = fig.add_subplot(122, projection='3d')
#
ax2.scatter(X1[:, 0], X1[:, 1], y1)
# set the x-axis label
ax2.set_xlabel('x0-axis (Games Played)')
# set the y-axis label
ax2.set_ylabel('x1-axis (Defensive Rebounds)')
# set the z-axis label
ax2.set_zlabel('y-axis (Target_5Yrs)')
# set the minimum and maximum range for x-axis
ax2.set_xlim(0, np.max(np.array(X1[:, 0])))
# set the minimum and maximum range for y-axis
ax2.set_ylim(0, np.max(np.array(X1[:, 1])))
# set the minimum and maximum range for z-axis
ax2.set_zlim(-0.2, np.max(np.array(y1)) + 0.2)
# display result on the figure
plot_text2 = fig.text(x=ax2.get_position().x0 - 0.5, y=-0.27, s=result_text1, fontsize=10,
                      transform=ax2.transAxes, wrap=True, bbox=props)



# move the figure up and sideways so that it's not on top of the slider
fig.subplots_adjust(left=0.2, bottom=0.3)

# plot the text box on the figure
pred_legend = ax1.text(0.49, 0.35, 'To predict: \n-Hover mouse cursor on plot\n-Click mouse to select input',
                       fontsize=12, verticalalignment='top', bbox=props, transform=ax1.transAxes)


# function for input prediction
def update_text(input_data, prediction):
    # make variable visible outside the function
    global pred_legend
    # store text to be displayed
    predict_text = 'X Input value: ' + str(input_data) + '\n' + 'Prediction: ' + str(prediction)
    # check if text box plotted already
    if pred_legend is not None:
        # Remove the text box
        pred_legend.remove()
    # display model prediction at the text box
    pred_legend = ax1.text(0.5, 0.35, predict_text, fontsize=12,
                           verticalalignment='top', bbox=props, transform=ax1.transAxes)
    # Display the prediction on the console
    print(predict_text)
    # save figure image
    fig.savefig('gaussian_naive_bayes.png')
    # return to called location
    return


# function to get mouse clicked position on the plot
def predict_value(event):
    # make variable visible outside the function
    global line_1, plotCount
    # get mouse coordinates
    input_data = event.xdata
    # check if input value is not None
    if input_data is not None:
        # round up input value
        input_data = round(input_data)
        # check if input is greater than 0
        if input_data > 0:
            # predict input
            prediction = model.predict([[input_data]])
            # call text update function
            update_text(input_data, prediction[0])

    # Plot remove routine that removes the previous plotted lines in every 2 counts
    plotCount = plotCount + 1
    if plotCount == 2:
        plotCount = 1
        l1 = line_1.pop(0)
        # remove the vertical and horizontal line tracker
        l1.remove()
    # Plot Vertical Line
    line_1 = ax1.plot([input_data, input_data], [-1.2, np.max(np.array(y)) + 0.2], color='red', linestyle='dashed')
    # Return to called location
    return


# get the mouse hovering position
def graph_hover(event):
    # get mouse coordinates
    input_data = event.xdata
    if input_data is not None:
        ax1.set_title('X Input value: ' + str(round(input_data)))
    # Return to called location
    return


# Detect when mouse is hovered on the graph
mouse_hover = Cursor(ax1, color='r', lw=1, horizOn=False, vertOn=True)
# call function predict_value is called while hovering over the graphs
plt.connect('motion_notify_event', graph_hover)
# call function predict_value is called while clicking over the graphs
plt.connect('button_press_event', predict_value)

# show plot
plt.show()




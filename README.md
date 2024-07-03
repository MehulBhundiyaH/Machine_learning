***Project 1: Data Handling and Analysis (Data Handling, Data Manipulation, Outliers Identification, RMSE, Standardization, Normalization)
Description:
In this project, I worked with landslide-related data from sensors to perform various data analysis and cleaning tasks. The main focus was on handling missing values, performing correlation analysis, identifying outliers, and implementing standardization and normalization techniques.

Learnings:

Data Handling and Cleaning: Managed and cleaned the dataset by addressing missing values through linear interpolation and other techniques.
Data Manipulation: Manipulated the data using Pandas to prepare it for analysis.
Correlation Analysis: Computed Pearson correlation to identify relationships between different sensor attributes.
Outliers Identification: Identified outliers using the quartile range method.
RMSE Calculation: Calculated the Root Mean Squared Error (RMSE) between the original and filled missing values to measure the accuracy of the imputed data.
Standardization and Normalization: Applied standardization and normalization techniques to prepare the data for machine learning models.


***Project 2: Iris Data Analysis and Classification (PCA, Outliers Correction, Data Reconstruction, KNN Classification)
Description:
In this project, I worked with the Iris dataset, performing tasks that include outlier correction, dimensionality reduction using Principal Component Analysis (PCA), data reconstruction, and building a K-Nearest Neighbour (KNN) classifier.

Learnings:

Data Handling and Extraction: Extracted attributes into a matrix and the target attribute into an array for analysis.
Outliers Correction: Identified and replaced outliers in the dataset with the median values of respective attributes.
Principal Component Analysis (PCA): Implemented PCA to reduce the dimensionality of the dataset, facilitating easier visualization and analysis.
Data Visualization: Created scatter plots of the dimension-reduced data, superimposing eigen directions for better understanding.
Data Reconstruction: Reconstructed the original data from the dimension-reduced dataset, and computed the RMSE to assess the reconstruction accuracy.
KNN Classification: Built a KNN classifier from scratch, evaluated its performance using a confusion matrix, and visualized the results for interpretability.


***Project 3: Iris Data Classification Using Bayesian Methods (PCA, Univariate and Multivariate Gaussian Distributions)
Description:
In this project, I applied Bayesian classification techniques to the Iris dataset, using both univariate and multivariate Gaussian distributions. The tasks involved dimension reduction, model building, and performance evaluation on both reduced and original datasets.

Learnings:

Dimensionality Reduction:

Reduced the dimensions of the Iris dataset to one using PCA, facilitating simpler model construction and visualization.
Bayes Classifier - Univariate Gaussian:

Built a Bayes classifier on the one-dimensional train set by estimating parameters of univariate Gaussian distributions.
Classified test samples by computing their likelihood for each class and implemented the steps without using built-in classification functions.
Evaluated model performance by computing the confusion matrix and accuracy, providing insights into the classifier's effectiveness.
Bayes Classifier - Multivariate Gaussian:

Constructed a Bayes classifier on the four-dimensional train set, estimating parameters of multivariate Gaussian distributions.
Classified test samples using their likelihoods computed with scipy.stats functions.
Evaluated this model's performance using confusion matrix and accuracy metrics.
Performance Comparison:

Computed and analyzed the difference in accuracy between the models built using the original four-dimensional data and the dimension-reduced data, highlighting the impact of dimensionality reduction on classification performance.


***Project 4: Linear and Polynomial Regression on Abalone Dataset
Description:
This project involved predicting the age of Abalones using various measurements by building linear and polynomial regression models. The task aimed to simplify the age prediction process for these marine snails.

Learnings:

Data Handling and Splitting:

Loaded and split the Abalone dataset into training (70%) and testing (30%) sets using Pandas and scikit-learn, ensuring reproducibility with a fixed random state.
Saved the split datasets as abalone_train.csv and abalone_test.csv.
Linear Regression:

Identified the attribute with the highest Pearson correlation with the target attribute "Rings".
Developed a simple linear regression model using this attribute to predict the number of rings, without relying on built-in classification functions.
Plotted the best-fit line for the training data, visually depicting the relationship between the selected attribute and the number of rings.
Evaluated the model's prediction accuracy on both training and test sets using Root Mean Squared Error (RMSE), providing insight into the model's performance.
Created a scatter plot comparing actual vs. predicted rings on the test data.
Polynomial Regression:

Used the attribute with the highest Pearson correlation to build nonlinear regression models with polynomial curve fitting (degrees 2, 3, 4, 5).
Implemented polynomial regression from scratch, avoiding built-in classification functions.
Assessed prediction accuracy for different polynomial degrees on both training and test datasets using RMSE.
Plotted bar graphs of RMSE against polynomial degrees, facilitating a comparison of model performance across different complexities.
Visualized the best-fit curve on the training data, identifying the optimal polynomial degree based on the minimum test RMSE.


***Project 5: Image Classification using FCNN and CNN on CIFAR-3 Class Dataset
Description:
This project focused on classifying images of aeroplanes, cars, and birds using Fully Connected Neural Networks (FCNN) and Convolutional Neural Networks (CNN). The dataset included RGB images of these three classes, and the task involved implementing both FCNN and CNN models using Keras.

Learnings:

Data Loading and Visualization:

Developed a custom function to load images from the Train and Test datasets without using any in-built functions.
Visualized sample images from the dataset using matplotlib to understand the data distribution.
Fully Connected Neural Network (FCNN):

Data Preparation:
Split the train dataset into training and validation sets, retaining 10% for validation using scikit-learn.
Resized images into vectors and normalized the pixel values.
Model Architecture:
Built an FCNN model with three hidden layers (256, 128, and 64 neurons) using ReLU activation functions.
Added an output layer to classify images into three classes (aeroplane, car, bird).
Training and Evaluation:
Trained the model for 500 epochs with a batch size of 200 using the Adam optimizer and an appropriate loss function.
Plotted epoch-wise training and validation accuracies to visualize the learning process.
Evaluated the model on the test dataset, printing test loss and accuracy.
Convolutional Neural Network (CNN):

Data Preparation:
Used the same custom function for loading images as in the FCNN task.
Normalized images similarly to the FCNN task.
Model Architecture:
Designed a CNN with two pairs of convolutional layers (64 filters of size 3x3) followed by max-pooling layers.
Added two more pairs of convolutional layers (128 filters of size 3x3) followed by max-pooling layers.
Included a flattening layer to convert feature maps into vectors, followed by two fully connected layers (512 and 100 neurons).
Output layer designed to classify images into three classes.
Training and Evaluation:
Trained the CNN model for 50 epochs with a batch size of 200 using the Adam optimizer and an appropriate loss function.
Saved the trained model.
Plotted epoch-wise training and validation accuracies to track model performance.
Loaded the saved model and used it to predict class probabilities for test images.
Computed and displayed the confusion matrix and test accuracy.

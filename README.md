<h1>Machine Learning Applications on MNIST Data Set</h1>
<p>The  code is a comprehensive implementation of machine learning techniques using the MNIST dataset,
which consists of handwritten digits. 
The MNIST dataset is loaded, and features (images) and labels (digits) are extracted. 
The dataset is split into training and testing sets.
A Stochastic Gradient Descent (SGD) classifier is trained and evaluated using cross-validation, with performance metrics such as precision, recall, and F1 score calculated.
The code also explores the relationship between precision and recall at different thresholds, plots the ROC curve, and compares the performance of a Random Forest classifier.
Additionally, the use of Support Vector Machines (SVM) and K-Nearest Neighbors (KNN) for multi-label classification, 
and shows how to handle noisy data by cleaning it with KNN.
Overall, the code provides a thorough exploration of various classification techniques and evaluation methods for the MNIST dataset.</p>

<h3>About MNIST Data set</h3>
<p>The MNIST database (Modified National Institute of Standards and Technology) is a 
large database of handwritten digits that is commonly used for training various image 
processing systems.The data set contains 70,000 small images of digits 
handwritten by high school students and employees of the US Census Bureau. Each image is 
labeled with the digit it represents. This set has been studied so much that it is often called the 
“hello world” of Machine Learning.Datasets loaded by Scikit-Learn. There are 70,000 images, and each image has 784 features. 
This is because each image is 28 × 28 1pixels, and each feature simply represents one pixel’s 
intensity, from 0 (white) to 255 (black).</p>

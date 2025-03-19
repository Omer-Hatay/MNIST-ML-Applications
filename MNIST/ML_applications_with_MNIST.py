# MNIST 
# In[1]: 
import numpy as np 
import os 
np.random.seed(42) 
# To plot pretty figures 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
mpl.rc('axes', labelsize=14) 
mpl.rc('xtick', labelsize=12) 
mpl.rc('ytick', labelsize=12) 
# Where to save the figures 
PROJECT_ROOT_DIR = "." 
CHAPTER_ID = "classification" 
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID) 
os.makedirs(IMAGES_PATH, exist_ok=True) 
def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300): 
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension) 
    print("Saving figure", fig_id) 
    if tight_layout: 
        plt.tight_layout() 
    plt.savefig(path, format=fig_extension, dpi=resolution) 
# In[2]: 
from sklearn.datasets import fetch_openml 
mnist = fetch_openml('mnist_784', version=1, as_frame=False) 
mnist.keys() 
# In[3]: 
X, y = mnist["data"], mnist["target"] 
print(X.shape) 
# In[4]: 
print(y.shape) 
# In[5]: 
import matplotlib as mpl 
import matplotlib.pyplot as plt 

some_digit = X[0] 
some_digit_image = some_digit.reshape(28, 28) 
plt.imshow(some_digit_image, cmap=mpl.cm.binary) 
plt.axis("off") 
save_fig("some_digit_plot") 
plt.show() 
# In[6]: 
print(y[0]) 
# In[7]: 
y = y.astype(np.uint8) 
# In[8]: 
def plot_digit(data): 
    image = data.reshape(28, 28) 
    plt.imshow(image, cmap = mpl.cm.binary,interpolation="nearest") 
    plt.axis("off") 
# In[9]: 
def plot_digits(instances, images_per_row=10, **options): 
    size = 28 
    images_per_row = min(len(instances), images_per_row) 
    # This is equivalent to n_rows = ceil(len(instances) / images_per_row): 
    n_rows = (len(instances) - 1) // images_per_row + 1 
    # Append empty images to fill the end of the grid, if needed: 
    n_empty = n_rows * images_per_row - len(instances) 
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0) 
    # Reshape the array so it's organized as a grid containing 28×28 images: 
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size)) 
    # Combine axes 0 and 2 (vertical image grid axis, and vertical image axis), 
    # and axes 1 and 3 (horizontal axes). We first need to move the axes that we 
    # want to combine next to each other, using transpose(), and only then we 
    # can reshape: 
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size, 
    images_per_row * size) 
    # Now that we have a big image, we just need to show it: 
    plt.imshow(big_image, cmap = mpl.cm.binary, **options) 
    plt.axis("off") 
    # In[10]: 
plt.figure(figsize=(9,9)) 
example_images = X[:100] 

plot_digits(example_images, images_per_row=10) 
save_fig("more_digits_plot") 
plt.show() 
# In[11]: 
print(y[0]) 
# In[12]: 
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:] 
# In[13]: 
y_train_5 = (y_train == 5) 
y_test_5 = (y_test == 5) 
# In[14]: 
from sklearn.linear_model import SGDClassifier 
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42) 
print(sgd_clf.fit(X_train, y_train_5)) 
# In[15]: 
#detection operation. 
print(sgd_clf.predict([some_digit])) 
# In[16]: 
from sklearn.model_selection import cross_val_score 
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")) 
# In[17]: 
from sklearn.model_selection import StratifiedKFold 
from sklearn.base import clone 
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) 
for train_index, test_index in skfolds.split(X_train, y_train_5): 
    clone_clf = clone(sgd_clf) 
    X_train_folds = X_train[train_index] 
    y_train_folds = y_train_5[train_index] 
    X_test_fold = X_train[test_index] 
    y_test_fold = y_train_5[test_index] 
    clone_clf.fit(X_train_folds, y_train_folds) 
    y_pred = clone_clf.predict(X_test_fold) 
    n_correct = sum(y_pred == y_test_fold) 
    (n_correct / len(y_pred)) 
# In[18]: 
from sklearn.base import BaseEstimator 
class Never5Classifier(BaseEstimator): 

    def fit(self, X, y=None): 
        pass 
    def predict(self, X): 
        return np.zeros((len(X), 1), dtype=bool) 
# In[19]: 
never_5_clf = Never5Classifier() 
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")) 
# In[20]: 
from sklearn.model_selection import cross_val_predict 
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3) 
# In[21]: 
from sklearn.metrics import confusion_matrix 
print(confusion_matrix(y_train_5, y_train_pred)) 
# In[22]: 
print("\n\n") 
y_train_perfect_predictions = y_train_5  # pretend we reached perfection 
confusion_matrix(y_train_5, y_train_perfect_predictions) 
# In[23]: 
from sklearn.metrics import precision_score, recall_score 
print(precision_score(y_train_5, y_train_pred) ) 
print(recall_score(y_train_5, y_train_pred) ) 
# In[24]: 
cm = confusion_matrix(y_train_5, y_train_pred) 
cm[1, 1] / (cm[0, 1] + cm[1, 1]) 
# In[25]: 
recall_score(y_train_5, y_train_pred) 
# In[26]: 
cm[1, 1] / (cm[1, 0] + cm[1, 1]) 
# In[27]: 
from sklearn.metrics import f1_score 
print(f1_score(y_train_5, y_train_pred)) 
# In[28]: 
cm[1, 1] / (cm[1, 1] + (cm[1, 0] + cm[0, 1]) / 2) 

# In[29]: 
y_scores = sgd_clf.decision_function([some_digit]) 
print("y_scores= ",y_scores) 
# In[30]: 
threshold = 0 
y_some_digit_pred = (y_scores > threshold) 
# In[31]: 
print(y_some_digit_pred) 
# In[32]: 
threshold = 8000 
y_some_digit_pred = (y_scores > threshold) 
y_some_digit_pred 
# In[33]: 
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,method="decision_function") 
# In[34]: 
from sklearn.metrics import precision_recall_curve 
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores) 
# In[35]: 
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds): 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2) 
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2) 
    plt.legend(loc="center right", fontsize=16)   
    plt.xlabel("Threshold", fontsize=16)   
    plt.grid(True)               
    plt.axis([-50000, 50000, 0, 1])       
    
recall_90_precision = recalls[np.argmax(precisions >= 0.90)] 
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] 
plt.figure(figsize=(8, 4)) 
plot_precision_recall_vs_threshold(precisions, recalls, thresholds) 
plt.plot([threshold_90_precision, threshold_90_precision], [0., 0.9], "r:") 
plt.plot([-50000, threshold_90_precision], [0.9, 0.9], "r:") 
plt.plot([-50000, threshold_90_precision], [recall_90_precision, recall_90_precision], "r:") 
plt.plot([threshold_90_precision], [0.9], "ro") 
plt.plot([threshold_90_precision], [recall_90_precision], "ro") 
save_fig("precision_recall_vs_threshold_plot") 
plt.show() 
25 
# In[36]: 
(y_train_pred == (y_scores > 0)).all() 
# In[37]: 
def plot_precision_vs_recall(precisions, recalls): 
    plt.plot(recalls, precisions, "b-", linewidth=2) 
    plt.xlabel("Recall", fontsize=16) 
    plt.ylabel("Precision", fontsize=16) 
    plt.axis([0, 1, 0, 1]) 
    plt.grid(True) 
plt.figure(figsize=(8, 6)) 
plot_precision_vs_recall(precisions, recalls) 
plt.plot([recall_90_precision, recall_90_precision], [0., 0.9], "r:") 
plt.plot([0.0, recall_90_precision], [0.9, 0.9], "r:") 
plt.plot([recall_90_precision], [0.9], "ro") 
save_fig("precision_vs_recall_plot") 
plt.show() 
# In[39]: 
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)] 
# In[40]: 
#threshold_90_precision 
# In[41]: 
#To make predictions instead of calling the classifier’s predict() method, we can run this code 
y_train_pred_90 = (y_scores >= threshold_90_precision) 
# In[42]: 
print(precision_score(y_train_5, y_train_pred_90)) 
# In[43]: 
print(recall_score(y_train_5, y_train_pred_90)) 
# In[44]: 
from sklearn.metrics import roc_curve 
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores) 
# In[45]: 
def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=2, label=label) 

    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal 
    plt.axis([0, 1, 0, 1])     
                              
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)  
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)    
    plt.grid(True)               
                            
plt.figure(figsize=(8, 6)) 
plot_roc_curve(fpr, tpr) 
                                    
fpr_90 = fpr[np.argmax(tpr >= recall_90_precision)]
            
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:")   
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:")  
plt.plot([fpr_90], [recall_90_precision], "ro")         
       
save_fig("roc_curve_plot")       
plt.show() 
# In[46]: 
from sklearn.metrics import roc_auc_score 
print(roc_auc_score(y_train_5, y_scores)) 
# In[47]: 
                              
from sklearn.ensemble import RandomForestClassifier 
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42) 
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, 
cv=3,method="predict_proba") 
# In[48]: 
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class 
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest) 
# In[49]: 
recall_for_forest = tpr_forest[np.argmax(fpr_forest >= fpr_90)] 
plt.figure(figsize=(8, 6)) 
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD") 
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest") 
plt.plot([fpr_90, fpr_90], [0., recall_90_precision], "r:") 
plt.plot([0.0, fpr_90], [recall_90_precision, recall_90_precision], "r:") 
plt.plot([fpr_90], [recall_90_precision], "ro") 
plt.plot([fpr_90, fpr_90], [0., recall_for_forest], "r:") 
plt.plot([fpr_90], [recall_for_forest], "ro") 
plt.grid(True) 
plt.legend(loc="lower right", fontsize=16) 
save_fig("roc_curve_comparison_plot") 

plt.show() 
# In[50]: 
print(roc_auc_score(y_train_5, y_scores_forest)) 
print("hello world") 
# In[51]: 
y_train_pred_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3) 
precision_score(y_train_5, y_train_pred_forest) 
# In[52]: 
recall_score(y_train_5, y_train_pred_forest) 
# In[53]: 
from sklearn.svm import SVC 
svm_clf = SVC(gamma="auto", random_state=42) 
svm_clf.fit(X_train[:1000], y_train[:1000]) # y_train, not y_train_5 
print(svm_clf.predict([some_digit])) 
# In[54]: 
some_digit_scores = svm_clf.decision_function([some_digit]) 
print(some_digit_scores) 
# In[55]: 
print(np.argmax(some_digit_scores)) 
# In[56]: 
print(svm_clf.classes_) 
# In[57]: 
print(svm_clf.classes_[5]) 
# In[58]: 
from sklearn.multiclass import OneVsRestClassifier 
ovr_clf = OneVsRestClassifier(SVC(gamma="auto", random_state=42)) 
ovr_clf.fit(X_train[:1000], y_train[:1000]) 
print(ovr_clf.predict([some_digit])) 
# In[59]: 
print(len(ovr_clf.estimators_)) 
# In[60]: 
sgd_clf.fit(X_train, y_train) 
print(sgd_clf.predict([some_digit])) 

# In[61]: 
print(sgd_clf.decision_function([some_digit])) 
# In[62]: 
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")) 
# In[63]: 
from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64)) 
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")) 
# In[64]: 
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3) 
conf_mx = confusion_matrix(y_train, y_train_pred) 
print(conf_mx) 
# In[65]: 
# since sklearn 0.22, you can use sklearn.metrics.plot_confusion_matrix() 
def plot_confusion_matrix(matrix): 
    #If you prefer color and a colorbar 
    fig = plt.figure(figsize=(8,8)) 
    ax = fig.add_subplot(111) 
    cax = ax.matshow(matrix) 
    fig.colorbar(cax) 
# In[66]: 
plt.matshow(conf_mx, cmap=plt.cm.gray) 
save_fig("confusion_matrix_plot", tight_layout=False) 
plt.show() 
# In[67]: 
row_sums = conf_mx.sum(axis=1, keepdims=True) 
norm_conf_mx = conf_mx / row_sums 
# In[68]: 
#Fill the diagonal with zeros to keep only the errors, and plot the result 
np.fill_diagonal(norm_conf_mx, 0) 
plt.matshow(norm_conf_mx, cmap=plt.cm.gray) 
save_fig("confusion_matrix_errors_plot", tight_layout=False) 
plt.show() 
# In[69]: 

cl_a, cl_b = 3, 5 
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)] 
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)] 
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)] 
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)] 
plt.figure(figsize=(8,8)) 
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5) 
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5) 
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5) 
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5) 
save_fig("error_analysis_digits_plot") 
plt.show() 
# In[70]: 
from sklearn.neighbors import KNeighborsClassifier 
y_train_large = (y_train >= 7) 
y_train_odd = (y_train % 2 == 1) 
y_multilabel = np.c_[y_train_large, y_train_odd] 
knn_clf = KNeighborsClassifier() 
knn_clf.fit(X_train, y_multilabel) 
# In[71]: 
print(knn_clf.predict([some_digit])) 
# In[72]: 
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3) 
print(f1_score(y_multilabel, y_train_knn_pred, average="macro")) 
# In[73]: 
noise = np.random.randint(0, 100, (len(X_train), 784)) 
X_train_mod = X_train + noise 
noise = np.random.randint(0, 100, (len(X_test), 784)) 
X_test_mod = X_test + noise 
y_train_mod = X_train 
y_test_mod = X_test 
some_index = 0 
plt.subplot(121); plot_digit(X_test_mod[some_index]) 
plt.subplot(122); plot_digit(y_test_mod[some_index]) 
save_fig("noisy_digit_example_plot") 
plt.show() 
# In[74]: 
knn_clf.fit(X_train_mod, y_train_mod) 
clean_digit = knn_clf.predict([X_test_mod[some_index]]) 
plot_digit(clean_digit) 
save_fig("cleaned_digit_example_plot")
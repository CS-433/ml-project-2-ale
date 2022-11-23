from models import *
from plots import *
from helpers import *

x_train, y_train = load_csv_data(r'Hamid_ML4Science_ALE/train_beta0.1')
x_test, y_test = load_csv_data(r'Hamid_ML4Science_ALE/test_beta0.1')

# make the prediction
model, pred, train_accuracy, pred_score = SVM_predict(x_train, y_train, x_test, C=0.1, gamma=10, kernel='rbf')
# calculate accuracy
test_accuracy = accuracy_score(y_test, pred)
# plot similarity matrix
# plot_similarity_matrix(pred_score, save_fig=True, show_values=True)
plot_confusion_matrix(model, x_test, y_test, ids=y_train)

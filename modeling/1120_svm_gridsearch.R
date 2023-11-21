# Load the necessary libraries
install.packages("xgboost")
install.packages("e1071") #SVM ��Ű��
library(e1071)
library(xgboost)
library(glmnet)

library(readxl)
# Read in the data
data <- read_excel('C:/Users/hyeseung/Desktop/4-2/ML/����������/��ٽ���1113.xlsx')
data = as.matrix(data)
head(data)


# Set the windoww size and prediction horizon
windoww <- nrow(data)-100 
predict_ahead <- 1

# Initialize vectors to store predictions and errors



tune_grid <- expand.grid(C = c(0.1, 1, 10), kernel = c("linear", "radial"))

# �׸��� ��ġ ����
svm_tune <- tune(svm, y_train ~ ., data = data.frame(X_train, y_train),
                 ranges = tune_grid, scale = FALSE)

# ���� �Ķ���� Ȯ��
best_parameters <- svm_tune$best.parameters
print(best_parameters)

# ���� �Ķ���͸� ����Ͽ� SVM �� �Ʒ�
final_svm_model <- svm(y_train ~ ., data = data.frame(X_train, y_train),
                       kernel = best_parameters$kernel, cost = best_parameters$C)

# �׽�Ʈ �����Ϳ� ���� ����
predictions_svm <- predict(final_svm_model, newdata = data.frame(X_test))

# ���� ���� ���
err.mod <- c(err.mod, (y_test - predictions_svm)^2)
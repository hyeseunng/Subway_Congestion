# Load the necessary libraries
install.packages("xgboost")
install.packages("e1071") #SVM 패키지
library(e1071)
library(xgboost)
library(glmnet)

library(readxl)
# Read in the data
data <- read_excel('C:/Users/hyeseung/Desktop/4-2/ML/최종데이터/출근승차1113.xlsx')
data = as.matrix(data)
head(data)


# Set the windoww size and prediction horizon
windoww <- nrow(data)-100 
predict_ahead <- 1

# Initialize vectors to store predictions and errors



tune_grid <- expand.grid(C = c(0.1, 1, 10), kernel = c("linear", "radial"))

# 그리드 서치 수행
svm_tune <- tune(svm, y_train ~ ., data = data.frame(X_train, y_train),
                 ranges = tune_grid, scale = FALSE)

# 최적 파라미터 확인
best_parameters <- svm_tune$best.parameters
print(best_parameters)

# 최적 파라미터를 사용하여 SVM 모델 훈련
final_svm_model <- svm(y_train ~ ., data = data.frame(X_train, y_train),
                       kernel = best_parameters$kernel, cost = best_parameters$C)

# 테스트 데이터에 대한 예측
predictions_svm <- predict(final_svm_model, newdata = data.frame(X_test))

# 예측 오차 계산
err.mod <- c(err.mod, (y_test - predictions_svm)^2)
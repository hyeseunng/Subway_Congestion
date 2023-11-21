# Load the necessary libraries
library(glmnet)
install.packages("e1071") #SVM 패키지
library(e1071)
library(readxl)
# Read in the data
data <- read_excel('C:/Users/hyeseung/Desktop/4-2/ML/최종데이터/출근승차1113.xlsx')
data = as.matrix(data)
head(data)

# Set the windoww size and prediction horizon
windoww <- nrow(data)-100 
predict_ahead <- 1

# Initialize vectors to store predictions and errors

rmse.mat = NULL 
# Loop over the data using a rolling windoww approach
for (i in windoww:530) {
  print(i)
  ### ready 
  err.mod = NULL 
  ### partition 
  train_data <- data[1:i, ]
  head(train_data)
  test_data <- data[(i + 1):(i + predict_ahead),,drop=F]
  head(test_data)
  X_train <- train_data[,!(colnames(train_data)%in%c('승하차수','상대습도'))]
  y_train <- train_data[,colnames(train_data)=='승하차수']
  X_test <- test_data[,!(colnames(test_data)%in% c('승하차수','상대습도')),drop=F]
  y_test <- test_data[,colnames(test_data)=='승하차수',drop=F]
  
  ################################### glm 
  fit_glm = glm(y_train~.,data=data.frame(y_train,X_train))
  err.mod = c(err.mod,(y_test-predict.glm(fit_glm,newdata=data.frame(X_test)))^2)
  ############################$###### lasso 
  fit_lasso <- glmnet(X_train, y_train, alpha = 1)  
  # option 1  
  cv_fit <- cv.glmnet(X_train, y_train, alpha = 1)
  # option 2
  e.mat = NULL 
  for(j in 1:30){
    ttrain_data <- data[1:(i-j), ]
    ttest_data <- data[(i-j+1):(i-j+predict_ahead),,drop=F]
    X_ttrain <- ttrain_data[,!(colnames(ttrain_data)%in%c('승하차수','상대습도'))]
    y_ttrain <- ttrain_data[,colnames(ttrain_data)=='승하차수']
    X_ttest <- ttest_data[,!(colnames(ttest_data)%in% c('승하차수','상대습도'))]
    y_ttest <- ttest_data[,colnames(ttest_data)=='승하차수']
    fitt_lasso <- glmnet(X_ttrain, y_ttrain, alpha = 1,lambda=fit_lasso$lambda)
    e.mat = rbind(e.mat,y_ttest-predict.glmnet(fitt_lasso,newx=X_ttest))
  }
  opt = which.min(colMeans(e.mat^2))
  pre = predict.glmnet(fit_lasso,newx=X_test)[,opt]
  err.mod = c(err.mod,(y_test-pre)^2)
  ############################$###### ridge 
  fit_lasso <- glmnet(X_train, y_train, alpha = 0.01)  
  # opttion 1  
  cv_fit <- cv.glmnet(X_train, y_train, alpha = 0.01)
  # opttion 2
  e.mat = NULL 
  for(j in 1:30){
    ttrain_data <- data[1:(i-j), ]
    ttest_data <- data[(i-j+1):(i-j+predict_ahead),,drop=F]
    X_ttrain <- ttrain_data[,!(colnames(ttrain_data)%in%c('승하차수','상대습도'))]
    y_ttrain <- ttrain_data[,colnames(ttrain_data)=='승하차수']
    X_ttest <- ttest_data[,!(colnames(ttest_data)%in% c('승하차수','상대습도'))]
    y_ttest <- ttest_data[,colnames(ttest_data)=='승하차수']
    fitt_lasso <- glmnet(X_ttrain, y_ttrain, alpha = 0.01,lambda=fit_lasso$lambda)
    pre = predict.glmnet(fitt_lasso,newx=X_ttest)
    e.mat = rbind(e.mat,y_ttest-pre)
  }
  opt = which.min(colMeans(e.mat^2))
  pre = predict.glmnet(fit_lasso,newx=X_test)[,opt]
  err.mod = c(err.mod,(y_test-pre)^2)
  ####################################################### svm 
  
  fit_svm <- svm(X_train,y_train)
  ######그리드 서치 해서 최적 파라미터 찾기!
  
  pre <- predict(fit_svm, newdata=X_test)
  err.mod = c(err.mod,(y_test-pre)^2)
  
  ####################################################### boost 
  
  #pre = predict.glmnet(fit_lasso,newx=X_test)[,opt]
  #err.mod = c(err.mod,(y_test-pre)^2)
  
  #### final 
  rmse.mat = rbind(rmse.mat,err.mod)
}

boxplot(sqrt(rmse.mat),ylim=c(0,500))
sqrt(colMeans(rmse.mat))

boxplot(sqrt(rmse.mat),
        ylim = c(0, 500),
        names = c("GLM", "Lasso", "Ridge", "SVM"),
        main = "Boxplot of Square Root of RMSE Values",
        xlab = "Models",
        ylab = "Square Root of RMSE")

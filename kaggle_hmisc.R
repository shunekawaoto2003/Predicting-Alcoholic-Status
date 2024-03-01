install.packages("randomForest")
library(randomForest)
# library(ggplot2)
install.packages("caret")
library(caret)
install.packages("glmnet")
library(glmnet)
install.packages("factoextra")
# library(factoextra)
library(e1071)
install.packages("xgboost")
library(xgboost)


test <-  read.csv('TestSAData2NoY.csv')
train <- read.csv('TrainSAData2.csv')

test <- test[, -1]
train <- train[, -1]

train$Alcoholic.Status <- as.factor(train$Alcoholic.Status)
train$sex <- as.factor(train$sex)
train$hear_left <- as.factor(train$hear_left)
train$hear_right <- as.factor(train$hear_right)
train$BMI.Category <- as.factor(train$BMI.Category)
train$AGE.Category <- as.factor(train$AGE.Category)
train$Smoking.Status <- as.factor(train$Smoking.Status)

test$sex <- as.factor(test$sex)
test$hear_left <- as.factor(test$hear_left)
test$hear_right <- as.factor(test$hear_right)
test$BMI.Category <- as.factor(test$BMI.Category)
test$AGE.Category <- as.factor(test$AGE.Category)
test$Smoking.Status <- as.factor(test$Smoking.Status)

install.packages("Hmisc")
library(Hmisc)
new_train <- aregImpute(~Alcoholic.Status+sex+age+height+weight+waistline+sight_left+sight_right+hear_left+hear_right+SBP+DBP+BLDS
                        +tot_chole+HDL_chole+LDL_chole+triglyceride+hemoglobin+urine_protein+serum_creatinine+SGOT_AST+SGOT_ALT+
                          gamma_GTP+BMI+BMI.Category+AGE.Category+Smoking.Status, data=train, n.impute=5)
new_train <- impute.transcan(new_train, imputation=1, data=train, list.out=TRUE)
new_train <- as.data.frame(new_train)

new_test <- aregImpute(~sex+age+height+weight+waistline+sight_left+sight_right+hear_left+hear_right+SBP+DBP+BLDS
                       +tot_chole+HDL_chole+LDL_chole+triglyceride+hemoglobin+urine_protein+serum_creatinine+SGOT_AST+SGOT_ALT+
                         gamma_GTP+BMI+BMI.Category+AGE.Category+Smoking.Status, data=test, n.impute=5)
new_test <- impute.transcan(new_test, imputation=1, data=test, list.out=TRUE)
new_test <- as.data.frame(new_test)

write.csv(new_train, "C:\\Users\\kawao\\Documents\\UCLA\\2023 Fall\\Stats 101C\\Kaggle Project\\hmisc_train.csv", row.names=FALSE)
write.csv(new_test, "C:\\Users\\kawao\\Documents\\UCLA\\2023 Fall\\Stats 101C\\Kaggle Project\\hmisc_test.csv", row.names=FALSE)

setwd("~/UCLA/2023 Fall/Stats 101C/Kaggle Project")
new_train <- read.csv("hmisc_train.csv")
new_test <- read.csv("hmisc_test.csv")
new_train$Alcoholic.Status <- as.factor(new_train$Alcoholic.Status)
new_train$sex <- as.factor(new_train$sex)
new_train$hear_left <- as.factor(new_train$hear_left)
new_train$hear_right <- as.factor(new_train$hear_right)
new_train$BMI.Category <- as.factor(new_train$BMI.Category)
new_train$AGE.Category <- as.factor(new_train$AGE.Category)
new_train$Smoking.Status <- as.factor(new_train$Smoking.Status)

new_test$sex <- as.factor(new_test$sex)
new_test$hear_left <- as.factor(new_test$hear_left)
new_test$hear_right <- as.factor(new_test$hear_right)
new_test$BMI.Category <- as.factor(new_test$BMI.Category)
new_test$AGE.Category <- as.factor(new_test$AGE.Category)
new_test$Smoking.Status <- as.factor(new_test$Smoking.Status)


## LASSO variable selection
X <- model.matrix(~ . -Alcoholic.Status -1, data = new_train)
y <- new_train$Alcoholic.Status
X <- scale(X)

set.seed(123)
cv_fit <- cv.glmnet(X, y, family = "binomial", alpha = 1)
best_lambda <- cv_fit$lambda.min
lasso_model <- glmnet(X, y, family = "binomial", alpha = 1, lambda = best_lambda)

coef(lasso_model, s = best_lambda)

rf_full_mtry2_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=2)
rf_full_mtry3_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=3)
rf_full_mtry4_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=4)     # best score: 0.72883
rf_full_mtry5_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=5)
rf_full_mtry6_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=6)
rf_full_mtry7_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=7)
rf_full_mtry8_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=8)
rf_full_mtry9_hmisc <- randomForest(Alcoholic.Status ~ ., data=new_train, mtry=9)

output_mtry2 <- predict(rf_full_mtry2_hmisc, new_test)
output_mtry3 <- predict(rf_full_mtry3_hmisc, new_test)
output_mtry4 <- predict(rf_full_mtry4_hmisc, new_test)
output_mtry5 <- predict(rf_full_mtry5_hmisc, new_test)
output_mtry6 <- predict(rf_full_mtry6_hmisc, new_test)
output_mtry7 <- predict(rf_full_mtry7_hmisc, new_test)
output_mtry8 <- predict(rf_full_mtry8_hmisc, new_test)
output_mtry9 <- predict(rf_full_mtry9_hmisc, new_test)

write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry2), file='output_mtry2.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry3), file='output_mtry3.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry4), file='output_mtry4.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry5), file='output_mtry5.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry6), file='output_mtry6.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry7), file='output_mtry7.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry8), file='output_mtry8.csv', row.names = FALSE)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=output_mtry9), file='output_mtry9.csv', row.names = FALSE)


## xgboost
set.seed(123)
xgb_m1 <- train(Alcoholic.Status ~ ., data=new_train, method="xgbTree",
                trControl = trainControl("cv", number=10))
pred_xgb <- predict(xgb_m1, new_test)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb), file='pred_xgb.csv', row.names = FALSE)
# kaggle score = 0.73176

xgbGrid <- expand.grid(
  nrounds = c(100, 150),
  max_depth = c(3, 6),
  eta = c(0.01, 0.1),
  gamma = c(0, 0.1),
  colsample_bytree = c(0.5, 0.8),
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.8)
)
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
set.seed(123)
xgb_model <- train(Alcoholic.Status ~ ., 
                   data = new_train, 
                   method = "xgbTree",
                   trControl = train_control,
                   tuneGrid = xgbGrid)
pred_xgb2 <- predict(xgb_model, new_test)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb2), file='pred_xgb2.csv', row.names = FALSE)
# kaggle score 0.73053


# CV to find best learning rate
# CV to find best max tree depth
X_train <- model.matrix(Alcoholic.Status ~ ., data=new_train)
X_test <- model.matrix(~., data=new_test)
X_train <- X_train[,-1]
X_test <- X_test[,-1]
y <- as.numeric(new_train$Alcoholic.Status) - 1

xgb_param <- xgboost(data=X_train,
                     label=y,
                     eta=0.01,
                     max_depth=20,
                     nround=500,
                     objective="binary:logistic")
pred_xgb4 <- predict(xgb_param, X_test, iterationrange = c(1:500), xgb_param$best_iteration)
pred_xgb4 <- ifelse(pred_xgb4 > 0.5, "Y", "N")
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb4), file='pred_xgb4.csv', row.names = FALSE)
# kaggle score 0.66526

etas <- seq(0.1, 0.5, by=0.05)
depths <- seq(2, 20, by=2)
results <- expand.grid(eta = etas, max_depth = depths, error = NA)
for(i in 1:nrow(results)) {
  params <- list(
    objective = "binary:logistic",
    eta = results$eta[i],
    max_depth = results$max_depth[i],
    label=y,
    nthread= -1
  )
  cv <- xgb.cv(params = params, 
               data = X_train, 
               nrounds = 50, 
               nfold = 10, 
               metrics = "error", 
               early_stopping_rounds = 10,
               verbose = FALSE,
               label=y,
               )
  
  results$error[i] <- min(cv$evaluation_log$test_error_mean)
}
best_params <- results[which.min(results$error),]
best_params
# eta = 0.25, max_depth=4

xgb_cv <- xgboost(data=X_train,
                     label=y,
                     eta=0.25,
                     max_depth=4,
                     nround=500,
                     objective="binary:logistic")
pred_xgb5 <- predict(xgb_param, X_test, iterationrange = c(1:500), xgb_param$best_iteration)
pred_xgb5 <- ifelse(pred_xgb4 > 0.5, "Y", "N")
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb5), file='pred_xgb5.csv', row.names = FALSE)
# kaggle score 0.50046


xgbGrid2 <- expand.grid(
  nrounds = 500,
  max_depth = 4,
  eta = .25,
  gamma = c(0, 0.1),
  colsample_bytree = c(0.5, 0.8),
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.8)
)
train_control <- trainControl(method = "cv", number = 10, allowParallel = TRUE)
set.seed(123)
xgb_model2 <- train(Alcoholic.Status ~ ., 
                   data = new_train, 
                   method = "xgbTree",
                   trControl = train_control,
                   tuneGrid = xgbGrid2)
pred_xgb6 <- predict(xgb_model2, new_test)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb6), file='pred_xgb6.csv', row.names = FALSE)
# kaggle score 0.72363


set.seed(123)
xgb_cv20 <- train(Alcoholic.Status ~ ., data=new_train, method="xgbTree",
                trControl = trainControl("cv", number=20))
pred_xgb20 <- predict(xgb_cv20, new_test)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb20), file='pred_xgb20.csv', row.names = FALSE)
# kaggle score 0.73113


xgbGrid3 <- expand.grid(
  nrounds = c(100, 150),
  max_depth = seq(1, 6, by = 1),
  eta = seq(0.05, 0.5, by=0.05),
  gamma = c(0, 0.1),
  colsample_bytree = c(0.5, 0.8),
  min_child_weight = c(1, 3),
  subsample = c(0.5, 0.8)
)
train_control3 <- trainControl(method = "cv", number = 20, allowParallel = TRUE)
set.seed(123)
xgb_model3 <- train(Alcoholic.Status ~ ., 
                   data = new_train, 
                   method = "xgbTree",
                   trControl = train_control3,
                   tuneGrid = xgbGrid3)
pred_xgb30 <- predict(xgb_model3, new_test)
write.csv(data.frame("ID"=c(1:30000),"Alcoholic.Status"=pred_xgb2), file='pred_xgb30.csv', row.names = FALSE)

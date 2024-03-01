# loading packages 
library(dplyr)
library(class)
library(ggplot2)
library(glmnet)

# data 
training <- read.csv("completedData.csv")
testing <- read.csv("full_test.csv")
training$Alcoholic.Status <- factor(training$Alcoholic.Status)

# using lasso shrinkage to subset model with predictors
i <- seq(10, -2, length = 100)
lambda.v <- 10^i
length(lambda.v)

x <- model.matrix(Alcoholic.Status~.-1, data = training)
y <- training$Alcoholic.Status

# # SCALING REQUIRED
# x <- scale(x)

lasso1 <- glmnet(x, y, alpha = 1, lambda = lambda.v, family=binomial())
names(lasso1)
summary(lasso1)
coeffsL <- coef(lasso1)
dim(coeffsL)
coeffsL[1:33, 1:6]
lasso1$lambda[1:6]

set.seed(33)
cv.outputL <- cv.glmnet(x,y, alpha = 1, family = binomial())
qplot(log(cv.outputL$lambda), cv.outputL$cvsd)
plot(cv.outputL)
names(cv.outputL)
bestlamb.cvL <- cv.outputL$lambda.min
bestlamb.cvL

out <- glmnet(x, y, alpha = 1, lambda = bestlamb.cvL, family = binomial())
lasso.coef <- predict(out, type = "coefficients", s = bestlamb.cvL)[1:33,]
lasso.coef
lasso.coef[abs(lasso.coef) > 0.1]

# training data with suggest predictors
train_p <- training %>%
  select('sex', 'BMI', 'AGE.Category', 'Smoking.Status', 'Alcoholic.Status')

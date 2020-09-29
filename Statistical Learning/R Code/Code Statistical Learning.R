library(ggcorrplot)
library(boot)
library("scatterplot3d")
library(leaps)
library(glmnet)
library(factoextra)
library(doBy)
library(plotly) #3D plot
library(plotmo) #for the new glmnet plot
library(packHV)
library(xtable)
diamonds <- read.csv(file = 'Diamonds.csv')

#====================================================================================================================================================
#### INTRO ####
#====================================================================================================================================================
attach(diamonds)
diamonds <- diamonds[, - 1]  # to remove the first columns of indices
diamonds$cut <- factor(diamonds$cut, levels = c('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))
contrasts(diamonds$cut)
diamonds$color <- factor(diamonds$color, levels = c('J', 'I', 'H', 'G', 'F', 'E', 'D'))
contrasts(diamonds$color)
diamonds$clarity <- factor(diamonds$clarity, levels = c('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))
contrasts(diamonds$clarity)

summary(diamonds)
sum(is.na(diamonds)) #check presence of NA

par(mfrow=c(1,2))
hist_boxplot(carat,col = 'cadetblue1',freq = FALSE,main="carat")
hist_boxplot(depth,col = 'deepskyblue',freq = FALSE,main="depth")
par(mfrow=c(1,1))

par(mfrow=c(1,2))
hist_boxplot(table,col = 'dodgerblue2',freq = FALSE,main="table")
hist_boxplot(price,col = 'blue',freq = FALSE,main="price",xlab="price [$]")
par(mfrow=c(1,1))

par(mfrow=c(1,3))
hist_boxplot(x,col = 'yellow',freq = FALSE,main="x [mm]")
hist_boxplot(y,col = 'orange',freq = FALSE,main="y [mm]")
hist_boxplot(z,col = 'red',freq = FALSE,main="z [mm]")
par(mfrow=c(1,1))

layout(matrix(c(1,1,2,3), 2, 2, byrow = TRUE), widths=c(1,2))
barplot(table(diamonds$clarity)/nrow(diamonds), main = 'Clarity', col = 'springgreen',ylab="Density")
barplot(table(diamonds$color)/nrow(diamonds), main = 'Color', col = 'springgreen3',ylab="Density")
barplot(table(diamonds$cut)/nrow(diamonds), main = 'Cut', col = 'springgreen4', ylab="Density")
par(mfrow=c(1,1))

diamonds <- diamonds[!(z==0 | y==0 | x==0),] #remove rows with x or y or z =0
diamonds[diamonds$y > 30 | diamonds$z > 30, ]
summary(diamonds[diamonds$price > 2000 & diamonds$color == 'E' & diamonds$clarity == 'VS1', ])

attach(diamonds)

# new datasets  #
diamonds_norm <- diamonds
diamonds_norm[, c(1,5,6,8,9,10)] <- lapply(diamonds[, c(1,5,6,8,9,10)],function(x) c(scale(x))) #normalize numerical cols
diamonds_new <- diamonds #additional dataset with x/y
diamonds_new$r <- diamonds$x/diamonds$y
diamonds_new_norm <- diamonds_new
diamonds_new_norm[, c(1,5,6,8,9,10,11)] <- lapply(diamonds_new[, c(1,5,6,8,9,10,11)], function(x) c(scale(x)))
diamonds_new_norm_log <- diamonds_new_norm
diamonds_new_norm_log$carat <- scale(log(diamonds$carat)) #both r=x/y and log(carat)
diamonds_norm_log <- diamonds_norm 
diamonds_norm_log$carat <- scale(log(diamonds$carat)) #only log(carat)

# QQ plot #
par(mfrow=c(1,3))
hist(log(diamonds$price),freq = FALSE, main= "Histogram of log(price)")
qqnorm(diamonds$price,main="Q-Q plot of price")
qqline(diamonds$price)
qqnorm(log(diamonds$price),main="Q-Q plot of log(price)")
qqline(log(diamonds$price))
par(mfrow=c(1,1))

par(mfrow=c(2,2))
hist(diamonds$carat,freq = FALSE)
qqnorm(diamonds$carat,main="Normal Q-Q plot of carat")
qqline(diamonds$carat)
hist(log(diamonds$carat),freq = FALSE, main="Normal Q-Q plot of log(carat)")
qqnorm(log(diamonds$carat),main="Normal Q-Q plot of log(carat)")
qqline(log(diamonds$carat))
par(mfrow=c(1,1))

# 1vs1 plot #
par(mfrow=c(3,1))
plot(price~carat, xlab="carat", ylab="price [$]",main="carat vs price") 
plot(price~depth, xlab="depth", ylab="price [$]",main="depth vs price") 
plot(price~table, xlab="table [mm]", ylab="price [$]",main="table vs price")
par(mfrow=c(1,1))

par(mfrow=c(3,1))
plot(price~x, xlab="x [mm]", ylab="price [$]",main="x vs price")
plot(price~y, xlab="y [mm]", ylab="price [$]",main="y vs price")
plot(price~z, xlab="z [mm]", ylab="price [$]",main="z vs price")
par(mfrow=c(1,1))

plot(log(diamonds$carat),log(diamonds$price), main = "log(carat) vs log(price)", xlab= "log(carat)", ylab="log(price)", type = "p")

# 3D plot #
fig <- plot_ly(data = diamonds, x = diamonds$x, y = diamonds$y, z = diamonds$z,
               marker = list(color = diamonds$price, colorscale = c("yellow", "red"), showscale = TRUE))
fig <- fig %>% add_markers()
fig <- fig %>% layout(scene = list(xaxis = list(title = 'X [mm]',range =c(3,10)),
                                   yaxis = list(title = 'Y [mm]',range =c(3,10)),
                                   zaxis = list(title = 'Z [mm]',range =c(1,10))),
                      annotations = list(
                        x = 1.13,
                        y = 1.05,
                        text = 'Price [$]',
                        xref = 'paper',
                        yref = 'paper',
                        showarrow = FALSE
                      ))
fig

plot(depth~table, xlab="table [mm]", ylab="price [$]",main="table vs price")

my_screen_step1 <- split.screen(c(2, 1))
screen(my_screen_step1[1])
plot(log(diamonds$carat),log(diamonds$price), main = "log(carat) vs log(price)", xlab= "log(carat)", ylab="log(price)", type = "p",col=5)
my_screen_step2 <- split.screen(c(1, 2), screen = my_screen_step1[2])
screen(my_screen_step2[1])
hist(log(diamonds$carat),freq = FALSE,main="", xlab = "log(carat)", col = 5)
screen(my_screen_step2[2])
hist(log(diamonds$price),freq = FALSE,main="", xlab = "log(price)", col = 5)

# Matrix of corr # 
cormat <- round(cor(diamonds[sapply(diamonds, is.numeric)]), 2)
ggcorrplot(cormat,outline.col = "white",lab = TRUE, ggtheme = ggplot2::theme_gray,colors = c("#00AFBB", "#E7B800", "#FC4E07"))

cormat <- round(cor(diamonds_new_norm), 2)
ggcorrplot(cormat,outline.col = "white",lab = TRUE, ggtheme = ggplot2::theme_gray,colors = c("#00AFBB", "#E7B800", "#FC4E07"))

#### naive approach ####
lm_original <- lm(price~.,data=diamonds_norm)
summary(lm_original) 
par(mfrow = c(2,2))
plot(lm_original)
title("Naive Approach with normalized data",line = -1.1, outer = TRUE)

#### log(price) + normalized ####
lm_log <- lm (log(price)~.,data = diamonds_norm)
summary(lm_log)
par(mfrow = c(2,2))
plot(lm_log)
title("log(price) & normalized data",line = -1.1, outer = TRUE)

#### log(price) + log(carat) + normalized ####
lm_log_log_carat <- lm (log(price)~.,data = diamonds_norm_log)
summary(lm_log_log_carat)
par(mfrow = c(2,2))
plot(lm_log_log_carat)
title("log(price), log(carat) & normalized data",line = -1.1, outer = TRUE)

#### log(price) +log(carat) + r=x/y ####
lm_new <- lm (log(price)~.,data = diamonds_new_norm_log)
summary(lm_new)
par(mfrow = c(2,2))
plot(lm_new)
title("log(price), log(carat), r=x/y & normalized data",line = -1.1, outer = TRUE)

#====================================================================================================================================================
#### FEATURE SELECTION FOR log(price) & log(carat) & X/Y ####
#====================================================================================================================================================

# BEST SUBSET #
regfit.full <- regsubsets(log(price)~.,nvmax=24, method="exhaustive", data = diamonds_new_norm_log)
summary(regfit.full)
reg.summary <- summary(regfit.full)


par(mfrow=c(2,2))
# panel 1
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l")
which.min(reg.summary$rss)
points(which.min(reg.summary$rss),reg.summary$rss[which.min(reg.summary$rss)], col="red",cex=2,pch=20)
# panel 2
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)
points(which.max(reg.summary$adjr2),reg.summary$adjr2[which.max(reg.summary$adjr2)], col="red",cex=2,pch=20)
# panel 3
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type='l')
which.min(reg.summary$cp)
points(which.min(reg.summary$cp),reg.summary$cp[which.min(reg.summary$cp)],col="red",cex=2,pch=20)
# panel 4
plot(reg.summary$bic,xlab="Number of Variables", ylab="BIC", type='l')
which.min(reg.summary$bic)
points(which.min(reg.summary$bic),reg.summary$bic[which.min(reg.summary$bic)],col="red",cex=2,pch=20)
par(mfrow=c(1,1))

par(mfrow = c(1,3))
plot(regfit.full, main = "Adjusted R squared", scale = 'adjr2')
plot(regfit.full, main = "Mallow's Cp", scale = 'Cp')
plot(regfit.full, main = "BIC", scale = 'bic')

summary(regfit.full)$cp   #Cp coefficients varying the number of features

# BACKWARD #
lm_log_full = lm(log(price)~., data = diamonds_new_norm_log)
lm_log_AIC_b = step(lm_log_full, scope = log(price) ~1, trace=1, direction = "backward") #search based on AIC
summary(lm_log_AIC_b)
lm_log_BIC_b = step(lm_log_full, scope = log(price) ~ 1, trace=1, direction = "backward", k = log(nrow(diamonds_new_norm_log)),data = diamonds_new_norm_log)
summary(lm_log_BIC_b)

# FORWARD #
lm_log_empty = lm(log(price) ~ 1, data = diamonds_new_norm_log)
lm_log_AIC_f = step(lm_log_empty, scope = log(price) ~ carat + cut + color + clarity + depth + 
                      table + x + y + z + r, trace=1 ,direction = "forward")
summary(lm_log_AIC_f)
lm_log_BIC_f = step(lm_log_empty, scope = log(price) ~ carat + cut + color + clarity + depth + 
                      table + x + y + z + r, trace=1 ,direction = "forward",k = log(length(diamonds_new_norm_log)),data = diamonds_new_norm_log) #search based on BIC
summary(lm_log_BIC_f)

# BOTH #
lm_log_empty = lm(log(price) ~ 1, data = diamonds_new_norm_log)
lm_log_AIC = step(lm_log_empty, scope = log(price) ~ carat + cut + color + clarity + depth + 
                    table + x + y + z + r, trace=1 ,direction = "both")
summary(lm_log_AIC)
lm_log_BIC = step(lm_log_empty, scope = log(price) ~ carat + cut + color + clarity + depth + 
                    table + x + y + z + r, trace=1 ,direction = "both",k = log(length(diamonds_new_norm_log)),data = diamonds_new_norm_log) #search based on BIC
summary(lm_log_BIC)



set.seed(100)
test_percentage <- 0.3
m <- dim(diamonds)[1] ### number of instances

test_ind <- sample(1: m, round(m * test_percentage))

train <- diamonds[-test_ind,]
test <- diamonds[test_ind,]

X_train <- model.matrix(log(train[,7])~., data = train)
X_test <- model.matrix(log(test[,7])~., data = test)

#====================================================================================================================================================
#### LM con log(price) & log(carat) + r ####
#====================================================================================================================================================

lm_reduced <- lm(log(price)~. - table - z, data = diamonds_new_norm_log, subset = -test_ind)
summary(lm_reduced)
par(mfrow = c(2,2))
plot(lm_reduced)
title("Linear model reduced",line = -1.1, outer = TRUE)
par(mfrow = c(1,1))

pred_lm_reduced <- predict(lm_reduced, newdata = diamonds_new_norm_log[test_ind,])

mse.lm_reduced <- sqrt(mean((diamonds_new_norm_log[test_ind, 7] - exp(pred_lm_reduced))^2))
mae.lm_reduced <- mean(abs(diamonds_new_norm_log[test_ind, 7] - exp(pred_lm_reduced)))
# Analysis of leverage points #
lev_points <- hatvalues(lm_reduced, type = "diagonal")
summary(lev_points)
sort(lev_points, decreasing = T)[1:10]#evaluate high leverage points
pos <- which.maxn(lev_points,n=5) #return position of 10 highest value, 
diamonds_new_norm_log[pos, -c(6,10)]
diamonds_new[pos, -c(6,10)]

# Cook #
cd <- cooks.distance(lm_reduced) #evaluate cooks dist from original model
bound <- 4/length(diamonds_new_norm_log) #find upper bound for CD
diamonds_new_norm_log[cd >= bound, -c(6,10)]
diamonds_new[cd >= bound,]

#==============================================================================================================================
#### RIDGE / LASSO with log(price) + log(carat) + r ####
#==============================================================================================================================

train_new_red <- diamonds_new_norm_log[-test_ind,]
test_new_red <- diamonds_new_norm_log[test_ind,]

X_train_new <- model.matrix(log(train_new_red[,7])~., data = train_new_red[,-7])
X_test_new <- model.matrix(log(test_new_red[,7])~., data = test_new_red[,-7])

#### RIDGE REGRESSION with CV WITH ####
lambdas <- 10^seq(2, -10, length=100)

lm_ridge_train.cv_new <- cv.glmnet(X_train_new[, -7], log(train_new_red[, 7]), alpha = 0, lambda = lambdas) #10 folds
plot(lm_ridge_train.cv_new)
title("Ridge Regression", line = +2.5)
coef(lm_ridge_train.cv_new)
lm_ridge_train.cv_new$lambda.1se

# ridge regressin on the entire training set
lm_ridge_train_new <- glmnet(X_train_new[,-7], log(train_new_red[, 7]), alpha=0, lambda = lambdas)
plot_glmnet(lm_ridge_train_new, label=TRUE) 
abline(v = log(lm_ridge_train.cv_new$lambda.1se), lty = 2, col = 4)
title("Ridge Regression", line = +3.5)

lm_ridge_train <- glmnet(X_train_new[,-7], log(train_new_red[, 7]), alpha=0, lambda = lm_ridge_train.cv_new$lambda.1se)

#prediction on the test set
pred_lm_ridge <- predict(lm_ridge_train, newx=X_test_new[,-7])
mse.lm_ridge <- sqrt(mean((exp(pred_lm_ridge) - test_new_red[,7])^2))
mae.lm_ridge <- mean(abs(exp(pred_lm_ridge) - test_new_red[,7]))


# LASSO REGRESSION with CV #
lambdas2 <- 10^seq(3, -5, length=100)

lm_lasso_train.cv_new <- cv.glmnet(X_train_new[, -7], log(train_new_red[, 7]), alpha = 1, lambda = lambdas2) #10 folds
plot(lm_lasso_train.cv_new, xvar="lambda", label=T)
title("Lasso Regression", line = +2.5)

coef(lm_lasso_train.cv_new)
lm_lasso_train.cv_new$lambda.1se

# lasso regressin on the entire training set
lm_lasso_train_new <- glmnet(X_train_new[,-7], log(train_new_red[, 7]), alpha=1, lambda = lambdas2)
plot_glmnet(lm_ridge_train_new, label=TRUE) 
abline(v = log(lm_ridge_train.cv_new$lambda.1se), lty = 2, col = 4)
title("Ridge Regression", line = +3.5)

lm_lasso_train <- glmnet(X_train_new[,-7], log(train_new_red[, 7]), alpha=1, lambda = lm_lasso_train.cv_new$lambda.1se)

#prediction on the test set
pred_lm_lasso <- predict(lm_lasso_train, newx=X_test_new[,-7])
mse.lm_lasso <- sqrt(mean((exp(pred_lm_lasso) - test_new_red[,7])^2))
mae.lm_lasso <- mean(abs(exp(pred_lm_lasso) - test_new_red[,7]))


#### PCA without log ####

diamonds_num_scaled_new <- diamonds_new_norm[, c(1,5,6,8,9,10,11)] #get scaled numerical cols except price

pc_new <- princomp(diamonds_num_scaled_new, cor=TRUE) #compute PCA
fviz_eig(pc_new, addlabels = TRUE)
fviz_pca_var(pc_new, repel = TRUE, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

x.pc_new <- as.data.frame(pc_new$scores[,1:4])
x.pc_new <- cbind(x.pc_new, diamonds$price)
names(x.pc_new)[names(x.pc_new) == 'diamonds$price'] <- 'price'
x.pc.full_new <- cbind(x.pc_new, diamonds[,c(2,3,4)])

lm_PCA_new <- glm(log(price)~., data = x.pc_new[-test_ind,])
summary(lm_PCA_new)
summary(lm_PCA_new)$r.squared
par(mfrow = c(2,2))
plot(lm_PCA_new)

cv.err_new <- cv.glm(data = x.pc_new[-test_ind, ], glmfit = lm_PCA_new, K = 5)
cv.err_new$delta

pred_PCA_new <- predict(lm_PCA_new, newdata = x.pc_new[test_ind,])
mse.PCA_new <- sqrt(mean((exp(pred_PCA_new) - x.pc_new[test_ind, 5])^2))
mae.PCA_new <- mean(abs(exp(pred_PCA_new) - x.pc_new[test_ind, 5]))

#FULL#
lm_PCA_full_new <- glm(log(price)~., data = x.pc.full_new[-test_ind,])
summary(lm_PCA_full_new)
cv.err_full_new <- cv.glm(data = x.pc.full_new[-test_ind,], glmfit = lm_PCA_full_new, K = 5)
cv.err_full_new$delta

pred_PCA_full_new <- predict(lm_PCA_full_new, newdata = x.pc.full_new[test_ind,])
mse.PCA_full_new <- sqrt(mean((exp(pred_PCA_full_new) - x.pc.full_new[test_ind, 5])^2))
mae.PCA_full_new <- mean(abs(exp(pred_PCA_full_new) - x.pc.full_new[test_ind, 5]))


#PCA WITH LOG(CARAT) #
diamonds_num_scaled_new_log <- diamonds_new_norm_log[, c(1,5,6,8,9,10,11)] #get scaled numerical cols except price

pc_new_log <- princomp(diamonds_num_scaled_new_log, cor=TRUE) #compute PCA
fviz_eig(pc_new_log, addlabels = TRUE)
fviz_pca_var(pc_new_log, repel = TRUE, col.var = "contrib", gradient.cols = c("#00AFBB", "#E7B800", "#FC4E07"))

x.pc_new_log <- as.data.frame(pc_new_log$scores[,1:4])
x.pc_new_log <- cbind(x.pc_new_log, diamonds$price)
names(x.pc_new_log)[names(x.pc_new_log) == 'diamonds$price'] <- 'price'
x.pc.full_new_log <- cbind(x.pc_new_log, diamonds[,c(2,3,4)])

lm_PCA_new_log <- glm(log(price)~., data = x.pc_new_log[-test_ind,])
summary(lm_PCA_new_log)
cv.err_new_log <- cv.glm(data = x.pc_new_log[-test_ind, ], glmfit = lm_PCA_new_log, K = 5)
cv.err_new_log$delta

pred_PCA_new_log <- predict(lm_PCA_new_log, newdata = x.pc_new_log[test_ind,])
mse.PCA_new_log <- sqrt(mean((exp(pred_PCA_new_log) - x.pc_new_log[test_ind, 5])^2))
mae.PCA_new_log <- mean(abs(exp(pred_PCA_new_log) - x.pc_new_log[test_ind, 5]))

#FULL#
lm_PCA_full_new_log <- glm(log(price)~., data = x.pc.full_new_log[-test_ind,])
summary(lm_PCA_full_new_log)
cv.err_full_new_log <- cv.glm(data = x.pc.full_new_log[-test_ind,], glmfit = lm_PCA_full_new_log, K = 5)
cv.err_full_new_log$delta

pred_PCA_full_new_log <- predict(lm_PCA_full_new_log, newdata = x.pc.full_new_log[test_ind,])
mse.PCA_full_new_log <- sqrt(mean((exp(pred_PCA_full_new_log) - x.pc.full_new_log[test_ind, 5])^2))
mae.PCA_full_new_log <- mean(abs(exp(pred_PCA_full_new_log) - x.pc.full_new_log[test_ind, 5]))








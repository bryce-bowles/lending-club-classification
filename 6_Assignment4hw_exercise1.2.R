library(rpart) #for the classification trees
library(caret) #classification and regression training
library(ROCR)
library(dplyr)
load("lending_data_2017_Q2.rda")
set.seed(12345)


my_lend_df <- lending_data %>%
  dplyr::select(loan_status, 
                application_type, 
                total_bal_ex_mort, 
                total_bc_limit,
                revol_util,
                int_rate,
                emp_length,
                annual_inc,
                dti,
                fico_range_low, 
                fico_range_high) %>%
  filter(loan_status %in% c("Fully Paid", "Charged Off")) %>%
  droplevels() 
summary(my_lend_df)
summary(my_lend_df$loan_status)


train_lend_rows <- createDataPartition(my_lend_df$loan_status, 
                                 p = 0.1, #chose %10 to have enough to learn pattern & to predict
                                 list=FALSE)
train_lend <- my_lend_df[train_lend_rows,]
test_lend <- my_lend_df[-train_lend_rows,]


#Proportions of train / test loan_status
summary(train_lend$loan_status)
summary(test_lend$loan_status)

summary(train_lend)

#Impute medians
#calculate medians from training data
lend_med_impute <- preProcess(train_lend, method="medianImpute")

#use same medians to impute to test data
train_lend <- predict(lend_med_impute, train_lend)
test_lend <- predict(lend_med_impute, test_lend)
summary(train_lend)
summary(test_lend)


#Weights of loan_status
summary(train_lend$loan_status) #use this to determine weights below - inverse the ratio
my_lend_weights <- numeric(nrow(train_lend))
my_lend_weights[train_lend$loan_status == "Fully Paid"] <- 1
my_lend_weights[train_lend$loan_status == "Charged Off"]  <- 4


## Logistic Regression Model 
my_lend_lr <- glm(loan_status ~ ., 
		data=train_lend, 
		weights=my_lend_weights,
		family=binomial("logit"))
my_lend_lr_predict <- predict(my_lend_lr, 
			   newdata=test_lend, 
			   type="response") #probability of being in the positive class
#my_lend_lr_predict #above .5 is more likely of being fully paid

#to read results of Log. Reg. model, create a empty list of strings
my_lend_lr_predict_class <- character(length(my_lend_lr_predict))
my_lend_lr_predict_class[my_lend_lr_predict < 0.5] <- "Charged Off"
my_lend_lr_predict_class[my_lend_lr_predict >= 0.5] <- "Fully Paid"

# Confusion Matrix
my_lend_lg_cm <- table(test_lend$loan_status, my_lend_lr_predict_class)
my_lend_lg_cm
#Of those that were Charged off, 6477 correctly classified and 8655 were not. 
#Of those that were Fully Paid, 22856 were incorrectly classified and 42240 were. 
#want larger numbers on the diagonal

# Misclassification Rate
1-sum(diag(my_lend_lg_cm))/sum(my_lend_lg_cm)

## Classification Tree Model 
my_lend_rpart <- rpart(loan_status ~ ., data=train_lend, weights=my_lend_weights)
my_lend_rpart
my_lend_rpart$variable.importance
my_lend_rpart_predict <- predict(my_lend_rpart, newdata=test_lend, type="class")
my_lend_rpart_predict

#Classification Tree Confusion Matrix
my_lend_rpart_cm <- table(test_lend$loan_status, my_lend_rpart_predict)
my_lend_rpart_cm
#Classification Tree Misclassification Rate
1-sum(diag(my_lend_rpart_cm))/sum(my_lend_rpart_cm)

my_lend_rpart$variable.importance

#generate probabilities for logistic regression model using predict function
lend_lr_predict <- predict(my_lend_lr, test_lend, type="response")
lend_lr_pred <- prediction(lend_lr_predict, 
			 test_lend$loan_status,
			 label.ordering=c("Charged Off", "Fully Paid"))

lend_lr_pred #prediction object

lend_lr_perf <- performance(lend_lr_pred, "tpr", "fpr")
lend_lr_perf

#"prob" for classification trees
lend_rpart_predict  <- predict(my_lend_rpart, test_lend, type="prob")
#lend_rpart_predict #matrix of probabilities
lend_rpart_pred <- prediction(lend_rpart_predict[,2], #only want 2nd column
			    test_lend$loan_status,
			    label.ordering=c("Charged Off", "Fully Paid"))
lend_rpart_perf <- performance(lend_rpart_pred, "tpr", "fpr") #performance object


plot(lend_lr_perf, col=1)
plot(lend_rpart_perf, col=2, add=TRUE)
legend(0.5, 0.6, c("Log. Reg.", "Class. Tree"), col=1:2, lwd=3)

lend_lr_auc <- performance(lend_lr_pred, "auc")
lend_lr_auc@y.values[[1]]
lend_rpart_auc <- performance(lend_rpart_pred, "auc")
lend_rpart_auc@y.values[[1]]

#Gains plot
lend_lr_gains <- performance(lend_lr_pred, "tpr", "rpp")
lend_rpart_gains <- performance(lend_rpart_pred, "tpr", "rpp")
plot(lend_lr_gains, col=1)
plot(lend_rpart_gains, col=2, add=TRUE)
legend(0.7, 0.6, c("Log. Reg.", "Class. Tree"), col=1:2, lwd=3)

#########################################################
#Q3
lend_df_1 <- lending_data %>%
  dplyr::select(loan_status, 
                loan_amnt, 
                funded_amnt_inv, 
                term, int_rate, 
                installment, grade, 
                emp_length, 
                home_ownership, 
                annual_inc, 
                verification_status, 
                loan_status, 
                purpose, 
                title, dti, 
                total_pymnt, 
                delinq_2yrs, 
                open_acc, 
                pub_rec, 
                last_pymnt_d, 
                last_pymnt_amnt, 
                application_type, 
                revol_bal, 
                revol_util, 
                recoveries) %>%
  filter(loan_status %in% c("Fully Paid", "Charged Off")) %>%
  droplevels() 
summary(lend_df_1)

# Fixing home_ownership grouping
summary(lend_df_1$home_ownership)
levels(lend_df_1$home_ownership)
levels(lend_df_1$home_ownership) <- c("RENT",
                                      "MORTGAGE",
                                      "RENT",#grouping any and none with the rent
                                      "OWN",
                                      "RENT")
summary(lend_df_1$home_ownership)

# Fixing last_pymnt_d format
summary(lend_df_1$last_pymnt_d)
lend_df_1$last_pymnt_d <- as.POSIXct(lend_df_1$last_pymnt_d)

# Partition Data
train_lend_rows <- createDataPartition(lend_df_1$loan_status, 
                                       p = 0.1, 
                                       list=FALSE)
train_lend_1 <- lend_df_1[train_lend_rows,]
summary(train_lend_1)
test_lend_1 <- lend_df_1[-train_lend_rows,]

## NA's ##
# Identifying NA's
apply(is.na(train_lend_1),2,sum)
#Impute medians
#calculate medians from training data
lend_med_impute_1 <- preProcess(train_lend_1, method="medianImpute")
train_lend_1 <- predict(lend_med_impute_1, train_lend_1)
test_lend_1 <- predict(lend_med_impute_1, test_lend_1)
summary(train_lend_1)
summary(test_lend_1)

# Identifying NA's left
apply(is.na(train_lend_1),2,sum)
# Fixing NA's (last_pymnt_d)
train_lend_1$last_pymnt_d[is.na(train_lend_1$last_pymnt_d)] <-
  median(train_lend_1$last_pymnt_d, na.rm=TRUE)
apply(is.na(train_lend_1),2,sum)
summary(train_lend_1)

test_lend_1$last_pymnt_d[is.na(test_lend_1$last_pymnt_d)] <-
  median(train_lend_1$last_pymnt_d, na.rm=TRUE)
apply(is.na(train_lend_1),2,sum)
summary(test_lend_1)


summary(train_lend_1$loan_status)
my_lend_weights_1 <- numeric(nrow(train_lend_1))
my_lend_weights_1[train_lend_1$loan_status == "Fully Paid"] <- 1
my_lend_weights_1[train_lend_1$loan_status == "Charged Off"]  <- 4



## Logistic Regression Model ##
my_lend_lr_1 <- glm(loan_status ~ ., 
                    data=train_lend_1, 
                    weights=my_lend_weights_1,
                    family=binomial("logit"))
my_lend_lr_predict_1 <- predict(my_lend_lr_1, 
                                newdata=test_lend_1, 
                                type="response")
my_lend_lr_predict_class_1 <- character(length(my_lend_lr_predict_1))
my_lend_lr_predict_class_1[my_lend_lr_predict_1 < 0.5] <- "Charged Off"
my_lend_lr_predict_class_1[my_lend_lr_predict_1 >= 0.5] <- "Fully Paid"
# Confusion Matrix
my_lend_lg_cm_1 <- table(test_lend_1$loan_status, my_lend_lr_predict_class_1)
my_lend_lg_cm_1
# Misclassification Rate
1-sum(diag(my_lend_lg_cm_1))/sum(my_lend_lg_cm_1)



## Classification Tree Model ##
my_lend_rpart_1 <- rpart(loan_status ~ ., data=train_lend_1, weights=my_lend_weights_1)
my_lend_rpart_predict_1 <- predict(my_lend_rpart_1, newdata=test_lend_1, type="class")
# Confusion Matrix
my_lend_rpart_cm_1 <- table(test_lend_1$loan_status, my_lend_rpart_predict_1)
my_lend_rpart_cm_1
# Misclassification Rate
1-sum(diag(my_lend_rpart_cm_1))/sum(my_lend_rpart_cm_1)
# Variable importance
my_lend_rpart_1$variable.importance


#Prediction
lend_lr_predict_1 <- predict(my_lend_lr_1, test_lend_1, type="response")
lend_lr_pred_1 <- prediction(lend_lr_predict_1, 
                             test_lend_1$loan_status,
                             label.ordering=c("Charged Off", "Fully Paid"))
lend_lr_perf_1 <- performance(lend_lr_pred_1, "tpr", "fpr")

lend_rpart_predict_1  <- predict(my_lend_rpart_1, test_lend_1, type="prob")
lend_rpart_pred_1 <- prediction(lend_rpart_predict_1[,2], 
                                test_lend_1$loan_status,
                                label.ordering=c("Charged Off", "Fully Paid"))
lend_rpart_perf_1 <- performance(lend_rpart_pred_1, "tpr", "fpr")

#AUC Values
lend_lr_auc_1 <- performance(lend_lr_pred_1, "auc")
lend_lr_auc_1@y.values[[1]]
lend_rpart_auc_1 <- performance(lend_rpart_pred_1, "auc")
lend_rpart_auc_1@y.values[[1]]

plot(lend_lr_perf, col=1)
plot(lend_rpart_perf, col=2, add=TRUE)
plot(lend_lr_perf_1, col=3, add=TRUE)
plot(lend_rpart_perf_1, col=4, add=TRUE)
legend(0.7, 0.6, c("Log. Reg. 1", "Class. Tree 1", "Log. Reg. 2", "Class. Tree 2"), col=1:2:3:4, lwd=3)









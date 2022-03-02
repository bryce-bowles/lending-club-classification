# Lending Club Classification Analysis: 
Built a logistic regression model and a classification tree model for predicting the final status of a loan based on multiple variables available. Confusion matrix and misclassification rate for each model for a test dataset. Includes variables that appear to be important for predicting outcome. Plotted and described the ROC curves and AUC for the four models to provide a recommendation. ([Report](https://github.com/bryce-bowles/lending-club-classification/blob/f14d7c757bd5d56aa24c444df32d96931a77d536/6_Assignment4.pdf))

## Project Instructions
11/06/2020

1.	Read Chapter 2.2 of Introduction to Statistical Learning in R and the article "The Foundations of Algorithmic Bias".   Write a 1/2 to 1 page response (prose, in complete sentences).  In your essay address the following:
o	Provide two definitions of bias.
o	How can we avoid some of the undesirable effects of each type of bias?
o	In addition to bias, what are other ethical challenges facing data scientists today?  Are there other concerns that are more important than bias?

Use the Lending Club 2017 Q2 data for the remaining questions.

2. Build a logistic regression model and a classification tree model for predicting the final status of a loan based on variables available at the time at which a loan is awarded.  Provide a confusion matrix and misclassification rate for each model for a test dataset.  Which variables appear to be important for predicting outcome?
 
3. Build a logistic regression model and a classification tree model for predicting the final status of a loan based on the variables loan_amnt, funded_amnt_inv, term, int_rate, installment, grade, emp_length, home_ownership, annual_inc, verification_status, loan_status, purpose, title, dti, total_pymnt, delinq_2yrs, open_acc, pub_rec, last_pymnt_d, last_pymnt_amnt, application_type, revol_bal, revol_util, recoveries.   Use the same training and testing observations as for question 2.  Provide a confusion matrix and misclassification rate for each model for a test dataset.   To format the date in last_pymnt_d properly, use the following code: my_df$last_pymnt_d <- as.POSIXct(my_df$last_pymnt_d)
 
4. Plot the ROC curves for your four models.  Which models perform best?  To what do you attribute the differences in performance between the models in #2 and #3?
 
5. If you were considering investing in Lending Club loans, which model would you use to support your decision making?
 
Update 9/10/20:  Create a report that contains a summary of the steps that you took and the results that you obtained.  Do not include code in your report and avoid specific references to R - write the report as if you are reporting to someone unfamiliar with R (but you may assume familiarity with the analytics methods).  Submit your code as a separate .R file so that the instructor can run it if necessary.  Refer to the syllabus for more details.

Update 10/19/20: A link to the article "The Foundations of Algorithmic Bias" is in the Module Online Resources on the page Analytics Articles.
 
# 1 - Essay
Techniques to evaluate models are provided throughout the readings to help assess model accuracy or decide which method produces the best results. Since no one Statistical Learning method dominates all others over all possible data sets, it is important to try more than one so that a model gets evaluated for optimal results. Quality of fit, bias-variance trade-off and classification setting all influence model accuracy. 

Bias refers to the error that is introduced by approximating a real-life problem, which may be extremely complicated, by a much simpler model. One way bias gets added to models is by “overfitting the data”. This can be explained by how well a model’s predictions actually match the observed data. The way we may measure this is to quantify the extent to which the predicted response value for a given observation is close to the true response value for that observation. In a regression setting, the most commonly used measure is the mean squared error (MSE). The MSE will be small if the predicted responses are very close to the true responses. Another way Bias can affect models is by the way the data are selected. I thought the “The Foundations of Algorithmic Bias” article mentioned an interesting and important point of when training a machine learning model, it is only going to learn on the training data’s scenarios and other scenarios that may be in the test dataset will be left out. The data itself may be bias, depending on how parameters were selected and if individuals choosing those parameters contained any bias. 

To avoid some of the undesirable effects of each type of bias, you will need to evaluate each way the data is being collected and used. For example, in linear regression, you may test the true f linear relationship between dependent variable and other variables in the data. If f is linear, linear regression will have no bias, making it very hard for a more flexible method to compete. In contrast, if the true f is highly non-linear and we have an ample number of training observations, then we may do better using a highly flexible approach.  Another method is to use cross-validation, a way to estimate the test MSE using the training data. When collecting data, in general, the more data you use, the better off you will be. 

Other ethical challenges facing data scientists today include MSE, variance, and squared bias. Variance refers to the amount by which f would change if we estimated it using a different training data set. As a general rule, using more flexible methods, the variance will increase and the bias will decrease. Low bias and low variance are best for predictions. Another ethical challenge may be getting a good test set. Good test set performance of a statistical learning method bias-variance requires low variance as well as low squared bias. The challenge lies in finding a method for which both the variance and the squared bias are low. In a real-life situation in which f is unobserved, it is generally not possible to explicitly compute the test MSE, bias, or variance for a statistical learning method. 

Assessing model accuracy depends on quality of fit, bias-variance trade-off, classification setting and much more. Bias may be included in a predetermined dataset. After the readings, it is my recommendation that data scientists should be aware of this and continuously collect data to avoid predetermined bias’s so that models adjust to changes made over time. In addition to continuously collecting data, continuously evaluating the performance of the model as the data changes will also be essential in obtaining optimal results. 

 
# 2 - Logistic regression and classification tree model: 
For predicting the final status of a loan based on variables available at the time at which a loan is awarded. 

After reading in the data, columns were chosen to help predict the final status of a loan such as application_type, total_bal_ex_mort, total_bc_limit and revol_util, int_rate, emp_length, annual_inc, dti, fico_range_low, and fico_range_high. Columns where then filtered to exclude all status types except “Fully Paid” and “Charged Off” and to only include the final status’. Linear regression can only predict classification models in instances where you have two outcomes. The data was split to get a sample of 10% while choosing the same proportions of status types in both the training and test data. 1,000 samples will provide a good enough turnout while keeping computational speed fast. Next, the missing data (N/As) are replaced, using the median from training data to fill the test data to prevent leakage. Since the ratio is about 4 “charged off” to 1 “fully paid”, weights were added to indicate the significance of each type.

A logistic regression model was then built to predict the loan status. Probability for each loan that was fully paid was calculated and we used this to determine the prediction. If the probability was anything over 0.5 its prediction is “fully Paid” and anything under 0.5 is “Charged Off”. A confusion matrix was then used to compare the actual results verses the predicted. Of those that were Charged off, 8235 correctly classified and 6897 were not. Of those that were Fully Paid, 46362 were correctly classified and 18734 were not. The logistic regression model had a misclassification rate of 0.319477.

The Classification Tree model was then built to compare to the logistic regression model, still predicting the loan status. A confusion matrix was used was used to compare the actual results verses the predicted. Of those that were Charged off, 9972 were correctly classified and 5160 were not. Of those that were Fully Paid, 40291 were correctly classified and 24805 were not. This model had a misclassification rate of 0.373498. Below is how the tree is broken down as well as variable importance. The int_rate, fico_range_high and fico_range_low appear to be important for predicting outcome.

Tree:

![image](https://user-images.githubusercontent.com/65502025/156423506-c097c8b2-4412-48ff-80cf-cb2f94d42518.png)


Variable Importance: 

![image](https://user-images.githubusercontent.com/65502025/156423531-a6480f46-8d66-48f6-aa2d-0e55518bf8b7.png)

 
A ROC Curve was used to display the models. The Classification Tree model had an AUC (Area under the ROC Curve) value of 0.6710865 of and Logistic Regression had a 0.6883902 value, indicating the logistic regression is the best model for this scenario.

![image](https://user-images.githubusercontent.com/65502025/156423567-fc1d4ac1-ba02-4a51-a508-d1708776ddba.png)

![image](https://user-images.githubusercontent.com/65502025/156423580-17e747ed-2e5c-4788-b895-01aa636e3527.png)

 

 
# 3 - Logistic regression model and a classification tree model for:
Predicting the final status of a loan based on the variables loan_amnt, funded_amnt_inv, term, int_rate, installment, grade, emp_length, home_ownership, annual_inc, verification_status, loan_status, purpose, title, dti, total_pymnt, delinq_2yrs, open_acc, pub_rec, last_pymnt_d, last_pymnt_amnt, application_type, revol_bal, revol_util, recoveries.  

First, the variables were modified from the previous models. Columns where again filtered to exclude all status types except “Fully Paid” and “Charged Off” and to only include the final status’. Similar preprocessing steps were used to remove any NA’s and adjust the home_owndership variable categories. Once again, the categories “ANY” and “NONE” were grouped with RENT.  The below summary was used to identify variable preprocessing needs. The data was split to get a sample of 10% while choosing the same proportions of status types in both the training and test data. Next, the missing data (N/As) are replaced, using the median from training data to fill the test data to prevent leakage. Since the ratio of loan_status is about 4 “charged off” to 1 “fully paid”, weights were added to indicate the significance of each type.

![image](https://user-images.githubusercontent.com/65502025/156423607-9cf0e014-bc8c-4c85-96c6-f82de1a4ad8e.png)


When creating a logistic regression and classification tree models, the same steps in the above question 2 model were used.

Logistic Regression model: Of those that were Charged off, 14876 correctly classified and 256 were not. Of those that were Fully Paid, 64867 were correctly classified and 229 were not. The logistic regression model had a misclassification rate of 0.006045271.

Classification Tree model: Of those that were Charged off, 14759 were correctly classified and 373 were not. Of those that were Fully Paid, 61500 were correctly classified and 3596 were not. This model had a misclassification rate of 0.04947151. The recoveries, last_pumnt_amnt, last_pymnt_d, total_pymnt etc… in that order appear to be important for predicting outcome. (the higher the value, the more important) 

 ![image](https://user-images.githubusercontent.com/65502025/156423648-c202d9e3-62cd-438a-b49c-5ebb7f3f2c25.png)

# 4 ROC Curve Comparrison
The Classification Tree model had an AUC (Area under the ROC Curve) value of 0.9885012 of and Logistic Regression had a 0.9967356 value, indicating the logistic regression is the best model for this scenario.

![image](https://user-images.githubusercontent.com/65502025/156423667-0b316282-aaad-4e45-bcb0-ed15d87e4bf8.png)


# 5 Recommendation
The 2nd Logistic Regression created (green) performs the best. Other factors that come into play are overfitting and biases. If I were to consider investing in Lending Club loans, I’d use the first linear regression model (Black) to support my decision making. This model does not perform as well but it also suffers from less leakage and less overfitting of the model. 



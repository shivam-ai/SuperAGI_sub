# SuperAGI_sub
SuperAGI Assignment

Name: Shivam Kanojia
Entry Number: 2019CS510131


You train Logistic Regression with a certain set of features and learn weights w_0, w_1 till w_n. Feature n gets weight w_n at the end of training. Say you now create a new dataset where you duplicate feature n into feature (n+1) and retrain a new model. Suppose this new model weights are w_{new_0}, w_{new_1} till w_{new_n}, w_{new_{n+1}}. What is the likely relationship between w_{new_0}, w_{new_1} , w_{new_n}, and w_{new_{n+1}}?

Soln.  When we duplicate feature n into feature (n+1) and retrain logistic regression model, the    weights wnew0, wnew1, wnew n, wnew n+1 will exhibit a specific relationship.
When you duplicate a feature n into feature (n+1), it means that both features n and (n+1) are essentially representing the same information in the model. Therefore, the weights corresponding to these duplicated features are expected to be similar or identical.
In a logistic regression model, the weights are assigned to features to indicate the strength and direction of their influence on the predicted outcome. If you duplicate feature n into feature (n+1), it implies that both features have the same impact on the output, and the model should ideally assign similar weights to both.
Wnew0 (bias term): This weight might be similar to w0     in the original model, but the exact value could be influenced by the overall learning process and the interaction with other features.
Wnew1: This weight is associated with the original feature 1 and is not directly influenced by the duplication of feature n into feature (n+1). Therefore, wnew1 may be similar to w1    in the original model, but it could be influenced by the retraining process.

You currently have an email marketing template A and you want to replace it with a better template. A is the control_template. You also test email templates B, C, D, E. You send exactly 1000 emails of each template to different random users. You wish to figure out what email gets the highest click through rate. Template A gets 10% click through rate (CTR), B gets 7% CTR, C gets 8.5% CTR, D gets 12% CTR and E gets 14% CTR. You want to run your multivariate test till you get 95% confidence in a conclusion. Which of the following is true?
We have too little data to conclude that A is better or worse than any other template with 95% confidence.
E is better than A with over 95% confidence, B is worse than A with over 95% confidence. You need to run the test for longer to tell where C and D compare to A with 95% confidence.
Both D and E are better than A with 95% confidence. Both B and C are worse than A with over 95% confidence
Soln.  To determine the significance of the results in a multivariate test, we can use hypothesis testing and confidence intervals. The click-through rates (CTRs) you provided are as follows:
A (control_template): 10% CTR
B: 7% CTR
C: 8.5% CTR
D: 12% CTR
E: 14% CTR
Now, let's evaluate the options:
We have too little data to conclude that A is better or worse than any other template with 95% confidence.
This option acknowledges the potential impact of sample size on the statistical significance of the results. Given the information provided, it's a valid consideration.
E is better than A with over 95% confidence, B is worse than A with over 95% confidence. You need to run the test for longer to tell where C and D compare to A with 95% confidence.
This option seems reasonable. It acknowledges the confidence in the comparison between E and A, as well as B and A. It also recognizes the need for more data to assess the comparisons between C and A, and D and A.
Both D and E are better than A with 95% confidence. Both B and C are worse than A with over 95% confidence.
This statement is making a stronger claim. It suggests definitive conclusions about D and E being better than A and B and C being worse than A. Such strong conclusions might be premature without further testing.
Considering the caution warranted in making strong claims and the need for more data to assess the comparisons involving Templates C and D, the most accurate option is:
We have too little data to conclude that A is better or worse than any other template with 95% confidence.
You have m training examples and n features. Your feature vectors are however sparse and the average number of non-zero entries in each train example is k and k << n. What is the approximate computational cost of each gradient descent iteration of logistic regression in modern well written packages?

Soln. In logistic regression, the cost of each gradient descent iteration is determined by the complexity of computing the gradient of the cost function with respect to the parameters. The specific implementation details can vary across different packages, but I'll provide a general overview.
Let's denote the number of training examples as m, the number of features as n, and the average number of non-zero entries in each training example as k, where k≪n.
In logistic regression, the cost function is typically the negative log-likelihood, and the gradient with respect to the parameters is computed as the difference between the predicted probabilities and the true labels multiplied by the feature values. This involves performing operations on the entire dataset.
The computational cost of each gradient descent iteration is dominated by the following factors:
Matrix Multiplication:
The main computational cost comes from multiplying the feature matrix by the difference between predicted probabilities and true labels. In the sparse case, this involves selectively updating only the non-zero entries.
Regularization Term (if used):
If regularization is applied (L1 or L2 regularization), there will be additional computations for the regularization term.
Modern well-written packages, such as scikit-learn or TensorFlow, are optimized for efficiency. They often leverage optimized linear algebra libraries (e.g., BLAS or LAPACK) and may use sparse matrix representations to speed up computations for sparse data.
The approximate computational cost per iteration can be expressed in big-O notation as: O(m⋅n⋅k). This takes into account the matrix multiplication and the fact that k is the average number of non-zero entries in each training example. Keep in mind that this is a rough estimate and the actual implementation details can vary across packages. The efficiency gains from sparse matrix representations and optimized linear algebra libraries are crucial for handling large datasets with sparse features.


We are interested in building a high quality text classifier that categorizes news stories into 2 categories - information and entertainment. We want the classifier to stick with predicting the better among these two categories (this classifier won't try to predict a percent score for these two categories). You have already trained V1 of a classifier with 10,000 news stories from the New York Times, which is one of 1000 new sources we would like the next version of our classifier (let's call it V2) to correctly categorize stories for. You would like to train a new classifier with the original 10,000 New York Times news stories and an additional 10,000 different news stories and no more. Below are approaches to generating the additional 10,000 pieces of train data for training V2.


Run our V1 classifier on 1 Million random stories from the 1000 news sources. Get the 10k stories where the V1 classifier’s output is closest to the decision boundary and get these examples labeled.
Get 10k random labeled stories from the 1000 news sources we care about.
Pick a random sample of 1 million stories from 1000 news sources and have them labeled. Pick the subset of 10k stories where the V1 classifier’s output is both wrong and farthest away from the decision boundary.
Ignore the difference in costs and effort in obtaining train data using the different methods described above. In terms of pure accuracy of classifier V2 when classifying a bag of new articles from 1000 news sources, what is likely to be the value of these different methods?How do you think the models will rank based on their accuracy?
Soln. Let's analyze each approach in terms of its potential impact on the accuracy of the V2 classifier when classifying news articles into the "information" and "entertainment" categories:
Approach 1: Run V1 on 1 Million random stories and select closest to decision boundary
Potential Impact: This approach focuses on the examples where the V1 classifier's output is close to the decision boundary. These examples are likely to be challenging for the current model, and correctly labeling them can improve the model's ability to handle borderline cases. However, this assumes that the V1 model's misclassifications near the decision boundary are genuine errors rather than noise.
Accuracy Expectation: This approach could improve the V2 model's accuracy, especially in handling ambiguous cases near the decision boundary.
Approach 2: Get 10k random labeled stories
Potential Impact: Randomly selecting labeled stories provides a diverse set of examples for training. It covers various topics, writing styles, and potential biases present in the dataset. However, it may not specifically target challenging cases or focus on the weaknesses of the current model.
Accuracy Expectation: This approach contributes to the overall diversity of the training set, which is generally beneficial. It may not lead to significant improvements in accuracy compared to focusing on challenging cases.
Approach 3: Random sample of 1 million stories, label, and select wrong cases farthest from the decision boundary
Potential Impact: This approach focuses on cases where the V1 classifier is both wrong and far from the decision boundary. Selecting such cases aims to identify instances where the model is confidently wrong. This can help the new model correct serious errors made by the V1 model.
Accuracy Expectation: This approach has the potential to address specific weaknesses of the V1 model by targeting instances where it confidently misclassifies examples. It may lead to improvements in accuracy, especially in cases where V1 is consistently making strong mistakes.
Overall Ranking Expectation:
The ranking in terms of accuracy improvement is subjective and depends on the specific characteristics of the dataset and the V1 model's strengths and weaknesses.
Approach 3 may have the highest potential for accuracy improvement, as it specifically targets confidently misclassified examples far from the decision boundary.
Approach 1 may also lead to improvements by focusing on challenging cases near the decision boundary.
Approach 2 contributes to overall diversity but may not provide as much targeted improvement as the other approaches.
In practice, it is advisable to experiment with different approaches, evaluate their impact on a validation set, and choose the strategy that yields the best results for the specific task at hand.

You wish to estimate the probability, $p$ that a coin will come up heads, since it may not be a fair coin. You toss the coin $n$ times and it comes up heads $k$ times. You use the following three methods to estimate $p$


Maximum Likelihood estimate (MLE)
Bayesian Estimate: Here you assume a continuous distribution uniform prior to $p$ from $[0,1]$ (i.e. the probability density function for the value of $p$ is uniformly $1$ inside this range and $0$ outside. Our estimate for $p$ will be the expected value of the posterior distribution of $p$. The posterior distribution is conditioned on these observations.
Maximum a posteriori (MAP) estimate: Here you assume that the prior is the same as (b). But we are interested in the value of $p$ that corresponds to the mode of the posterior distribution.
What are the estimates?
Soln. Let's denote the number of coin tosses as n and the number of times the coin comes up heads as k. The probability of getting heads in a single toss is p.
Maximum Likelihood Estimate (MLE):
The MLE for p is given by the ratio of the number of heads observed to the total number of coin tosses: MLE: pMLE=k/n
Bayesian Estimate:
For the Bayesian estimate, we assume a uniform prior on p from 0 to 1. The posterior distribution is a Beta distribution, and the Bayesian estimate is the expected value of this distribution:
Bayesian Estimate: pBayesian=(k+1)/(n+2)
This Bayesian estimate is often referred to as Laplace smoothing or Laplace correction.
Maximum a Posteriori (MAP) Estimate:
Similar to the Bayesian estimate, we assume a uniform prior on p from 0 to 1. The posterior distribution is a Beta distribution, and the MAP estimate is the mode (peak) of this distribution. The MAP estimate is given by:
MAP Estimate: pMAP=k/n
It's important to note that in this case, the MAP estimate coincides with the MLE because the uniform prior does not introduce any additional information.
In summary:
MLE:pMLE=k/n
Bayesian Estimate:pBayesian=(k+1)/(n+2)
MAP Estimate:pMAP=k/n

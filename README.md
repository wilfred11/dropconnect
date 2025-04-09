In this project the performance of two digit recognition models are compared statistically. The first model contains 2 linear layers, the second model contains 2 linear layers of which one is a DropConnectLayer (I am a bit doubtful with respect to this layer, but I was mainly trying to do some statistical testing.). A DropConnection layer is a layer that promotes generalization by randomly setting weights to zero when backpropagating. The results of the performances for the two models are somewhat similar (97.23% and 97.91% for 10000 individual tests), but the statistical test shows the results should be considered different with a confidence interval of 95% and 99%. The test statistic is calculated using the following formula. It uses a $\hat{p}$ that is a weighted mean p for both of the distributions.

The hypotheses are:

$H_0: p1 = p2 $ 
There is no statistical difference between the two binomial success values (97.23% and 97.91%).

$H_a: p1 \not= p2 $ 
There is a statistical difference between the two binomial success values (97.23% and 97.91%).

$z = \frac{p1-p2}{\sqrt{\hat{p}(1-\hat{p})*(\frac{1}{n1}+\frac{1}{n2})}}$

where 

$\hat{p} = \frac{p1n1+p2n2}{n1+n2}$

Using values p1=0.9723, p2=0.9791, n1=10000, n2=10000

abs(z) = 3.12

Using the following code snippet one can generate a standard normal distribution.
To get the critical value, the value above which the z-score should be considered different, use the allowed error (.5) and the ppf function.
The ppf function the percent point function returns the z-value at which some arbitrary part of the data is included in the normal curve.
As I want to assert whether the success values for the samples are different, and I have no expectancy whether one sample p will be bigger or smaller than the other I need to halve the allowed error to get a critical value at one side. This critical value is 


`from scipy.stats import norm

D=norm()

D.ppf(.975)`


calculate the corresponding critical region value to compare your test statistic too. For example, if you are testing this hypothesis at the 95% confidence level then you need to compare the absolute value of your test statistic against the critical region value 



Provided there are enough tests in the sample a binomial distribution can be approximated by a normal distribution. The mean for this normal distribution is n*p (number of tests times success rate) and its standard deviation is the square root of (n * p *(1-p)). These normal distribution approximations for both performance results are plotted in two histograms using 10000 randomly generated observations (every observation will contain the result for 10000 individual tests).

As can be seen, the randomly generated histograms have a very small overlap in their extreme tails, so when assuming the mean and standard deviation used to generate the samples are representative for all respective samples, be it of linear or DropConnect linear origin, than it is next to impossible to produce two samples (containing results for 10000 individual tests) that have their means closer to the overlapping tails so that 5% or 1%  of the samples could belong to the other sample. 

In other words the comparison of the two samples seems to indicate the DropConnect linear layer does a better job when recognizing digits of the MNIST dataset. When being able to get similar results using other samples it would be even more indicative. Using a technique like cross validation it would be possible to give a higher weight to these statistical tests.

![norm_binom](https://github.com/user-attachments/assets/e48d5919-5723-4156-acf9-4aa155b9ebdc)

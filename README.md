In this project the performance of two digit recognition models are compared statistically. The first model contains 2 linear layers, the second model contains 2 linear layers of which one is a DropConnectLayer (I am a bit doubtful with respect to this layer, but I was mainly trying to do some statistical testing.). A DropConnection layer is a layer that promotes generalization by randomly setting weights to zero when backpropagating. 

The results of the performances for the two models are somewhat similar (97.23% and 97.91% for 10000 individual tests), but the statistical test shows the results should be considered different with a confidence interval of 95% and even 99%. 

### Testing whether two models differ considering performances
The test statistic is calculated using the following formula. It uses a $\hat{p}$ that is a weighted mean p for both of the distributions.
#### Hypotheses
The hypotheses are:

$H_0: p1 = p2 $ 
There is no statistical difference between the two binomial success values (97.23% and 97.91%).

$H_a: p1 \not= p2 $ 
There is a statistical difference between the two binomial success values (97.23% and 97.91%).

#### Calculating z-score

$z-score = \frac{p1-p2}{\sqrt{\hat{p}(1-\hat{p})*(\frac{1}{n1}+\frac{1}{n2})}}$

where 

$\hat{p} = \frac{p1n1+p2n2}{n1+n2}$

Using values p1=0.9723, p2=0.9791, n1=10000, n2=10000

abs(z-score) = 3.12

#### Get the critical value for an allowed error
As I want to assert whether the success values for the two samples are different, and I have no expectancy whether one sample p will be bigger or smaller than the other, I need to halve the allowed error to get a critical value at one side. 

This critical value is 1.96 when using an allowed error of 0.05. 


Half of 0.05 (allowed error) is 0.025, and 1-0.025 is 0.975. When using 0.975 in the ppf function of the norm, it returns the critical value of 1.96. The idea behind this critical value is that 2.5% of the data will be in the region beyond the critical value (one tail). The other 2.5% of the data will be in the other tail, beyond the critical value of -1.96. 

D=norm() will create a standardized normal distribution (with mean =0, and sd=1), while ppf(0.975) (percent point function) returns the critical value that includes 97.5% of the data.

`from scipy.stats import norm`

`D=norm()`

`crit_value = D.ppf(.975)`

As the absolute value of the z_score is above the critical value of 1.96 the p1 and p2 are considered statistically different with a confidence level of 95%.

### Normal distribution to approximate a binomial distribution

Provided there are enough tests in the sample a binomial distribution can be approximated by a normal distribution. The mean for this normal distribution is n*p (number of tests times success rate) and its standard deviation is the square root of (n * p *(1-p)). These normal distribution approximations for both performance results are plotted in two histograms using 10000 randomly generated observations (every observation will contain the result for 10000 individual tests).

As can be seen, the randomly generated histograms have a very small overlap in their extreme tails, so when assuming the mean and standard deviation used to generate the samples are representative for all respective samples, be it of linear or DropConnect linear origin, than it is next to impossible to produce two samples (containing results for 10000 individual tests) that have their means closer to the overlapping tails so that 5% or 1%  of the samples could belong to the other sample. 

In other words the comparison of the two samples seems to indicate the DropConnect linear layer does a better job when recognizing digits of the MNIST dataset. When being able to get similar results using other samples it would be even more indicative. Using a technique like cross validation it would be possible to give a higher weight to these statistical tests.

![norm_binom](https://github.com/user-attachments/assets/e48d5919-5723-4156-acf9-4aa155b9ebdc)

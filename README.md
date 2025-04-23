In this project the performance of two digit recognition models are compared statistically. The first model contains 2 linear layers, the second model contains 2 linear layers of which one is a DropConnectLayer (I am a bit doubtful with respect to this layer, but I was mainly trying to do some statistical testing). A DropConnection layer is a layer that promotes generalization by randomly setting weights, linked to this layer, to zero when backpropagating. 

The results of the performances for the two models are somewhat similar (97.23% and 97.91% for 10000 individual tests), but the statistical test shows the results should be considered different with a confidence interval of 95% and even 99%. 

### Testing whether two models differ in terms of performance
The test statistic is calculated using the following formula, this formula assumes both permormance rates are in the same distribution. It uses a $\hat{p}$ (chance of success) that is a weighted mean p for both of the chances of success to calculate a common mean standard deviation. The zscore is an expression of how many standard errors the performance rate difference amounts for. 

$zscore = \frac{p1-p2}{\sqrt{\hat{p}(1-\hat{p})\times(\frac{1}{n1}+\frac{1}{n2})}}$

where 

$\hat{p} = \frac{p1 \times n1+p2\times n2}{n1+n2}$

#### Hypotheses
The hypotheses are:

$H_0: p1 = p2 $ 
There is no statistical difference between the two binomial success values (97.23% and 97.91%).

$H_a: p1 \not= p2 $ 
There is a statistical difference between the two binomial success values (97.23% and 97.91%).

#### Calculating z-score

Using values p1=0.9723, p2=0.9791, n1=10000, n2=10000 and the formula for the zscore.

abs(zscore) = 3.12

#### Get the critical value for an allowed error
As I want to assert whether the success values for the two samples are different, and I have no expectancy whether one sample p will be bigger or smaller than the other, I need to halve the allowed error to get a critical value at one side. 

If the allowed error is 0.5. Than half of 0.05 (allowed error) is 0.025, and 1-0.025 is 0.975. So I need to find the positive z-value that contains 97.5% of the data, the so-called critical value when allowing an error of 0.05. The idea behind this critical value is that 2.5% of the data will be in the region beyond the critical value (the right tail). The other 2.5% of the data will be in the other tail (the left tail), beyond the negative critical value. 

D=norm() will create a standardized normal distribution (with mean =0, and sd=1), while D.ppf(0.975) (ppf=percent point function) returns the critical value that includes 97.5% of the data.

`from scipy.stats import norm`

`D=norm()`

`crit_value = D.ppf(.975)`
#### Critical value

When using 0.975 in the ppf function of the norm, it returns the critical value of 1.96.

As the absolute value of the zscore (3.12) is above the critical value of 1.96, p1 and p2 are considered statistically different with a confidence level of 95%.

#### Get the p-value for a zscore
The p-value expresses the chance to get a value as extreme as the zscore (3.12). To get this value I can use the cdf function for the norm. D.cdf(3.12) returns a percental amount of data that is present between - $\infty$ and 3.12, when substracting 0.5 from this value I get the amount of data that is present between the mean and 3.12, doubling that value will return the percental amount of data present between -3.12 and +3.12. When deducting this value from 1 the p-value is found.

The formula for the p-value is :

`from scipy.stats import norm`

`D=norm()`

`pvalue = 1-2X(D.cdf(abs(zscore))-.5)`

where 
zscore=3.12

The value for D.cdf(abs(zscore)) is 99.9%. This value minus 50% is 49.9%. 49.9% times two is 99.8..%. So 99.82% of the data is between -3.12 and +3.12. 1 minus 99.82% is 0.18%. So only 0.18% of the data is further away from the mean than our zscore, this zscore represents the difference between our two sample performance rates expressed in a standardized error, only 0.18% of the data is outside our boundary of -3.12 and +3.12. The pvalue is 0.0018, this very small value expresses the impossiblity to produce this zscore of 3.12 without using another distribution with a different mean and sd.

#### Two samples are different

In other words the comparison of the two samples seems to indicate the DropConnect linear layer does a better job when recognizing digits of the MNIST dataset. When being able to get similar results using other, real samples it would be even more indicative. Using a technique like k-fold cross validation on the full dataset (containing 70000 items), it would be possible to give a higher weight to the statistical test described here, this by producing more performance rates.

### Normal distribution to approximate a binomial distribution

Provided there are enough tests in the sample a binomial distribution can be approximated by a normal distribution. The mean for this normal distribution is n*p (number of tests times success rate) and its standard deviation is the square root of (n * p *(1-p)). These normal distribution approximations for both performance results are plotted in two histograms using 10000 randomly generated observations (every observation will contain the result for 10000 individual tests).

#### Generate 10000 samples

To create a distribution with some mean and standard deviation use the following code.

`from scipy.stats import norm`

`D = norm(mean, sd)`

To generate 10000 sample results, this is 10000 samples of 10000 individual tests, the following code is used.
The distribution created before is used to create 10000 random variates (D.rvs(size=10000).

`x_virtual = D.rvs(size=10000)`

`x_min = min(x)`
 
 `x_max = max(x)`
 
 `plt.hist(x, bins=int(x_max)- int(x_min), color='red', alpha=0.5, label='virtual histogram')`

#### Small overlap

As can be seen, the randomly generated histograms for both models have a very small overlap in their extreme tails. So when assuming the respective mean and standard deviation used to generate the samples are representative for all respective samples, be it of linear or DropConnect linear origin, than it is next to impossible to produce two samples (containing results for 10000 individual tests) from the two distributions that lie closer to  each other so that at least 5% or 1%  of the samples could belong to the other sample. 

#### Generate normal curve

To create the normal curve from a distribution D, I need to generate some inputs using np.linspace. This function returns a evenly spaced numbers over an interval. The function will create numbers between the minimum number of successful tests and the maximum number of successful tests for all samples.

`x_successful_tests = np.linspace(start=x_min, stop=x_max, num=int(x_max)- int(x_min))`

To generate the number of samples that contains a specific number of successful individual tests use the pdf function (probability distribution function), it will return a probability for every number of successes. The probability times 10000 the number of samples will result in a number that lies on the normal curve for this distribution. 

`plt.plot(x_successful_tests, D.pdf(x_successful_tests)*10000, color="darkgreen", label="normal curve")`


![bin_test](https://github.com/user-attachments/assets/448c57c5-6de1-4a53-a6db-0b05f98ef134)

### Power

There is a function to get the p-value and zscore for our proportions and performance rates.

`zscore, pvalue = statsmodels.api.stats.proportions_ztest(count=[9723,9791]], nobs=[10000,10000])`

The zscore using this function is -3.12.

The power of proportional z-test is shown in the image below. It is shows the chance to correctly reject the null hypothesis. This chance is given by the part of the H_alternative normal graph (mean=z-score, std=1) that is left of the negative critical value and the part of the H_alternative normal graph that is right of the positive critical value. This chance is calculated using a cdf function. But first calculate the critical values for 0.975. The 

zalpha is calculated using following code

`Dnull=norm()`

`zalpha=Dnull.ppf(0.975)`

zalpha is 1.96, this value is a typical value when using the standard normal distribution.

The respective powers (left, right) are calculated using the alternative distribution.

 `Dalt= norm(-3.12,1)`
 
 `powerl=Dalt.cdf(-z_alpha)`
 
 `powerr= 1- Dalt.cdf(z_alpha)`
 
powerl represents the part left of the negative critical value.
powerr represents the part right of the positive critical value.

The total power is 0.88 this is a good value. As power is a value between 0 and 1. In this case the power is made up almost completely of chances left of the negative critical value (-1.96). 

The value could also have been calculated using statsmodels function 

`statsmodels_power= statsmodels.stats.proportion.power_proportions_2indep( 0.0068, 0.9723, 10000, ratio=1, alpha=0.05, value=0, alternative='two-sided', return_results=False)`

![power_n](https://github.com/user-attachments/assets/91fc06df-ab5b-44f6-a789-64d98dbfbcbc)




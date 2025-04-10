import math

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from statistics import  NormalDist

def p_hat(n1,n2,p1,p2):
    p_hat = ((n1 * p1) + (p2 * n1)) / (n1 + n1)
    return p_hat

def test(n1,n2,p1,p2, err):
    p_h= p_hat(n1,n2,p1,p2)
    z = (p1 - p2) / math.sqrt(p_h * (1 - p_h) * ((1 / n1) + (1 / n2)))
    D = norm()
    crit_p = 1 - (err/2)
    print("crit_p: "+ str(crit_p))
    crit_value = D.ppf(crit_p)
    print("critical value: " + str(crit_value))
    p_value = 1-2*(D.cdf(abs(z))-.5)
    print("p_value: "+str(p_value))

    if abs(z)>crit_value:
        print("z>critical value: "+ "z="+ str(abs(z)) +"> critical value="+str(crit_value))
        print("reject the null hypothesis, p1 differs from p2 with a confidence level of "+ str(1- err) )
    else:
        print("z<=critical value: " + "z=" + str(abs(z)) + "<= critical value=" + str(crit_value))
        print("don't reject the null hypothesis, p1 equals p2 with a confidence level of "+ str(1- err) )

def show_binoms_as_normal_approx(n1,n2,p1,p2):
    print(
        "Provided n is large enough, norm(mu,sd) is a good approximation for binom(n, p) where mu = np and var = np (1 – p).")
    mu1 = n1 * p1
    print("n1: "+ str(n1))
    print("p1: "+str(p1))
    print("mu1:" + str(mu1))
    var1 = n1 * p1 * (1 - p1)
    sd1= math.sqrt(var1)
    print("sd1: " + str(sd1))
    D = norm(mu1, sd1)
    x1_virtual = D.rvs(size=10000)
    x1_virtual_min = min(x1_virtual)
    x1_virtual_max = max(x1_virtual)
    plt.hist(x1_virtual, bins=int(x1_virtual_max)- int(x1_virtual_min), color='red', alpha=0.5, label='Virtual histogram performance sample 1')
    print("n2: " + str(n2))
    print("p2: " + str(p2))
    mu2 = n2 * p2
    print("mu2:" + str(mu2))
    var2 = n2 * p2 * (1 - p2)
    sd2 = math.sqrt(var2)

    print("sd2: " + str(sd2))
    D1 = norm(mu2, sd2)
    x2_virtual = D1.rvs(size=10000)
    x2_virtual_min = min(x2_virtual)
    x2_virtual_max = max(x2_virtual)
    axes = plt.gca()

    axes.set_xlim([int(min(x2_virtual_min, x1_virtual_min)), int(max(x2_virtual_max,x1_virtual_max))])
    axes.set_ylim([0, 450])
    plt.hist(x2_virtual, bins=int(x2_virtual_max)- int(x2_virtual_min), color='green', alpha=0.5, label='Virtual histogram performance sample 2')
    x2 = np.linspace(start=x2_virtual_min, stop=x2_virtual_max, num=int(x2_virtual_max)- int(x2_virtual_min))
    #print("x2")
    #print(x2)
    x1 = np.linspace(start=x1_virtual_min, stop=x1_virtual_max, num=int(x1_virtual_max) - int(x1_virtual_min))
    plt.xlabel("number of succesful tests per sample")
    plt.ylabel("number of samples")
    plt.plot(x1, D.pdf(x1) * 10000, color="darkred", label="sample 1, mean=" + str(mu1) + ", sd=" + str(math.sqrt(var1)))
    plt.plot(x2, D1.pdf(x2)*10000, color="darkgreen", label="sample 2, mean="+str(mu2)+", sd="+str(math.sqrt(var2)))
    plt.rcParams["figure.figsize"] = (20, 45)
    plt.legend()
    #plt.fill_between(x, D1.pdf(x), color="r")
    plt.show()

    print(NormalDist(mu=mu1, sigma=sd1).overlap(NormalDist(mu=mu2, sigma=sd2)))



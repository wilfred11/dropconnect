import math
from matplotlib import pyplot as plt
from scipy.stats import norm


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
    plt.hist(x1_virtual, bins=25, color='red', alpha=0.5, label='Virtual classification performance sample 1')
    print("n2: " + str(n2))
    print("p2: " + str(p2))
    mu2 = n2 * p2
    print("mu2:" + str(mu2))
    var2 = n2 * p2 * (1 - p2)
    sd2 = math.sqrt(var2)

    print("sd2: " + str(sd2))
    D = norm(mu2, sd2)
    x2_virtual = D.rvs(size=10000)
    plt.hist(x2_virtual, bins=25, color='green', alpha=0.5, label='Virtual classification performance sample 2')
    plt.show()



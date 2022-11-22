"""Given previous number of agents and run times, estimate future run times."""
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize


def first_order(x, a, b):
    """First-order function."""
    return a * x + b


def second(x, a, b):
    """Second-order function."""
    return a * x ** 2 + b

def exponential(x, a, b):
    """Exponential function."""
    return x ** a + b


# based on previous running times on the beast,
# estimate how long the run time for a different number of agents
# will be time for total running

x_new = 50
n_repeats = 10
n_epochs = 50

# number of agents
xs = [1, 2, 3, 4, 5, 10, 15, 20, 30]

# total running time per epoch
ys = [3.41, 6.17, 9.05, 11.32, 14.42, 31.10, 48.36, 70.15, 115.04]


popt, pcov = optimize.curve_fit(first_order, xs, ys)
a, b = popt
print(f"a, b first errors = {np.sqrt(np.diag(pcov))}")

popt, pcov = optimize.curve_fit(second, xs, ys)
c, d = popt
print(f"c,d second errors = {np.sqrt(np.diag(pcov))}")

popt, pcov = optimize.curve_fit(exponential, xs, ys)
e, f = popt
print(f"e,f exponential errors = {np.sqrt(np.diag(pcov))}")

plt.figure()
plt.plot(xs, ys, 'o', label='data')
plt.plot(list(range(x_new)), [a * x + b for x in list(range(x_new))],
         label='first order')
plt.plot(list(range(x_new)), [c * x ** 2 + d for x in list(range(x_new))],
         label='second order')
plt.plot(list(range(x_new)), [x ** e + f for x in list(range(x_new))],
         label='exponential')
plt.legend()
plt.show()


for f, name, params in zip([first_order, second, exponential],
                           ['first_order', 'second', 'exponential'],
                           [[a, b], [c, d], [e, f]]):
    predict = f(x_new, *params)
    print(f"{name}: {x_new} agents predicted "
          f"time: {predict * n_epochs * n_repeats / 60 / 60} hours")

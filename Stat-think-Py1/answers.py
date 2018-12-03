#EX1

#(5) [...]It's better to jump into conclusions



################################################################################
#EX2

# (3) A nice looking plot[...]



################################################################################
#EX3

# Import plotting modules
import matplotlib.pyplot as plt
import seaborn as sns


# Set default Seaborn style

sns.set()

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)
plt.show()
# Show histogram






################################################################################
#EX4


# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length)

# Label axes
plt.xlabel('petal length (cm)')
plt.ylabel('count')

# Show histogram
plt.show()


################################################################################
#EX5


# Import numpy

import numpy as np

# Compute number of data points: n_data

n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins

n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins

n_bins = int(n_bins)

# Plot the histogram
_ = plt.hist(versicolor_petal_length, bins = n_bins)

# Label axes
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('count')

# Show histogram
plt.show()



################################################################################
#EX6

# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x='species', y='petal length (cm)', data = df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')
# Show the plot

plt.show()



################################################################################
#EX7


#(3) Virginica tend to be the longest




################################################################################
#EX8

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


################################################################################
#EX9

# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')

# Label the axes
_ = plt.xlabel('Petal lengths (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()




################################################################################
#EX10


# Compute ECDFs
x_set, y_set = ecdf(setosa_petal_length)
x_vers, y_vers = ecdf(versicolor_petal_length)
x_virg, y_virg = ecdf(virginica_petal_length)


# Plot all ECDFs on the same plot

_ = plt.plot(x_set, y_set, marker = '.', linestyle = 'none')
_ = plt.plot(x_vers, y_vers, marker = '.', linestyle = 'none')
_ = plt.plot(x_virg, y_virg, marker = '.', linestyle = 'none')

# Annotate the plot
plt.legend(('setosa', 'versicolor', 'virginica'), loc='lower right')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Display the plot
plt.show()



################################################################################
#EX11


# (2) An outlier can significantly affect the value of the mean, but not the median.



################################################################################
#EX12

# Compute the mean: mean_length_vers
mean_length_vers = np.mean(versicolor_petal_length)
# Print the result with some nice formatting
print('I. versicolor:', mean_length_vers, 'cm')



################################################################################
#EX13

# Specify array of percentiles: percentiles
percentiles = ([2.5,25,50,75,97.5])

# Compute percentiles: ptiles_vers
ptiles_vers = np.percentile(versicolor_petal_length, percentiles)

# Print the result
print(ptiles_vers)



################################################################################
#EX14


# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot

plt.show()



################################################################################
#EX15


# Plot the ECDF
_ = plt.plot(x_vers, y_vers, '.')
_ = plt.xlabel('petal length (cm)')
_ = plt.ylabel('ECDF')

# Overlay percentiles as red diamonds.
_ = plt.plot(ptiles_vers, percentiles/100, marker='D', color='red',
         linestyle='none')

# Show the plot

plt.show()


################################################################################
#EX16

# Array of differences to mean: differences
differences = versicolor_petal_length - np.mean(versicolor_petal_length)

# Square the differences: diff_sq
diff_sq = differences**2

# Compute the mean square difference: variance_explicit
variance_explicit = np.mean(diff_sq)

# Compute the variance using NumPy: variance_np
variance_np = np.var(versicolor_petal_length)

# Print the results
print(variance_explicit, variance_np)





################################################################################
#EX17

# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(np.sqrt(variance))

# Print the standard deviation
print(np.std(versicolor_petal_length))




################################################################################
#EX18


# Compute the variance: variance
variance = np.var(versicolor_petal_length)

# Print the square root of the variance
print(np.sqrt(variance))

# Print the standard deviation
print(np.std(versicolor_petal_length))



################################################################################
#EX19

#d,c,b (3)




################################################################################
#EX20

# Compute the covariance matrix: covariance_matrix

covariance_matrix = np.cov(versicolor_petal_length,versicolor_petal_width)

# Print covariance matrix
print(covariance_matrix)

# Extract covariance of length and width of petals: petal_cov
petal_cov = covariance_matrix[0,1] and covariance_matrix[1,0]

# Print the length/width covariance

print(petal_cov)



################################################################################
#EX21

def pearson_r(x, y):
    """Compute Pearson correlation coefficient between two arrays."""
    # Compute correlation matrix: corr_mat
    corr_mat = np.corrcoef(x,y)

    # Return entry [0,1]
    return corr_mat[0,1]

# Compute Pearson correlation coefficient for I. versicolor: r
r = pearson_r(versicolor_petal_length, versicolor_petal_width)

# Print the result
print(r)



################################################################################
#EX22

#All of these (4)


################################################################################
#EX23

#Probabilistic language is not very precise (2)



################################################################################
#EX24

# Seed the random number generator
np.random.absolute_importseed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()




################################################################################
#EX25

# Seed the random number generator
np.random.absolute_importseed(42)

# Initialize random numbers: random_numbers
random_numbers = np.empty(100000)

# Generate random numbers by looping over range(100000)
for i in range(100000):
    random_numbers[i] = np.random.random()

# Plot a histogram
_ = plt.hist(random_numbers)

# Show the plot
plt.show()




################################################################################
#EX26

# Seed random number generator
np.random.seed(42)

# Initialize the number of defaults: n_defaults
n_defaults = np.empty(1000)

# Compute the number of defaults
for i in range(1000):
    n_defaults[i] = perform_bernoulli_trials(100,0.05)


# Plot the histogram with default number of bins; label your axes
_ = plt.hist(n_defaults, normed=True)
_ = plt.xlabel('number of defaults out of 100 loans')
_ = plt.ylabel('probability')

# Show the plot
plt.show()


################################################################################
#EX27

# Compute ECDF: x, y
x,y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker = '.', linestyle='none')
_ = plt.xlabel('X')
_ = plt.ylabel('Y')


# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money

n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))


################################################################################
#EX28


# Compute ECDF: x, y
x,y = ecdf(n_defaults)

# Plot the ECDF with labeled axes
_ = plt.plot(x, y, marker = '.', linestyle='none')
_ = plt.xlabel('X')
_ = plt.ylabel('Y')


# Show the plot
plt.show()

# Compute the number of 100-loan simulations with 10 or more defaults: n_lose_money

n_lose_money = np.sum(n_defaults >= 10)

# Compute and print probability of losing money
print('Probability of losing money =', n_lose_money / len(n_defaults))




################################################################################
#EX29


# Compute bin edges: bins
bins = np.arange(0, max(n_defaults) + 1.5) - 0.5

# Generate histogram
_ = plt.hist(n_defaults, normed = True, bins = bins)

# Label axes
_ = plt.xlabel("Bins")
_ = plt.ylabel("PMF")

# Show the plot

plt.show()



################################################################################
#EX30


# Draw 10,000 samples out of Poisson distribution: samples_poisson

samples_poisson = np.random.poisson(10, size = 10000)


# Print the mean and standard deviation
print('Poisson:     ', np.mean(samples_poisson),
                       np.std(samples_poisson))

# Specify values of n and p to consider for Binomial: n, p

n = [20, 100, 1000]
p = [0.5, 0.1, 0.01]

# Draw 10,000 samples for each n,p pair: samples_binomial
for i in range(3):
    samples_binomial = np.random.binomial(size = 10000, n = n[i], p = p[i])

    # Print results
    print('n =', n[i], 'Binom:', np.mean(samples_binomial),
                                 np.std(samples_binomial))




################################################################################
#EX31

#Both Binomial and Poisson, though Poisson is easier to model and compute. (4)



################################################################################
#EX32

# Draw 10,000 samples out of Poisson distribution: n_nohitters

n_nohitters = np.random.poisson(251/115, size = 10000)

# Compute number of samples that are seven or greater: n_large
n_large = np.sum(n_nohitters >= 7)

# Compute probability of getting seven or more: p_large
p_large = n_large/10000

# Print the result
print('Probability of seven or more no-hitters:', p_large)



################################################################################
#EX32


#x is more likely than not less than 10. (1)



################################################################################
#EX33

#0.25(1)


################################################################################
#EX34

# Draw 100000 samples from Normal distribution with stds of interest: samples_std1, samples_std3, samples_std10
samples_std1 = np.random.normal(20 ,1 ,100000)
samples_std3 = np.random.normal(20 ,3 ,100000)
samples_std10 = np.random.normal(20 ,10 ,100000)

# Make histograms
_ = plt.hist(samples_std1, normed = True, histtype = 'step', bins = 100)
_ = plt.hist(samples_std3, normed = True, histtype = 'step', bins = 100)
_ = plt.hist(samples_std10, normed = True, histtype = 'step', bins = 100)

# Make a legend, set limits and show plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'))
plt.ylim(-0.01, 0.42)
plt.show()


################################################################################
#EX35

# Generate CDFs
x_std1, y_std1 = ecdf(samples_std1)
x_std3, y_std3 = ecdf(samples_std3)
x_std10, y_std10 = ecdf(samples_std10)


# Plot CDFs
_ = plt.plot(x_std1, y_std1, marker = '.', linestyle = 'none')
_ = plt.plot(x_std3, y_std3, marker = '.', linestyle = 'none')
_ = plt.plot(x_std10, y_std10, marker = '.', linestyle = 'none')
# Make a legend and show the plot
_ = plt.legend(('std = 1', 'std = 3', 'std = 10'), loc='lower right')
plt.show()


################################################################################
#EX36


#mean=3, std=1 (1)

################################################################################
#EX37

# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, 10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()


################################################################################
#EX38

# Take a million samples out of the Normal distribution: samples
samples = np.random.normal(mu, sigma, 10**6)

# Compute the fraction that are faster than 144 seconds: prob
prob = np.sum(samples < 144)/len(samples)

# Print the result
print('Probability of besting Secretariat:', prob)




################################################################################
#EX39

#Exponential(2)



################################################################################
#EX40
#Exponential: A horse as fast as Secretariat is a rare event, which
#can be modeled as a Poisson process, and the waiting time between
# arrivals of a Poisson process is Exponentially distributed. (4)



################################################################################
#EX41



def successive_poisson(tau1, tau2, size=1):
    """Compute time for arrival of 2 successive Poisson processes."""
    # Draw samples out of first exponential distribution: t1
    t1 = np.random.exponential(tau1, size)

    # Draw samples out of second exponential distribution: t2
    t2 = np.random.exponential(tau2, size)

    return t1 + t2

################################################################################
#EX42

# Draw samples of waiting times: waiting_times
waiting_times = successive_poisson(764,715,10**5)

# Make the histogram
_ = plt.hist(waiting_times, bins=100, normed=True, histtype='step')


# Label axes

_ = plt.xlabel('waiting time')
_ = plt.ylabel('pdf')

# Show the plot
plt.show()



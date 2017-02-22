from numpy.random import randn

noise = randn(50000,1);  # Normalized white Gaussian noise
x = filter([1], [1,1/2, 1/3, 1/4], noise)
print(x)
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234567890)

def plotDensity(samples, k, n, xmin, xmax):
    densX = [x/100 for x in range(xmin, xmax)]
    densY = []
    for x in densX:
        densY.append(calDensity(samples, k , x))
    plt.plot(densX, densY)


def getSamplesGauss(n):
    #loc: noraml distribution, scale: standard deviation, size: number of samples to return
    return np.random.normal(loc=1, scale=2, size=n)

def getSamplesUniform(n):
        r = np.random.random(n)
        samples = []
        for ri in r:
            if ri < 0.4:
                samples.append(np.random.random() + 2)
            else:
                samples.append(np.random.random() * 5 + 15)
        return np.asarray(samples)

def calDensity(samples, k, x):
    dist = abs(samples - x)
    n = len(samples)
    ki = np.argsort(dist)
    r = dist[ki[k-1]]

    return (k / (2 * n * r))

def fwhm2sigma(FWHM):
    return FWHM / np.sqrt(8 * np.log(2))

def getSamplesKernel(n):
    samplesX = np.arange(n)
    samplesY = getSamplesUniform(n)
    fwhm = 4
    sigma = fwhm2sigma(fwhm)
    samples = []
    for x in samplesX:
        kernelPos = np.exp(-(samplesX - x) ** 2 / (2 * sigma ** 2))
        kernelPos = kernelPos / sum(kernelPos)
        samples.append(sum(samplesY * kernelPos))
    return np.asarray(samples)

def step1():
    for (k,n) in [(1,1) , (2, 10) , (10, 10) , (10, 1000) , (100, 1000) , (50000, 50000)]:
        samples = getSamplesGauss(n)
        plotDensity(samples, k, n, -500, 500)
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.title('KNN Density With K = ' + str(k) + ' N = ' + str(n))
        plt.show()

def step2():
    for (k,n) in [(1,1) , (2, 10) , (10, 10) , (10, 1000) , (100, 1000) , (50000, 50000)]:
        samples = getSamplesUniform(n)
        plotDensity(samples, k, n, 0, 2200)
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.title('KNN Density (Uniform) With K = ' + str(k) + ' N = ' + str(n))
        plt.show()

def step3():
    for (k,n) in [(1,1) , (2, 10) , (10, 10) , (10, 1000) , (100, 1000) , (50000, 50000)]:
        samples = getSamplesKernel(n)
        plotDensity(samples, k, n, 0, 2200)
        plt.xlabel('X')
        plt.ylabel('Density')
        plt.title('KNN Density (Kernel) With K = ' + str(k) + ' N = ' + str(n))
        plt.show()

def main():
    #step1()
    #step2()
    step3()


if __name__ == "__main__":
    main()

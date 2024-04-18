from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import itertools as it
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    samples = np.random.normal(10,1,size = 1000)
    univariate_gaussian = UnivariateGaussian().fit(samples)
    print((np.round(univariate_gaussian.mu_,3), np.round(univariate_gaussian.var_, 3)))

    # Question 2 - Empirically showing sample mean is consistent

    X = np.linspace(10,1000,100).astype(int)
    Y = []
    for size in X:
        size_first_samples = samples[:size]
        Y.append(abs(10 - univariate_gaussian.fit(size_first_samples).mu_))

    go.Figure(go.Scatter(x = X,y = Y, mode= "markers", marker= dict(color = "red")), layout=go.Layout(xaxis_title = "Sample Size",
        yaxis_title = "Absolute Difference from mu",title={
        'text': r"$\text{absolute distance of the estimated expectation from }\mu$",
        'x': 0.5,
        'y': 0.9,
        'xanchor': 'center',
        'yanchor': 'top',
        'font': dict(
            size=50,
            color='black')
    })
).show()


# Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(go.Scatter(x=samples, y=univariate_gaussian.pdf(samples), mode="markers", marker=dict(color="purple")),
              layout=go.Layout(xaxis_title="Sample value",
                               yaxis_title="PDF", title={
                      'text': "empirical PDF",
                      'x': 0.5,
                      'y': 0.9,
                      'xanchor': 'center',
                      'yanchor': 'top',
                      'font': dict(
                          size=24,
                          color='black')
                  })
              ).show()


#
def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0,0,4,0])
    sigma = np.array([[1,0.2,0,0.5],[0.2, 2, 0, 0],[0,0,1,0], [0.5,0,0,1]])
    samples = np.random.multivariate_normal(mean=[0, 0, 4, 0], cov=sigma, size=1000)
    multivariate_gaussian = MultivariateGaussian().fit(samples)
    print(np.round(multivariate_gaussian.mu_,3))
    print(np.round(multivariate_gaussian.cov_,3))
    multivariate_gaussian.pdf(samples[:100])


#
#
#     # Question 5 - Likelihood evaluation
    mues = np.linspace(-10, 10, 200)
    data = np.array([[multivariate_gaussian.log_likelihood(np.array([f1, 0, f3, 0]), sigma, samples) for f3 in mues] for f1 in
            mues])

    heatmap = go.Figure(go.Heatmap(x=mues, y=mues, z=data, colorscale= "rainbow"

                                   ),
                        layout=dict(template="simple_white",
                                    title="Log-likelihood of a multivariate Gaussian distribution as a function of the"
                                          " expectation values of 1st and 3rd features of mu",
                                    xaxis_title="Feature 3",
                                    yaxis_title="Feature 1"))

    heatmap.show()


#
#     # Question 6 - Maximum likelihood
    print(np.round(mues[list(np.unravel_index(data.argmax(), data.shape))],3))


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()

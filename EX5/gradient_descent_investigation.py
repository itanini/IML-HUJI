import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type
from sklearn.metrics import roc_curve, auc
from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IMLearn.metrics.loss_functions import misclassification_error
from IMLearn.model_selection.cross_validate import cross_validate
from IMLearn.utils.utils_new import decision_surface


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    obj_val = []
    param_lst = []
    delta_lst = []
    iter_lst = []

    def func(**kwargs):
        obj_val.append(kwargs['val'])
        param_lst.append(kwargs['weights'])
        delta_lst.append(np.linalg.norm(kwargs['weights']))
        iter_lst.append(kwargs['t'])

    return func, obj_val, param_lst, delta_lst,iter_lst


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    l1_min_loss_lst = []
    l2_min_loss_lst = []

    # Q1

    Q1(init, l1_min_loss_lst, l2_min_loss_lst)
    # Q3
    for et in etas:
        Q3(et, init, l1_min_loss_lst, l2_min_loss_lst)
    # Q4
    print('min loss l1 : ' + str(np.min(l1_min_loss_lst)))
    print('min loss l2 : ' + str(np.min(l2_min_loss_lst)))
    return


def Q3(et, init, l1_min_loss_lst, l2_min_loss_lst):
    model_l1 = set_L1(et, init, l1_min_loss_lst)
    model_l2 = set_L2(et, init, l2_min_loss_lst)
    go.Figure(data=[go.Scatter(x=model_l1.callback_[4], y=model_l1.callback_[3])],
              layout=dict(title=dict(text="convergence rate L1, eta = " + str(et)))).write_image(
        f'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/l1_et={et}.png')
    go.Figure(data=[go.Scatter(x=model_l2.callback_[4], y=model_l2.callback_[3])],
              layout=dict(title=dict(text="convergence rate L2, eta = " + str(et)))).write_image(
        f'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/l2_et={et}.png')


def Q1(init, l1_min_loss_lst, l2_min_loss_lst):
    et = 0.01
    model_l1 = set_L1(et, init, l1_min_loss_lst)
    model_l2 = set_L2(et, init, l2_min_loss_lst)
    plot_descent_path(module=L1, descent_path=np.array(model_l1.callback_[2]),
                      title='l1 eta = ' +
                            str(et)).write_image(
        'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/decent_path_l1_et=0.01.png')
    plot_descent_path(module=L2, descent_path=np.array(model_l2.callback_[2]),
                      title='l2 eta = ' +
                            str(et)).write_image(
        'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/descent_path_l2_et=0.01.png')


def set_L1(et, init, l1_min_loss_lst):
    model_l1 = GradientDescent(FixedLR(et), callback=get_gd_state_recorder_callback())
    l1 = model_l1.fit(L1(init.copy()), X=None, y=None)
    l1_min_loss_lst.append(np.linalg.norm(l1))
    return model_l1


def set_L2(et, init, l2_min_loss_lst):
    model_l2 = GradientDescent(FixedLR(et), callback=get_gd_state_recorder_callback())
    l2 = model_l2.fit(L2(init.copy()), X=None, y=None)
    l2_min_loss_lst.append(np.linalg.norm(l2))
    return model_l2


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    min_loss_lst = []
    fig = make_subplots(rows=2, cols=2, subplot_titles=['gamma = 0.9', 'gamma = 0.95', 'gamma = 0.99', 'gamma = 1'])
    for i, gam in enumerate(gammas):
        model_l1 = GradientDescent(ExponentialLR(base_lr=eta, decay_rate=gam),
                                   callback=get_gd_state_recorder_callback())
        l1 = model_l1.fit(L1(init.copy()), X=None, y=None)
        min_loss_lst.append(np.linalg.norm(l1))
        fig.add_trace(go.Scatter(x=model_l1.callback_[4], y=model_l1.callback_[3], mode='markers+lines'),
                      row=i // 2 + 1, col=i % 2 + 1)
    fig.write_image(f'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/conv_for_dif_gammas.png')
    print('the min loss is : ' + str(np.min(min_loss_lst)))

    # Plot descent path for gamma=0.95
    model_l1 = GradientDescent(ExponentialLR(base_lr=eta, decay_rate=0.95), callback=get_gd_state_recorder_callback())
    l1 = model_l1.fit(L1(init.copy()), X=None, y=None)
    # print(model_l1.callback_[2])
    plot_descent_path(module=L1, descent_path=np.array(model_l1.callback_[2]),
                      title='l1 eta = ' + str(eta) + ' decay rate = 0.95').write_image\
        (f'C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/descent_plot_for_eta=0.1_decay=0.95.png')




def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    model = LogisticRegression(solver=GradientDescent()).fit(X_train.to_numpy(), y_train.to_numpy())
    y_prob = model.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y_train.to_numpy(), y_prob)
    go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="black", dash='dash'),
                               name="Random Class Assignment"),
                    go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds, name="", showlegend=False,
                               marker_size=5,
                               marker_color=[0.0],
                               hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
              layout=go.Layout(title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
                               xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
                               yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$"))).write_image\
        (f"C:/Users/itani/BioInformatics/2-B/IML/IML_ex5/IML_ex5_images/con_rate_logistic_regression.png")
    # Q9
    a_star = thresholds[np.argmax(tpr - fpr)]
    #a_star = np.max(tpr - fpr)
    model_q9 = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                  alpha=a_star).fit(X_train.to_numpy(), y_train.to_numpy())
    loss_model_q9 = model_q9.loss(X=X_test.to_numpy(), y=y_test.to_numpy())
    print('Q9 alpha star = ' + str(a_star) + ', and the loss over the test is : ' + str(loss_model_q9)) # str(a_star)
    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    l1_train_err = []
    l1_test_err = []
    l2_train_err = []
    l2_test_err = []
    lambdot = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]
    for lambd in lambdot:
        l1_current_model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                              alpha=0.5, penalty='l1', lam=lambd)
        l1_errors = cross_validate(l1_current_model, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)
        l1_train_err.append(l1_errors[0])
        l1_test_err.append(l1_errors[1])

    for lambd in lambdot:
        l2_current_model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                              alpha=0.5, penalty='l2', lam=lambd)
        l2_errors = cross_validate(l2_current_model, X_train.to_numpy(), y_train.to_numpy(), misclassification_error)
        l2_train_err.append(l2_errors[0])
        l2_test_err.append(l2_errors[1])

    l1_final_model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                        alpha=0.5, penalty='l1', lam=lambdot[np.argmin(l1_test_err)]).fit(
        X_train.to_numpy(), y_train.to_numpy())
    l1_linal_loss = l1_final_model.loss(X_test.to_numpy(), y_test.to_numpy())

    l2_final_model = LogisticRegression(solver=GradientDescent(max_iter=20000, learning_rate=FixedLR(1e-4)),
                                        alpha=0.5, penalty='l2', lam=lambdot[np.argmin(l2_test_err)]).fit(
        X_train.to_numpy(), y_train.to_numpy())
    l2_linal_loss = l2_final_model.loss(X_test.to_numpy(), y_test.to_numpy())

    print('l1 loss: ' + str(l1_linal_loss) + ', lambda = ' + str(lambdot[np.argmin(l1_test_err)]) + ', alpha = 0.5')
    print('l2 loss: ' + str(l2_linal_loss) + ', lambda = ' + str(lambdot[np.argmin(l2_test_err)]) + ', alpha = 0.5')


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    fit_logistic_regression()




import json

import numpy as np
from matplotlib import pyplot as plt


def get_percentiles(loss_history):
    """
    Returns top 25 and low 25 percentiles
    :param loss_history: A list of lists of loss histories
    """
    loss_history_transpose = np.array(loss_history).T

    mean = [np.percentile(x, 50) for x in loss_history_transpose]
    lower = [np.percentile(x, 25) for x in loss_history_transpose]
    upper = [np.percentile(x, 75) for x in loss_history_transpose]
    return mean, lower, upper


def plot_train_test(loss_histories, plot_list, data_stepsize):
    """
    Plots the loss history
    :param loss_history: A list of dict of dict of lists
    """
    hist_dict = loss_histories[0]
    for key in hist_dict["train"]:
        hist_dict["train"][key] = [hist_dict["train"][key]]
    for key in hist_dict["val"]:
        hist_dict["val"][key] = [hist_dict["val"][key]]

    for i, lh in enumerate(loss_histories):
        if i == 0:
            continue
        for key in hist_dict["train"]:
            hist_dict["train"][key].append([lh["train"][key]])
        for key in hist_dict["val"]:
            hist_dict["val"][key].append([lh["val"][key]])

    for plot_i in plot_list:
        if plot_i in hist_dict["train"]:
            mean, lower, upper = get_percentiles(hist_dict["train"][plot_i])
            xaxis = [x * data_stepsize for x in range(len(mean))]
            plt.plot(xaxis, mean, color="b", label=f"{plot_i} (train)")
            plt.fill_between(xaxis, lower, upper, color="b", alpha=0.1)

        if plot_i in hist_dict["val"]:
            mean, lower, upper = get_percentiles(hist_dict["val"][plot_i])
            xaxis = [x * data_stepsize for x in range(len(mean))]
            plt.plot(xaxis, mean, color="r", label=f"{plot_i} (validation)")
            plt.fill_between(xaxis, lower, upper, color="r", alpha=0.1)

        # plt.savefig(f"figure_{plot_i}",dpi=300)
        plt.legend()
        plt.show()
        # plt.close()


# testdict1 = {"train":{"mse":[0,1,2],"test":[2,3,2]},"val":{"mse":[1,2,2],"tes":[1,1,5],"test":[7,2,3]}}
# testdict2 = {"train":{"mse":[15,2,2],"test":[3,3,5]},"val":{"mse":[7,1,4],"tes":[5,6,2],"test":[4,4,3]}}
# test = [testdict1,testdict2, testdict2]
with open("foobar.json") as f:
    test = json.load(f)
plot_train_test([test], ["mse", "so3"], 10)

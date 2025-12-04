import matplotlib.pyplot as plt


def plot_regression(responses,predictions):
    plt.scatter(responses, predictions)
    plt.xlabel("Predictor")
    plt.ylabel("Response")
    plt.title("Regression actual Vs predicted")
    plt.plot([responses.min(), predictions.max()],
            [responses.min(), predictions.max()],
            'r--')  # ideal 1:1 line

    plt.show()

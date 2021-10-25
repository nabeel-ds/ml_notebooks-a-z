{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "Copy of polynomial_regression.ipynb",
      "provenance": [],
      "collapsed_sections": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/nabeel-ds/ml_notebooks-a-z/blob/main/Copy_of_polynomial_regression.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "vN99YjPTDena"
      },
      "source": [
        "# Polynomial Regression"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "ZIx_naXnDyHd"
      },
      "source": [
        "## Importing the libraries"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "FjnmdyPLD2tS"
      },
      "source": [
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6c8YExmOD5x5"
      },
      "source": [
        "## Importing the dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "nQOdXhjXD_AE"
      },
      "source": [
        "dataset = pd.read_csv('Position_Salaries.csv')\n",
        "X = dataset.iloc[:, 1:-1].values\n",
        "y = dataset.iloc[:, -1].values"
      ],
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Le8SEL-YEOLb"
      },
      "source": [
        "## Training the Linear Regression model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "2eZ4xxbKEcBk",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "41074f6d-44c7-4a04-fd49-14bda9fb2885"
      },
      "source": [
        "from sklearn.linear_model import LinearRegression\n",
        "lin_reg = LinearRegression()\n",
        "lin_reg.fit(X, y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 3
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "Rb5nWuSHEfBV"
      },
      "source": [
        "## Training the Polynomial Regression model on the whole dataset"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "HYplp4pTEm0O",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "4c3c03dd-0def-4584-a893-aa2e72629e8f"
      },
      "source": [
        "from sklearn.preprocessing import PolynomialFeatures\n",
        "poly_reg = PolynomialFeatures(degree = 4)\n",
        "X_poly = poly_reg.fit_transform(X)\n",
        "lin_reg_2 = LinearRegression()\n",
        "lin_reg_2.fit(X_poly, y)"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "0O8R0tzbEpvy"
      },
      "source": [
        "## Visualising the Linear Regression results"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "dcTIBAEdEyve",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "c242f259-d9e6-442a-f026-79dffab85972"
      },
      "source": [
        "plt.scatter(X, y, color = 'red')\n",
        "plt.plot(X, lin_reg.predict(X), color = 'blue')\n",
        "plt.title('Truth or Bluff (Linear Regression)')\n",
        "plt.xlabel('Position Level')\n",
        "plt.ylabel('Salary')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "stOnSo74E52m"
      },
      "source": [
        "## Visualising the Polynomial Regression results"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "UCOcurIQE7Zv",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "93927499-de98-4a31-a619-c373926cbe56"
      },
      "source": [
        "plt.scatter(X, y, color = 'red')\n",
        "plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')\n",
        "plt.title('Truth or Bluff (Polynomial Regression)')\n",
        "plt.xlabel('Position level')\n",
        "plt.ylabel('Salary')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "U_qsAMKnE-PJ"
      },
      "source": [
        "## Visualising the Polynomial Regression results (for higher resolution and smoother curve)"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "iE6EnC3fFClE",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 295
        },
        "outputId": "6ecb5687-3c8a-4b46-db4a-c4955c24b9de"
      },
      "source": [
        "X_grid = np.arange(min(X), max(X), 0.1)\n",
        "X_grid = X_grid.reshape((len(X_grid), 1))\n",
        "plt.scatter(X, y, color = 'red')\n",
        "plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')\n",
        "plt.title('Truth or Bluff (Polynomial Regression)')\n",
        "plt.xlabel('Position level')\n",
        "plt.ylabel('Salary')\n",
        "plt.show()"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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4px+z97F5md9jU7z5y5buWtX+PXQdLulWc6cCMzKaDjQv/rqsoN76TRlCnzwQeP3dO2axnKpq0uVtn37Lpvq6lJzzbJl+J89Enpe3apWQ2aVoIgoOoaVMmDAgKivry86DOsAFi5MB/VXXoGXX15WdNNQjNP0QN+9+7IDfcPUp0+a6upSqx1X1lq1kvRERAwot8xXBNahLViQDu4vvggvvZRVzmbTtGnpzL/BJ5hH39Wn03f7tfjv4zb9z5l8XV06q+/Rwwd665icCKzdi0jdJkyalG6umjQpHfhfeCGd8Zce7Hv1SmXz++6bHvvNfOQwV8kAAArYSURBVJR+153DFgsnsiEz0UfAi13htKEumrGa4aIhazciUr85zz6bOkubODG1r584MZXnN1hrrTSS1tZbw1ZbLZu23DKd1TdSV5cK/Jvq0ydlEbMOwkVD1u4sWJAO8s8803h6++1l66y/fmpfP3BgaomzzTaw7bapA7UWF+GsYHhGs1rhRGCFe//91FXCk08u6z7h+efTQOmQWuPssAMccUQaNH2HHdK0wQYVKLPv3bv8FUGZYRvNOionAmtT77+f+sipr0+PTzyRKnMbbLppunP2iCPgk59Mz7fYIse7aIcMadzPD6TMM2RITjs0qz5OBJab+fPT2f1jj6XBUB5/PLXcadC31xx2mfswJ/FPdt5gBjufdzgbfO+otg2yoULYbfithrmy2Cpi6dJ0kH/ssdQl8mOPpTL9hk7SNt009YbZMH3q5dtY/0fHf/xMfKhb65jlYUWVxU4E1iqzZy876I8dm543dKnQvXvqDXP33dPjrrvCJps02YBb65i1KbcaslWydGlqlz92LDz6aHqcODE155RSy52jjkoH/j32SK13mi3Td2sds6rhRFCLmhkN68MP4d//XnbQHzt22dn+uuumg/0xx6T+dHbdFdZZpxUxuLWOWdVwIqg1TUbDiilTmPytixn7SF/Gai/Gjk03bC1dmlbffvs0uPlee6UD/1ZbpYHPV5lb65hVDdcRtKVWjktbSe/33pF/T9uIx9idcezBOPbgXdYHUjfJu++eDvh77pmer7tujsFUwfdhVitcWTx8OONPv4n73tieYzd9mM0uPqXtDzhNx6WF3FvJLFwITz+dmm3++99pahjYHGBbJrInY9mDcezJOLZd/KxHvTLroGo7EWQH4IvnncyZ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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "diyJFZHhFFeK"
      },
      "source": [
        "## Predicting a new result with Linear Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Blmp6Hn7FJW6",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "f01610bc-b077-4df0-cae4-ea37c8b0037f"
      },
      "source": [
        "lin_reg.predict([[6.5]])"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([330378.78787879])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 8
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "DW7I7ZVDFNkk"
      },
      "source": [
        "## Predicting a new result with Polynomial Regression"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "uQmtnyTHFRGG",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 35
        },
        "outputId": "2739bf8a-6dfb-4226-b200-252ee8857097"
      },
      "source": [
        "lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))"
      ],
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "array([158862.45265155])"
            ]
          },
          "metadata": {
            "tags": []
          },
          "execution_count": 9
        }
      ]
    }
  ]
}

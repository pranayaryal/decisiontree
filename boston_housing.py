# Load libraries
import numpy as np
from sklearn import grid_search
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.cross_validation import train_test_split
from sklearn import metrics
from sklearn.metrics import make_scorer


def load_data():
    boston = datasets.load_boston()
    return boston

#exploratory data analysis
def explore_city_data(city_data):
    housing_prices = city_data.target
    housing_features = city_data.data
    num_houses = np.shape(city_data.data)
    num_features = np.shape(city_data.data)
    min_price = np.min(city_data.target)
    max_price = np.max(city_data.target)
    mean_price = np.mean(city_data.target)
    median_price = np.median(city_data.target)
    stand_dev = np.std(city_data.target)

#split data into train and test sets
def split_data(city_data):
   X, y = city_data.data, city_data.target
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
   return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    mse = metrics.mean_squared_error(label, prediction)
    return mse

    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics(scoring tables in sklearn)
    pass

#Calculate the performance of the model after a set of training data.
def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth

    for i, s in enumerate(sizes):

        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()

def find_nearest_neighbour_indexes(x, X): #x is the vector, X is the data set
    from sklearn.neighbors import NearestNeighbors
    neigh = NearestNeighbors(n_neighbors = 10)
    neigh.fit(X)
    distance, indexes = neigh.kneighbors( x )
    return indexes



def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}


    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    scores = make_scorer(metrics.mean_squared_error, greater_is_better=False)
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html


    clf = grid_search.GridSearchCV(regressor, parameters, scoring=scores)
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV

    # Fit the learner to the training data to obtain the best parameter set
    print "Final Model: "
    clf.fit(X,y)

    # Use the model to predict the output of a particular sample
    best_clf = clf.best_estimator_
    print best_clf
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = best_clf.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)
    print "Best model parameter:  " + str(clf.best_params_)

    # Assessing if prediction is reasonable by finding the nearest neighbors using k-nearest neighbor algorithm
    indexes = find_nearest_neighbour_indexes(x, X)
    sum_prices = []
    for i in indexes:
        sum_prices.append(city_data.target[i])
    neighbor_avg = np.mean(sum_prices)
    print "Nearest neighbour average: " + str(neighbor_avg)

def main():


    # Load data
    city_data = load_data()

    # Explore the data
    explore_city_data(city_data)

    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)

    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)


if __name__ == "__main__":
    main()

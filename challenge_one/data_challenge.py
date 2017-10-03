import csv
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


class DataViewer:
    def __init__(self):
        """
        Initialize the DataViewer class variables
        with None values
        """
        self._data = None
        self._classes = None
        self._summary_stats = None
        self._class_corr = None
        self._pca_corr = None
        self._pri_components = None

    def read_dataset(self, fname, delimiter='\t', skip_header=0):
        """
        This method attempts to read in a dataset from a file.
        Assumes that the 'class' column is the last in the file.
        Reads data into numpy array stored in ._data, and
        class gets read into ._classes
        :param fname: File name to read from
        :param delimiter: Type of column delimiter to use
        :param skip_header: Number of rows in file to skip for header
        :return: Nothing
        """
        if not os.path.isfile(fname):
            raise OSError
        try:
            data_set = np.genfromtxt(fname, delimiter=delimiter, skip_header=skip_header)
            self._data = data_set[:, :-1]
            self._classes = data_set[:, -1]
            print "Shape DataSet: {}".format(self._data.shape)
        except:
            print>> sys.stderr, "Unable to parse data file: {}".format(fname)

    def check_data(self):
        """
        Check to make sure ._data is not none. Just
        a simple sanity check
        :return: True if data is set, False otherwise
        """
        if self._data is None:
            print>> sys.stderr, "No Data in Dataset. Use read_dataset first"
            return False
        return True

    def clean_data(self):
        """
        Check the data in ._data for nan values. If nan
        values are found replace with the mean of that column
        Probably should check for neginf and posinf here as well.
        :return: Nothing
        """
        if not self.check_data():
            return False

        # Find locations where ._data is nan
        nan_locations = np.where(np.isnan(self._data))
        for nan_column in nan_locations[1]:
            # Replace nan with mean value of the data in column
            data_col = self._data[:, nan_column]
            data_filled_na = np.where(np.isnan(data_col),
                                      ma.array(data_col,
                                      mask=np.isnan(data_col)).mean(axis=0),
                                      data_col)
            data_col = data_filled_na
            self._data[:, nan_column] = data_col

    def compute_summary_stats(self, statfile="./summary_stats.csv"):
        """
        Compute summary statistics on the variables in the dataset.
        Stats computed are the same as pandas.describe(), e.g.,
        mean, standard deviation, min, max, and quartiles.
        Writes the stats to a csv file.
        :param statfile: csv filename to write stats to as
        :return: Nothing
        """
        if not self.check_data():
            return False
        self._summary_stats = np.zeros((7, self._data.shape[1]))
        with open(statfile, 'w') as statfilehandle:
            statwriter = csv.writer(statfilehandle)
            statwriter.writerow(['Feature', 'Mean', 'Std Dev',
                                 'Min Val', 'Max Val', '25 Percentile',
                                 '50 Percentile', '75 Percentile'])
            try:
                for i in range(self._data.shape[1]):
                    data_col = self._data[:, i]
                    self._summary_stats[0, i] = np.mean(data_col)
                    self._summary_stats[1, i] = np.std(data_col)
                    self._summary_stats[2, i] = np.min(data_col)
                    self._summary_stats[3, i] = np.max(data_col)
                    self._summary_stats[4, i] = np.percentile(data_col, 25)
                    self._summary_stats[5, i] = np.percentile(data_col, 50)
                    self._summary_stats[6, i] = np.percentile(data_col, 75)
                    statwriter.writerow([i+1] + self._summary_stats[:, i].tolist())
            except Exception as e:
                print>> sys.stderr, "Exception Encountered: {}".format(e)

    def compute_pca(self, n_components=2):
        """
        Compute the first n_component principle components
        of the dataset ._data. Store those components in
        ._pri_components.
        :param n_components: Number of principle components to compute
        :return: Nothing
        """
        if not self.check_data():
            return False

        pca = PCA(n_components=n_components)

        self._pri_components = pca.fit_transform(self._data)

    def compute_class_corr(self, corr_stat=stats.pointbiserialr):
        """
        Compute correlation between variable columns in ._data
        and the class column in ._classes.
        Plot the two dimensional Kernal Density Estimator between
        most significant stat and class column
        :param corr_stat: Stat to compute the relationship between
                          data column and class column
        :return:
        """
        self._class_corr = np.zeros((2, self._data.shape[1]))

        for i in range(self._data.shape[1]):
            data_col = self._data[:, i]
            (corr, pval) = corr_stat(data_col, self._classes)

            self._class_corr[0, i] = corr
            self._class_corr[1, i] = pval

        sig_col = np.argmin(self._class_corr[1, :])
        ax3 = sns.jointplot(self._data[:, sig_col], self._classes,
                            kind='kde', stat_func=corr_stat, size=6)
        ax3.set_axis_labels(xlabel='Statistically Sig. Variable',
                            ylabel='Class')
        ax3.fig.suptitle('Kernel Density Estimator for Most Sig. '
                         'Variable and Class')
        ax3.fig.tight_layout()
        plt.show(ax3)

    def compute_pca_corr(self, corr_stat=None):
        """
        Compute correlation between variable columns in ._data
        and the first principle component in ._pri_components.
        Plot the two dimensional Kernal Density Estimator between
        most significant stat and principle component
        :param corr_stat: Stat to compute the relationship between
                          data column and class column
        :return:
        """
        self._pca_corr = np.zeros((2, self._data.shape[1]))

        for i in range(self._data.shape[1]):
            data_col = self._data[:, i]
            (corr, pval) = corr_stat(data_col, self._pri_components[:, 0])

            self._pca_corr[0, i] = corr
            self._pca_corr[1, i] = pval

        sig_col = np.argmin(self._pca_corr[1, :])
        ax4 = sns.jointplot(self._data[:, sig_col], self._pri_components[:, 0],
                            kind='kde', stat_func=corr_stat, size=6)
        ax4.set_axis_labels(xlabel='Statistically Sig. Variable',
                            ylabel='First Principle Component')
        ax4.fig.suptitle('KDE for Most Sig. '
                         'Variable and First Prin. Component')
        ax4.fig.tight_layout()
        plt.show(ax4)

    def plot_heatmap(self, highlight_nan=False):
        """
        Plot a heatmap showing range of values in ._data
        columns across samples.
        highlight_nan shows columns and rows in data where
        nan value are located to search for anomalies in
        data collection
        :param highlight_nan: Whether or not to show columns
                              and rows with nan values. If true
                              does not show actual heatmap.
        :return:
        """
        if not self.check_data():
            return False

        data_to_plot = self._data
        title_txt = 'Heatmap of Data Set'

        if highlight_nan:
            temp_data = np.zeros(self._data.shape)
            nan_locations = np.where(np.isnan(self._data))
            for rowidx in nan_locations[0]:
                temp_data[rowidx, :] = 0.5
            for colidx in nan_locations[1]:
                temp_data[:, colidx] = 0.5

            temp_data[nan_locations] = 1.0

            data_to_plot = temp_data
            title_txt = 'Heatmap Showing NaN Locations'

        p1 = plt.figure()
        ax1 = sns.heatmap(data_to_plot, xticklabels=100, yticklabels=50)
        ax1.set_title(title_txt)
        ax1.set_ylabel('Samples')
        ax1.set_xlabel('Variables')

        plt.show(p1)

    def plot_pca(self, c0_color='b', c1_color='g'):
        """
        Plot a scatter plot of the two principle components
        stored in ._pri_components. Points in scatter plot
        are colored by the corresponding class value.
        :param c0_color: color for the 0 class points
        :param c1_color: color for the 1 class points
        :return:
        """
        if self._pri_components is None:
            print>> sys.stderr, "No Principle Component in Dataset. Use compute_pca first"

        fig2, ax2 = plt.subplots()
        zero_class = np.where(self._classes == 0)
        ax2.scatter(self._pri_components[zero_class, 0], self._pri_components[zero_class, 1],
                    c=c0_color, label='Class 0')
        one_class = np.where(self._classes == 1)
        ax2.scatter(self._pri_components[one_class, 0], self._pri_components[one_class, 1],
                    c=c1_color, label='Class 1')
        ax2.set_title('Scatter Plot of First Two Principle Components')
        ax2.set_xlabel('First Principle Component')
        ax2.set_ylabel('Second Principle Component')
        ax2.legend()
        plt.show(ax2)

    def create_classifier(self, classifier_class=RandomForestClassifier,
                          classifier_params={}, n_features=None, cv=10):
        """
        Create a classifier, and run n-fold cross validatation of the
        data in ._data to predict ._classes.

        :param classifier_class: Specify which classifier to use for model
        :param classifier_params: Parameters for classifier in dictionary form
        :param n_features: Use the top-N most statistically significant variables
                           from ._data. If None, use all variables
        :param cv: How many folds of cross validation to use
        :return: Prints mean of cross validation score
        """
        if not self.check_data():
            return False

        top_features = self._data[:, np.argsort(self._class_corr[1, :])[:n_features]]

        classifier = classifier_class(**classifier_params)
        cv_mean = np.mean(cross_val_score(classifier, top_features,
                                          self._classes, cv=cv))
        print "Mean of {}-fold Cross Validation: {}".format(cv, cv_mean)

    def grid_search(self, classifier_class=RandomForestClassifier,
                    grid_params={}, n_features=None, cv=10):
        """
        Perform grid search to tune classifier hyperparameters.
        :param classifier_class: Specify which classifier to use for model
        :param grid_params: Parameters to tune the classifier in dictionary
                            form. {'param1':[list of vals to try], 'param2':}
        :param n_features: Use the top-N most statistically significant variables
                           from ._data. If None, use all variables
        :param cv: How many folds of cross validation to use
        :return: Prints params and best score for grid search
        """
        if not self.check_data():
            return False

        top_features = self._data[:, np.argsort(self._class_corr[1, :])[:n_features]]

        classifier = classifier_class()
        grid_search = GridSearchCV(classifier, grid_params, cv=cv)
        grid_search.fit(top_features, self._classes)

        print "Num Features: {}".format(top_features.shape[1])
        print "Best Params"
        print grid_search.best_params_
        print "Best {}-fold Cross-Validation Score".format(cv)
        print grid_search.best_score_


if __name__ == "__main__":
    import sys

    argv = sys.argv[1:]
    fname = "./dataset_challenge_one.tsv"
    if len(argv) == 1:
        fname = argv[0]

    dataviewer = DataViewer()
    try:
        dataviewer.read_dataset(fname, delimiter='\t', skip_header=1)
    except OSError:
        print>> sys.stderr, "File Does Not Exist: {}".format(fname)

    dataviewer.plot_heatmap(highlight_nan=True)
    dataviewer.clean_data()
    dataviewer.compute_summary_stats()
    dataviewer.plot_heatmap()

    dataviewer.compute_pca()
    dataviewer.plot_pca(c0_color='r', c1_color='b')

    dataviewer.compute_class_corr(
        corr_stat=stats.pointbiserialr)

    dataviewer.compute_pca_corr(
        corr_stat=stats.pearsonr)

    # Tuned a RandomForest classifier over multiple
    # values of n_features
    # grid_params = {'n_estimators': [5, 10, 15, 20],
    #               'max_depth': [2, 3, 4, 5, 6],
    #              'max_features': [0.3, 0.4, 0.5, 0.6]}

    # for num_features in [None] + range(5, 20):
    #    dataviewer.grid_search(grid_params=grid_params,
    #                           n_features=num_features)

    # Through grid search determined following optimal parameters
    # RandomForest with n_estimators: 15, max_depth: 2, max_features: 0.5,
    # or half the number of total features. Using top 16 features from ._data
    # and 10-fold cross validation.
    classifier_params = {'n_estimators': 15,
                         'max_depth': 2,
                         'max_features': 0.5}

    dataviewer.create_classifier(classifier_class=RandomForestClassifier,
                                 classifier_params=classifier_params,
                                 n_features=16, cv=10)

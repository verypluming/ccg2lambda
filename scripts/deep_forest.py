#
# Inspired by https://arxiv.org/abs/1702.08835 and https://github.com/STO-OTZ/my_gcForest/
#
import logging
import numpy as np
import uuid

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict


def create_logger(instance, verbose):
    logger = logging.getLogger(str(uuid.uuid4()))
    fmt = logging.Formatter('{} - %(message)s'.format(instance))
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    return logger


#
# Taken from: http://stackoverflow.com/questions/4936620/using-strides-for-an-efficient-moving-average-filter
#
def rolling_window_lastaxis(a, window):
    """
    Directly taken from Erik Rigtorp's post to numpy-discussion.
    <http://www.mail-archive.com/numpy-discussion@scipy.org/msg29450.html>
    """
    if window < 1:
        raise ValueError("`window` must be at least 1.")
    if window > a.shape[-1]:
        raise ValueError("`window` is too long.")
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window(a, window):
    if not hasattr(window, '__iter__'):
        return rolling_window_lastaxis(a, window)
    for i, win in enumerate(window):
        if win > 1:
            a = a.swapaxes(i, -1)
            a = rolling_window_lastaxis(a, win)
            a = a.swapaxes(-2, i)

    return a


class MGCForest():
    """
    Multi-Grained Cascade Forest

    @param estimators_config    A dictionary containing the configurations for the estimators of
                                the estimators of the MultiGrainedScanners and the CascadeForest.
    @param stride_ratios        A list of stride ratios for each MultiGrainedScanner instance.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.

    Example:

    estimators_config={
        'mgs': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 30,
                'min_samples_split': 21,
                'n_jobs': -1,
            }
        }],
        'cascade': [{
            'estimator_class': ExtraTreesClassifier,
            'estimator_params': {
                'n_estimators': 1000,
                'min_samples_split': 11,
                'max_features': 1,
                'n_jobs': -1,
            }
        }]
    },
    """
    def __init__(
        self,
        estimators_config,
        stride_ratios=[1.0 / 4, 1.0 / 9, 1.0 / 16],
        folds=3,
        verbose=False
    ):
        self.mgs_instances = [
            MultiGrainedScanner(
                estimators_config['mgs'],
                stride_ratio=stride_ratio,
                folds=folds,
                verbose=verbose,
            )
            for stride_ratio in stride_ratios
        ]
        self.stride_ratios = stride_ratios

        self.c_forest = CascadeForest(estimators_config['cascade'], verbose=verbose)

    def fit(self, X, y):
        scanned_X = np.hstack([
            mgs.fit(X, y)
            for mgs in self.mgs_instances
        ])

        self.c_forest.fit(scanned_X, y)

    def predict(self, X):
        scan_pred = np.hstack([
            mgs.predict(X)
            for mgs in self.mgs_instances
        ])

        return self.c_forest.predict(scan_pred)

    def __repr__(self):
        return '<MGCForest {}>'.format(self.stride_ratios)


class MultiGrainedScanner():
    """
    Multi-Grained Scanner

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the MultiGrainedScanner.
    @param stride_ratio         The stride ratio to use for slicing the input.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """
    def __init__(
        self, estimators_config, stride_ratio=0.25, folds=3, verbose=False
    ):
        self.estimators_config = estimators_config
        self.stride_ratio = stride_ratio
        self.folds = folds

        self.windows_estimators = []

        self.logger = create_logger(self, verbose)

    def slices(self, X):
        """
        Given an input X with dimention N, return a ndarray of dimention 3 with all the instances
        values for each window.

        For example, if the input has shape (10, 400), and the stride_ratio is 0.25, then this
        will generate 301 windows with shape (10, 100). The final result would have a shape of
        (301, 10, 100).
        """
        self.logger.debug('Slicing X with shape {}'.format(X.shape))
        sample_shape = list(X[0].shape)

        window_shape = np.maximum(
            np.array([s * self.stride_ratio for s in sample_shape]), 1
        ).astype(np.int16)
        self.logger.debug('Got window shape: {}'.format(window_shape.shape))

        #
        # Calculate the windows that are going to be used and the total
        # number of new generated samples.
        #
        windows_count = [sample_shape[i] - window_shape[i] + 1 for i in range(len(sample_shape))]
        new_instances_total = np.prod(windows_count)

        self.logger.debug('Slicing {} windows.'.format(windows_count))

        #
        # For each sample, get all the windows with their values
        #
        sliced_X = np.array([
            rolling_window(x, window_shape)
            for x in X
        ])

        #
        # Swap the 0 and 1 axis so as to get for each window, the value of each sample.
        #
        sliced_X = np.swapaxes(sliced_X, 0, 1)

        if len(sliced_X.shape) > 3:
            shape = list(sliced_X.shape)
            sliced_X = sliced_X.reshape(shape[:2] + [np.prod(shape[2:])])

        self.logger.info(
            'Scanning turned X ({}) into sliced_X ({}). {} new instances were added '
            'per sample'.format(X.shape, sliced_X.shape, new_instances_total)
        )

        return sliced_X

    def fit(self, X, y):
        """
        Slice the input and for each window creates the estimators and save the estimators in
        self.window_estimators. Then for each window, fit the estimators with the data of all
        the samples values on that window and perform a cross_val_predict and get the predictions.
        """
        self.logger.info('Scanning and fitting for X ({}) and y ({}) started'.format(
            X.shape, y.shape
        ))
        self.n_classes = np.unique(y).size
        sliced_X = self.slices(X)

        #
        # Create an estimator for each generated window
        #
        self.windows_estimators = []
        predictions = []
        for window_index, window_X in enumerate(sliced_X):
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]
            self.windows_estimators.append(estimators)

            self.logger.debug(
                'Window #{}:: Training estimators for window with shape {}'.format(
                    window_index, window_X.shape
                )
            )

            for estimator_index, estimator in enumerate(estimators):
                self.logger.debug(
                    'Window #{}:: Fitting estimator #{} ({})'
                    .format(window_index, estimator_index, estimator.__class__)
                )
                estimator.fit(window_X, y)

                #
                # Gets a prediction of sliced_X with shape (len(newX), n_classes).
                # The method `predict_proba` returns a vector of size n_classes.
                #
                self.logger.debug('Window #{}:: Cross-validation with estimator #{} ({})'.format(
                    window_index, estimator_index, estimator.__class__
                ))
                prediction = cross_val_predict(
                    estimator,
                    window_X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

        self.logger.info('Finished fitting X ({}) and got predictions with shape {}'.format(
            X.shape, np.array(predictions).shape
        ))
        return np.hstack(predictions)

    def predict(self, X):
        self.logger.info('Predicting X ({})'.format(X.shape))
        sliced_X = self.slices(X)
        return np.hstack([
            estimator
            .predict_proba(window_X)
            for window_X, window_estimators in zip(sliced_X, self.windows_estimators)
            for estimator in window_estimators
        ])

    def __repr__(self):
        return '<MultiGrainedScanner stride_ratio={}>'.format(self.stride_ratio)


class CascadeForest():
    """
    CascadeForest

    @param estimators_config    A list containing the class and parameters of the estimators for
                                the CascadeForest.
    @param folds                The number of k-folds to use.
    @param verbose              Adds verbosity.
    """
    def __init__(self, estimators_config, folds=3, verbose=False):
        self.estimators_config = estimators_config
        self.folds = folds

        self.logger = create_logger(self, verbose)

    def fit(self, X, y):
        self.logger.info('Cascade fitting for X ({}) and y ({}) started'.format(X.shape, y.shape))
        self.classes = np.unique(y)
        self.level = 0
        self.levels = []
        self.max_score = None

        while True:
            self.logger.info('Level #{}:: X with shape: {}'.format(self.level + 1, X.shape))
            estimators = [
                estimator_config['estimator_class'](**estimator_config['estimator_params'])
                for estimator_config in self.estimators_config
            ]

            predictions = []
            for estimator in estimators:
                self.logger.debug('Fitting X ({}) and y ({}) with estimator {}'.format(
                    X.shape, y.shape, estimator
                ))
                estimator.fit(X, y)

                #
                # Gets a prediction of X with shape (len(X), n_classes)
                #
                prediction = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=self.folds,
                    method='predict_proba',
                    n_jobs=-1,
                )

                predictions.append(prediction)

            self.logger.info('Level {}:: got all predictions'.format(self.level + 1))

            #
            # Stacks horizontally the predictions to each of the samples in X
            #
            X = np.hstack([X] + predictions)

            #
            # For each sample, compute the average of predictions of all the estimators, and take
            # the class with maximum score for each of them.
            #
            y_prediction = self.classes.take(
                np.array(predictions).mean(axis=0).argmax(axis=1)
            )

            score = accuracy_score(y, y_prediction)
            self.logger.info('Level {}:: got accuracy {}'.format(self.level + 1, score))
            if self.max_score is None or score > self.max_score:
                self.level += 1
                self.max_score = score
                self.levels.append(estimators)
            else:
                break

    def predict(self, X):
        for estimators in self.levels:

            predictions = [
                estimator.predict_proba(X)
                for estimator in estimators
            ]
            self.logger.info('Shape of predictions: {} shape of X: {}'.format(
                np.array(predictions).shape, X.shape
            ))
            X = np.hstack([X] + predictions)

        return self.classes.take(
            np.array(predictions).mean(axis=0).argmax(axis=1)
        )

    def __repr__(self):
        return '<CascadeForest forests={}>'.format(len(self.estimators_config))

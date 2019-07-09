import numpy as np

from .partition import LabelSpacePartitioningClassifier
from skmultilearn.adapt import MLkNN
from .base import MLClassifierBase
from .random import RandomLabelSpaceClusterer

class RakelMLkNN(MLClassifierBase):
    """Distinct RAndom k-labELsets multi-label classifier.

    Divides the label space in to equal partitions of size k, trains a Label Powerset
    classifier per partition and predicts by summing the result of all trained classifiers.

    With this implementation we will try a greedy approach for the partition of the labelset.

    Parameters
    ----------
    labelset_size : int
        the desired size of each of the partitions, parameter k according to paper
        Default is 3, according to paper it has the best results


    Attributes
    ----------
    _label_count : int
        the number of labels the classifier is fit to, set by :meth:`fit`

    model_count_ : int
        the number of sub classifiers trained, set by :meth:`fit`

    classifier_: :class:`skmultilearn.ensemble.LabelSpacePartitioningClassifier`
        the underneath classifier that perform the label space partitioning using a
        random clusterer :class:`skmultilearn.ensemble.RandomLabelSpaceClusterer`


    References
    ----------

    If you use this class please cite the paper introducing the method:

    .. code :: latex

        @ARTICLE{5567103,
            author={G. Tsoumakas and I. Katakis and I. Vlahavas},
            journal={IEEE Transactions on Knowledge and Data Engineering},
            title={Random k-Labelsets for Multilabel Classification},
            year={2011},
            volume={23},
            number={7},
            pages={1079-1089},
            doi={10.1109/TKDE.2010.164},
            ISSN={1041-4347},
            month={July},
        }

    Examples
    --------

    Here's a simple example of how to use this class with a base classifier from scikit-learn to teach
    non-overlapping classifiers each trained on at most four labels:

    .. code :: python

        from sklearn.naive_bayes import GaussianNB
        from skmultilearn.ensemble import RakelMLkNN

        classifier = RakelMLkNN(
            labelset_size=4
        )

        classifier.fit(X_train, y_train)
        prediction = classifier.predict(X_test)

    """

    def __init__(self, labelset_size=3):
        super(RakelMLkNN, self).__init__()

        self.labelset_size = labelset_size
        self.copyable_attrs = ['labelset_size']

    def fit(self, X, y):
        """Fit classifier to multi-label data

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse
            input features, can be a dense or sparse matrix of size
            :code:`(n_samples, n_features)`
        y : numpy.ndaarray or scipy.sparse {0,1}
            binary indicator matrix with label assignments, shape
            :code:`(n_samples, n_labels)`

        Returns
        -------
        fitted instance of self
        """
        self._label_count = y.shape[1]
        self.model_count_ = int(np.ceil(self._label_count / self.labelset_size))
        self.classifier_ = LabelSpacePartitioningClassifier(
            classifier=MLkNN(),
            clusterer=RandomLabelSpaceClusterer(
                cluster_size=self.labelset_size,
                cluster_count=self.model_count_,
                allow_overlap=False
            ),
            require_dense=[False, False]
        )
        return self.classifier_.fit(X, y)

    def predict(self, X):
        """Predict label assignments

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of int
            binary indicator matrix with label assignments with shape
            :code:`(n_samples, n_labels)`
        """

        return self.classifier_.predict(X)

    def predict_proba(self, X):
        """Predict label probabilities

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csc_matrix
            input features of shape :code:`(n_samples, n_features)`

        Returns
        -------
        scipy.sparse of float
            binary indicator matrix with probability of label assignment with shape
            :code:`(n_samples, n_labels)`
        """

        return self.classifier_.predict_proba(X)

import unittest

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from Rakelmlknn.rakelmlknn import RakelMLkNN
from .classifier_basetest import ClassifierBaseTest

TEST_LABELSET_SIZE = 3


class RakelMLkNNTest(ClassifierBaseTest):

    def get_rakelmlknn(self):
        return RakelMLkNN(
            labelset_size=TEST_LABELSET_SIZE
        )

    def test_if_sparse_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_rakelmlknn
        self.assertClassifierWorksWithSparsity(classifier, 'sparse')
        self.assertClassifierPredictsProbabilities(classifier, 'sparse')

    def test_if_dense_classification_works_on_non_dense_base_classifier(self):
        classifier = self.get_rakelmlknn
        self.assertClassifierWorksWithSparsity(classifier, 'dense')
        self.assertClassifierPredictsProbabilities(classifier, 'dense')

    def test_if_works_with_cross_validation(self):
        classifier = self.get_rakelmlknn()
        self.assertClassifierWorksWithCV(classifier)


if __name__ == '__main__':
    unittest.main()

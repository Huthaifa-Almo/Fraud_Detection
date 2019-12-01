import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

if not sys.warnoptions:
    import warnings

    warnings.simplefilter("ignore")


class GaussianNormalDistribution:
    # prepare and split the dataset
    def prepare_split_data(self, dataset):
        df = pd.read_csv(dataset)

        hist = df.hist(bins=100, figsize=(20, 20))

        df_distributed = df[["V11", "V13", "V15", "V18", "V19", "V12", "Class"]]

        negative_df = df_distributed.loc[df_distributed['Class'] == 0]
        positive_df = df_distributed.loc[df_distributed['Class'] == 1]
        # the data are very sekwed as we have almost 450 anomalies samples and more than 200000 good results so we will
        # divide the data as following
        y_negative = negative_df["Class"]
        y_positive = positive_df["Class"]
        negative_df.drop(["Class"], axis=1, inplace=True)
        positive_df.drop(["Class"], axis=1, inplace=True)

        # 90% of good data are training data to estimate Gaussian factors (mean, Standard deviation and variance)
        negative_df_training, negative_df_testing, y_negative_training, y_negative_testing = train_test_split(
            negative_df,
            y_negative,
            test_size=0.1,
            random_state=0)

        # 5% for CV dataset, 5% for testing dataset
        negative_df_cv, negative_df_testing, y_negative_cv, y_negative_testing = train_test_split(negative_df_testing,
                                                                                                  y_negative_testing,
                                                                                                  test_size=0.5,
                                                                                                  random_state=0)
        # while 50% the anomalies data will be added to CV dataset and the other 50% will be added to Testing dataset
        positive_df_cv, positive_df_testing, y_positive_cv, y_positive_testing = train_test_split(positive_df,
                                                                                                  y_positive,
                                                                                                  test_size=0.5,
                                                                                                  random_state=0)

        df_cv = pd.concat([positive_df_cv, negative_df_cv], ignore_index=True)
        df_cv_y = pd.concat([y_positive_cv, y_negative_cv], ignore_index=True)
        df_test = pd.concat([positive_df_testing, negative_df_testing], ignore_index=True)
        df_test_y = pd.concat([y_positive_testing, y_negative_testing], ignore_index=True)

        y_negative_training = y_negative_training.values.reshape(y_negative_training.shape[0], 1)
        df_cv_y = df_cv_y.values.reshape(df_cv_y.shape[0], 1)
        df_test_y = df_test_y.values.reshape(df_test_y.shape[0], 1)

        return negative_df_training, df_cv, df_cv_y, df_test, df_test_y

    # estimate Gaussian factors (mean, Standard deviation and variance)
    def fit(self, X):
        stds = []
        mean = []
        variance = []

        mean = X.mean(axis=0)
        stds = X.std(axis=0)
        variance = stds ** 2

        stds = stds.values.reshape(stds.shape[0], 1)
        mean = mean.values.reshape(mean.shape[0], 1)
        variance = variance.values.reshape(variance.shape[0], 1)
        return stds, mean, variance

    # Calculate the PROBABILITY for any new data CV or Testing using the factor that we have calculated and using the
    # GAUSSIAN NORMAL DISTRIBUTION algorithm
    def predict(self, std_, mean_, variance_, positive_df):
        probability = []
        for i in range(positive_df.shape[0]):
            result = 1
            for j in range(positive_df.shape[1]):
                var1 = 1 / (np.sqrt(2 * np.pi) * std_[j])
                var2 = (positive_df.iloc[i, j] - mean_[j]) ** 2
                var3 = 2 * variance_[j]

                result *= (var1) * np.exp(-(var2 / var3))
            result = float(result)
            probability.append(result)
        return probability

    # select the best EPSILON by calculation the F1, PRECISION and RECALL and select the beat epsilon for each using
    # CV DATASET
    def selectEpsilon(self, y_actual, y_probability):
        best_epi = 0
        best_F1 = 0
        best_rec = 0
        best_pre = 0

        stepsize = (max(y_probability) - min(y_probability)) / 100000
        epi_range = np.arange(min(y_probability), max(y_probability), stepsize)
        for epi in epi_range:
            predictions = (y_probability < epi)[:, np.newaxis]
            tp = np.sum(predictions[y_actual == 1] == 1)
            fp = np.sum(predictions[y_actual == 0] == 1)
            fn = np.sum(predictions[y_actual == 1] == 0)

            prec = tp / (tp + fp)
            rec = tp / (tp + fn)

            if prec > best_pre:
                best_pre = prec
                best_epi_prec = epi

            if rec > best_rec:
                best_rec = rec
                best_epi_rec = epi

            F1 = (2 * prec * rec) / (prec + rec)

            if F1 > best_F1:
                best_F1 = F1
                best_epi = epi

        return best_epi, best_F1, best_pre, best_epi_prec, best_rec, best_epi_rec

    # F1, Percision and Recall for testing data
    def prediction_scores(self, y_actual, y_probability, epsilon):
        predictions = (y_probability < epsilon)[:, np.newaxis]
        tp = np.sum(predictions[y_actual == 1] == 1)
        fp = np.sum(predictions[y_actual == 0] == 1)
        fn = np.sum(predictions[y_actual == 1] == 0)
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        F1 = (2 * prec * rec) / (prec + rec)
        return prec, rec, F1


# select the file location
dataset_loc = "creditcard.csv"
# create the model
model = GaussianNormalDistribution()
# prepare the dataset's and split em into train, cv and test
negative_df_training, df_cv, df_cv_y, df_test, df_test_y = model.prepare_split_data(dataset_loc)
# fit the model with the training dataset and return the results (mean, Standard deviation and variance)
stds, mean, variance = model.fit(negative_df_training)
# use the cv dataset to make the predictions that will be used to select the best threshold
probability = model.predict(stds, mean, variance, df_cv)
# use the probability that we had to be used with the epsilon selection process
best_epi, best_F1, best_pre, best_epi_prec, best_rec, best_epi_rec = model.selectEpsilon(df_cv_y, probability)
print("The best epsilon Threshold over the cross validation set is :", best_epi)
print("The best F1 score over the cross validation set is :", best_F1)
print("The best epsilon Threshold over the cross validation set is for recall :", best_epi_rec)
print("The best Recall score over the cross validation set is :", best_rec)
print("The best epsilon Threshold over the cross validation set is for precision:", best_epi_prec)
print("The best Precision score over the cross validation set is :", best_pre)

# RUN the code and print out the Results for the test set
epsilon = best_epi
probability = model.predict(stds, mean, variance, df_test)
prec, rec, F1 = model.prediction_scores(df_test_y, probability, epsilon)
print("Precision on Testing Set:", prec)
print("Recall on Testing Set:", rec)
print("F1 on Testing Set:", F1)

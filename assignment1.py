import string
import pandas as pd
import numpy as np
import math
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

# note: since the original file is an xlsx file
# i created a csv file off the original xlsx file before i ran my functions
# NB: the to_csv function can only be called once
# df = pd.read_excel('movie_reviews.xlsx')
# df.to_csv('movie_reviews2.csv')
reviews = pd.read_csv('movie_reviews.csv', delimiter=',')


# this function will extract the data(reviews) and the label(positive/negative) from the training and test set
def question1(data):
    train = data[data['Split'] == 'train']
    test = data[data['Split'] == 'test']

    train_data = train['Review']
    train_label = train['Sentiment']
    test_data = test['Review']
    test_label = test['Sentiment']

    no_negative_reviews_train = len(train[train['Sentiment'] == 'negative'])
    no_positive_reviews_train = len(train[train['Sentiment'] == 'positive'])
    no_negative_reviews_test = len(test[test['Sentiment'] == 'negative'])
    no_positive_reviews_test = len(test[test['Sentiment'] == 'positive'])
    print("Number of negative reviews in Test Data: "+str(no_negative_reviews_test))
    print("Number of positive reviews in Test Data: "+str(no_positive_reviews_test))
    print("Number of negative reviews in Train Data: "+str(no_negative_reviews_train))
    print("Number of positive reviews in Train Data:"+str(no_positive_reviews_train))

    return train_data, train_label, test_data, test_label, no_negative_reviews_train, no_positive_reviews_train


training_data, training_label, testing_data, testing_label, \
    no_negative_reviews, no_positive_reviews = question1(reviews)


# this will return a list of words that are longer than the minWordLength and occurs more frequently than the minWordOccurrence
def question2(data_input, minWordLength, minWordOccurrence):
    data_input = data_input.apply(lambda x: ''.join([i for i in x if i not in string.punctuation]))
    data_input = data_input.apply(lambda x: x.lower())
    num_counts = data_input.str.split(expand=True).stack().value_counts()
    dictionaries = dict((k, v) for k, v in num_counts.items() if len(k) >= minWordLength and v >= minWordOccurrence)
    return list(dictionaries.keys())


# this function will count the number of reviews that contain this word
def question3(set_of_reviews, test_words):
    pos_dict = {}

    for word in test_words:
        pos_dict[word] = 0
    for i in set_of_reviews:
        for k in test_words:
            if k in i:
                pos_dict[k] += 1
    return pos_dict


# this is the function for question 4 which calculates the likelihoods of each words and the priors
def question4(pos_freq_count, neg_freq_count, no_pos_reviews, no_neg_reviews):
    prob_pos = no_pos_reviews/(no_pos_reviews+no_neg_reviews)
    prob_neg = no_neg_reviews/(no_pos_reviews+no_neg_reviews)
    alpha = 1
    length_of_positive_frequency = len(pos_freq_count)
    length_of_negative_frequency = len(neg_freq_count)
    if any(value == 0 for value in pos_freq_count.values()):
        for value in pos_freq_count:
            pos_freq_count[value] += alpha

    if any(value == 0 for value in neg_freq_count.values()):
        for value in neg_freq_count:
            neg_freq_count[value] += alpha
    no_features_neg = sum(neg_freq_count.values())
    no_features_pos = sum(pos_freq_count.values())
    for word in pos_freq_count:
        pos_freq_count[word] = pos_freq_count[word] / (no_features_pos + (length_of_positive_frequency * alpha))

    for word in neg_freq_count:
        neg_freq_count[word] = neg_freq_count[word] / (no_features_neg + (length_of_negative_frequency * alpha))

    return pos_freq_count, neg_freq_count, prob_pos, prob_neg


# this function predicts the label according to the likelihoods and priors
def question5(text, pos_like, neg_like, pos_p, pos_n):
    listed = []
    for word in text.split():
        listed.append(word)

    count = 0
    for word in listed:
        if word in pos_like.keys():
            count += math.log(pos_like[word])

    count_negative = 0
    for word in listed:
        if word in neg_like.keys():
            count_negative += math.log(neg_like[word])

    if count - count_negative > math.log(pos_n)-math.log(pos_p):
        result = "positive"
    else:
        result = "negative"

    return result


def question6():
    number_of_accuracy = []
    kf = KFold(n_splits=5, shuffle=False, random_state=None)
    for i in range(1, 10+1):
        print("Testing for word length of " + str(i))
        print("=====================================")
        arrays = []
        kfold_accuracies = []
        for k, kl in kf.split(training_data, training_label):
            kfold_arrays = []
            pos_review_dict = question3(training_data.iloc[k][training_label.iloc[k] == 'positive'],
                                        question2(training_data.iloc[k], i, 300))
            neg_review_dict = question3(training_data.iloc[k][training_label.iloc[k] == 'negative'],
                                        question2(training_data.iloc[k], i, 300))
            positive_likelihood, negative_likelihood, prior_pos, prior_neg = question4(pos_review_dict, neg_review_dict,
                                                                                       no_positive_reviews,
                                                                                       no_negative_reviews)
            for pika in training_data.iloc[kl]:
                kfold_arrays.append(question5(pika, positive_likelihood, negative_likelihood, prior_pos, prior_neg))
            kfold_accuracies.append(accuracy_score(training_label.iloc[kl].tolist(), kfold_arrays))
            print("Accuracy score KFold: "+str(accuracy_score(kfold_arrays, training_label.iloc[kl].tolist())))
        print("\nMean score of KFold: "+str(round(np.mean(kfold_accuracies), 2)))
        for tests in testing_data:
            arrays.append(question5(tests, positive_likelihood, negative_likelihood, prior_pos, prior_neg))
        c = confusion_matrix(testing_label.to_list(), arrays)
        print(c)
        print("True Positives: " + str(round((c[0, 0] / len(testing_data))*100, 2)) + "%")
        print("True Negatives: " + str(round((c[1, 1] / len(testing_data))*100, 2)) + "%")
        print("False Positives: " + str(round((c[0, 1] / len(testing_data))*100, 2)) + "%")
        print("False Negatives: " + str(round((c[1, 0] / len(testing_data))*100, 2)) + "%")
        accuracy = accuracy_score(testing_label.to_list(), arrays)
        print("Accuracy score on test data: "+str(round((accuracy * 100), 2))+"%")
        number_of_accuracy.append(accuracy)
    print("Optimal Length "+str(number_of_accuracy.index(max(number_of_accuracy))+1))


def question7():
    good_review = "I have watched so many movies but never before have i been so wowed by such a spectacular movie" \
                  "the acting was superb, the action scenes were wonderful and the overall story was so intriguing" \
                  "i will definitely recommend this masterpiece"
    bad_review = "Unfortunately, this movie fell short in a lot of areas, the acting was bland, the whole story" \
                 "did not make any sense. it was just a big catastrophic excuse of a movie. " \
                 "i would not watch this movie ever again"
    pos_review_dict = question3(training_data[training_label == 'positive'],
                                question2(training_data, 4, 300))
    neg_review_dict = question3(training_data[training_label == 'negative'],
                                question2(training_data, 4, 300))
    positive_likelihood, negative_likelihood, prior_pos, prior_neg = question4(pos_review_dict, neg_review_dict,
                                                                               no_positive_reviews,
                                                                               no_negative_reviews)
    # should print positive
    print(question5(good_review, positive_likelihood, negative_likelihood, prior_pos, prior_neg))
    # should print negative
    print(question5(bad_review, positive_likelihood, negative_likelihood, prior_pos, prior_neg))


question6()
# question7()

# for x, y in zip(arrays, testing_label.to_list()):
#    if x == y:
#        if x == 'positive' and y == 'positive':
#            tp += 1
#        elif x == 'negative' and y == 'negative':
#            tn += 1
#    if x != y:
#        if x == 'positive' and y == 'negative':
#            fp += 1
#        elif x == 'negative' and y == 'positive':
#            fn += 1

# print("[["+ str(tp) + " " + str(fp) + "]")
# print(" ["+ str(fn) + " " + str(tn) + "]]")
# accuracy = round(((tp + tn)/len(testing_data))*100, 2)
# print("True Positives: " + str(round((tp / len(testing_data))*100, 2)) + "%")
# print("True Negatives: " + str(round((tn / len(testing_data))*100, 2)) + "%")
# print("False Positives: " + str(round((fp / len(testing_data))*100, 2)) + "%")
# print("False Negatives: " + str(round((fn / len(testing_data))*100, 2)) + "%")
# print(str(accuracy) + "%")

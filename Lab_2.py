import numpy as np
import string
from collections import Counter


np.random.seed(1)

# Problem 1

def count_frequency(documents):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    lower_case_doc = []
    for s in documents:
        lower_case_doc.append(s.lower())

    no_punc_doc = []
    for s in lower_case_doc:
        no_punc_doc.append(s.translate(s.maketrans('', '', string.punctuation)))
            
    words_doc = []
    words_doc = [s.split() for s in no_punc_doc]

    frequency = Counter()
    
    for el in words_doc:
        for e in el:
            frequency[e] += 1
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return frequency

# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
def prior_prob(y_train):

    prior = {}
    for elem in y_train:
        if elem not in prior:
            prior[elem] = 1
        else:
            prior[elem] += 1
    
    for cl in prior:
        prior[cl] /= len(y_train)
    
    
    return prior


def conditional_prob(X_train, y_train):

    cond_prob = {}
    
    for elem in y_train:
        if elem not in cond_prob:
            cond_prob[elem] = []
    
    
    for i in range(len(X_train)):        
         cond_prob[y_train[i]].append(X_train[i])
    
    for cl in cond_prob:
        cond_prob[cl] = count_frequency(cond_prob[cl])
        tot_words = 0
        for word in cond_prob[cl]:
            tot_words += cond_prob[cl][word]
        for word in cond_prob[cl]:
            cond_prob[cl][word] = (cond_prob[cl][word] + 1) / (tot_words + 20000)
        

    return cond_prob

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


def predict_label(X_test, prior_prob, cond_prob):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    predict = []
    test_prob = []
    
    
    for elem in X_test:
        freq = count_frequency([elem])
        crt_prob = [0] * len(cond_prob)
        
        for cls in cond_prob:
            crt_prob[cls] = compute_test_prob(freq, prior_prob[cls], cond_prob[cls])

        m = max(crt_prob)
        denom = sum([np.exp(x - m) for x in crt_prob])
        crt_prob = [np.exp(x-m) / denom for x in crt_prob]
        
        test_prob.append(crt_prob)
        predict.append(np.argmax(crt_prob))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return predict, test_prob

def compute_test_prob(word_count, prior_cat, cond_cat):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    prob = np.log(prior_cat)
    for word in word_count:
        if word not in cond_cat:
            prob += np.log(1 / 20000)
        else:
            prob += np.log(cond_cat[word])
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return prob

def compute_metrics(y_pred, y_true):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    return acc, cm, f1


# Problem 2

def featureNormalization(X):
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_normalized = (X - X_mean) / X_std
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized, X_mean, X_std


def applyNormalization(X, X_mean, X_std):

    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    X_normalized = (X - X_mean) / X_std
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return X_normalized

def computeMSE(X, y, theta):
    
	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	error = [[0]]
	error[0] = np.sum(np.power(np.dot(X, np.squeeze(theta, 1).transpose()) - y, 2) * 1/(2 * X.shape[0]))
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	return error[0]

def computeGradient(X, y, theta):

	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	gradient = np.expand_dims(np.matmul(X.transpose(), np.dot(X, np.squeeze(theta, 1).transpose()) - y) * 1/X.shape[0], axis=1)
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

	return gradient

def gradientDescent(X, y, theta, alpha, num_iters):

	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	Loss_record = np.zeros((num_iters))
	m = len(y)
	for i in range(num_iters):
		theta = theta - alpha * computeGradient(X, y, theta)
		Loss_record[i] = computeMSE(X, y, theta)
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	return theta, Loss_record

def closeForm(X, y):

	# *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(X.transpose(), X)), X.transpose()), y.transpose())
  
	# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
	return theta

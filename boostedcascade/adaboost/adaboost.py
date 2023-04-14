
# Author: Donny

import numpy as np
import copy

from .decisionstump import DecisionStumpClassifier

class AdaBoostClassifier:
    """An AdaBoost classifier.

    Parameters
    ----------
    n_classes_ : int
        count of maximal weak classifiers (일부러 classifier의 개수를 제한)
                                          (다만, 특정 FPR에 도달하기를 원하는 게 우선)

    weak_classifier_ : A weak classifier class or a factory which
                       can return this kind of class.

        A weak classifier class, with:
            function train(self, X, y, W), where:
                    param X : [array] accepts inputs of the training samples.           # 전체 샘플이 들어옴 [n_samples, n_features]
                    param y : [array] accepts class labels of the training samples.     # 전체 샘플의 정답지가 들어옴 [n_samples, 2] (2는 정답지가 0 또는 1이기 때문에)
                    param W : [array] accepts weights of each samples                   # 각 샘플의 가중치
                    return  : [float] the sum of weighted errors                        # 가중화된 에러의 합을 리턴

            function predict(self, test_set), where:
                    param test_set : accepts test samples                               # 전체 테스트 샘풀이 들어옴 [m_samples, n_features]
                    return         : classify result                                    # weak classifier에서 예측한 답안을 리턴 [m_samples, 2] (2는 예측이 0 또는 1이기 때문에)
    """
    def __init__(self,
                 weak_classifier_ = DecisionStumpClassifier()):

        # Maximal weak classifiers (미리 weak classifier의 개수를 제한하려고)
        # self.mxWC = n_classes_
        self.mxWC = 0

        # Class of weak classifier (현재의 처리되는 weak clasifier...?)
        self.WCClass = weak_classifier_

        # self.WCs : [self.WCClass list] (weak classifier의 리스트)
        #   List of Weak classifiers.

        # self.nWC : [int] (현재 존재하는 weak classsifier의 개수)
        #   Number of weak classifiers used. (<= mxWC)
        self.nWC = 0

        # self.alpha : [float list] (각 weak classifier에 해당하는 가중치값: alpha_t)
        #   Alpha contains weights of each weak classifier,
        #   ie. votes they have.

        # self.features : [int] (feature 개수 - 현재 과제에서는 약 160,000개)
        #   Number of features.

        # self.sum_eval : [float] a1*h1 + a2*h2 + ... + at*ht의 결과값
        #   Sum of votes result of all evaluated
        #   weak classifiers.

    def train(self, X_, y_, n_classes_, is_continue=False, verbose=False):
        """Train the AdaBoost classifier with the training set (X, y).

        Parameters
        ----------
        X_ : array-like of shape = [n_samples, n_features]
            The inputs of the training samples.

        y_ : array-like of shape = [n_samples]
            The class labels of the training samples.

            Currently only supports class 0 and 1. (Viola Jones에서 이게 맞는 표현)
        """

        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = np.array(y_).flatten(1)
        y[y == 0] = -1  # 만약에 y가 0이라면 모두 -1로 바꾸어주기 (음...이 코드는 적용하지 말기)
        n_samples, n_features = X.shape

        assert n_samples == y.size

        if not is_continue or self.nWC == 0:
            # Initialize weak classifiers
            self.mxWC = n_classes_ # n_classes_가 뭐지? (최대 몇 개의 weak classifier로 할건지를 정해주는 듯)
            self.WCs = [copy.deepcopy(self.WCClass) for a in range(self.mxWC)]  # waek_classifier 생성!

            self.nWC = 0
            self.alpha = np.zeros((self.mxWC)) # 일단 alpha를 self.mxWC 크기의 numpy 배열에 0으로 지정
            self.features = n_features  # 총 feaure 개수 (160,000)
            self.sum_eval = 0           # 현재 구한 것도 없으니 sum_eval=0으로 지정

            # Initialize weights of inputs samples
            W = np.ones((n_samples)) / n_samples    # 초기 가중치는 모두 n빵 (/n_samples)

        else:
            # 아마도 else부분은 굳이 사용하지 않아도 될 듯
            # 최대 weak classifier 개수를 지정하다가 원하는 성능이 나오지를 않아서
            # 도중에서 weak classifier를 더 추가한 것 같은데
            # 애당초 원하는 FPR에 도달할 때까지 나는 weak classifier를 계속 추가할 예정
            # 처음이 아니라면
            self.mxWC = self.nWC + n_classes_ # 현재 클래스 개수에 (더 추가할 개수를 같이 더해줌) => self.mxWC 갱신

            # 기존에 있었던 self.WCs에 n_classes_개수만큼의 weak classifier를 합침
            self.WCs = np.concatenate((self.WCs[0:self.nWC], [copy.deepcopy(self.WCClass) for a in range(n_classes_)]))
            # 그렇게 aplha도 모두 합침
            self.alpha = np.concatenate((self.alpha[0:self.nWC], np.zeros((n_classes_))))
            # 가중치는 내부에 만들어졌던 거 그대로
            W = self.W


        for a in range(self.nWC, self.mxWC):
            # 추가 안했다면: 0~self.mxWC까지
            # 추가 했다면: self.nWC~self.mxWC까지

            if verbose: # 터미널에 정보 띄우고 싶다면
                # a 번째 weak classifier가 학습되고 있음을 보여준다.
                print('Training %d-th weak classifier' % a)

            # a번째 weak classifier에서 train하여 나온 err를 가져옴
            err = self.WCs[a].train(X, y, W, verbose)


            # error가 아예 0이 되거나, 1이 되는 순간에는
            # 알고리즘 구현과정에서 (beta 혹은 alpha)
            # DevisionZeroError가 발생하기 때문에 이를 막아줄 필요가 있음
            if err == 0: err = 1e-6
            elif err == 1: err = 1 - 1e-6

            # 학습이 다 끝난 a번째 weak classifier에 대해 X를 두고 다시 prediction 진행
            # h에는 n_samples에 대한 predictions이 존재
            h = self.WCs[a].predict(X).flatten(1)


            # self.alpha에 대해 train에서 나온 error를 가지고 구하게 됨
            # (다만, 이건 제공받은 PPT자료와 다름)
            self.alpha[a] = 0.5 * np.log((1 - err) / err)
            # 해당 alpha를 가지고 W를 갱신 (다만, 이건 제공받은 PPT 자료와 다름)
            W = W * np.exp(-self.alpha[a]*y*h)
            # 가중치 갱신
            W = W / W.sum()

            # # 바로 위 3줄에 코드에 대해 나라면 다음과 같이 진행
            # beta = err / (1-err)
            # self.alpha[a] = 1 / beta
            # W  = W * beta ** (1-abs(h - y))
            # W = W / W.sum()

            # result = 0
            # threshold = 0
            # for t in range(len(self.WCs)):
            #     result += self.alpha[t] * self.WCs[t].predict(X)
            #     threshold += 0.5 * np.log(self.alpha[t])

            # h = np.ones((n_samples))
            # h[result >= threshold] =  1
            # h[result < threshold]  = -1


            self.nWC = a+1
            if verbose: print('%d-th weak classifier: err = %f' % (a, err))
            if self._evaluate(a, h, y) == 0:
                print(self.nWC, "weak classifiers are enought to make error rate reach 0.0")
                break

        # 그렇게 가중치를 self.W에 갱신
        self.W = W

    def _evaluate(self, t, h, y):
        """Evaluate current model.

        Parameters
        ----------
        t : int
            Index of the last weak classifier.
            가장 마지막 단의 weak classifier

        h : np.array of shape = [n_samples]
            The predict result by the t-th weak classifier.
            t번째의 weak classifier의 예측

        y : np.array of shape = [n_samples]
            The class labels of the training samples.
            training sample의 정답지

        Returns
        -------
        cnt : int
            Count of mis-classified samples.
            잘못 분류한 샘플의 개수
        """

        self.sum_eval = self.sum_eval + h*self.alpha[t]
        # t번째 에측 * alpha[t] >> 교수님께서 제공해주신 파일에서는 어떻게 구성되어 있는지 파악하기

        yPred = np.sign(self.sum_eval) # sign으로 0, 1로 예측하네...?

        return np.sum(yPred != y) # Prediction과 다른 것만 뽑아서 더하기

    def predict(self, test_set_):
        """Predict the classes of input samples

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predict result of the testing samples.
        CI : np.array of shape = [n_samples]
            The confidence of the predict results.
            Confidence[신뢰도!]: 이걸로 overlapped detection window 문제를 해결하자!
        """

        # 모든 weak classifier에 대핸 weightedSum을 구함
        # husm [n_samples]
        hsum = self.weightedSum(test_set_)
        CI = abs(hsum) / np.sum(abs(self.alpha))

        yPred = np.sign(hsum) # sign으로 결정하네...교수님 코드 참고해보기
        yPred[yPred == -1] = 0 # -1로 답한 건 0으로 (이것도 과연 사용할지의 여부는 미지수)

        return yPred, CI

    def weightedSum(self, test_set_):
        """Return the weighted sum of all weak classifiers

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, n_features]
            The inputs of the testing samples.
            테스트 샘플을 싹다 가져옴 [n_samples, n_features]

        Returns
        -------
        hSum : np.array of shape = [n_samples]
            The predict result of the testing samples.
            각각의 m_samples에 대해 처리된 weak classifier의 weightedSum
        """

        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)

        assert test_set.shape[1] == self.features

        hsum = 0
        for i in range(self.nWC):
            # i번째 weak classifier에 대한 hypothesis * i번째 alpha
            hsum = hsum + self.alpha[i] * self.WCs[i].predict(test_set)

        return hsum

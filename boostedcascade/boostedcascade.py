#
# boostedcascade.py
#   Boosted cascade proposed by Viola & Jones
#
# Author : Donny
#

import os
import copy
import math
import time
import sys
import multiprocessing as mp
import numpy as np
from sklearn.utils import shuffle as skshuffle

from .adaboost import AdaBoostClassifier        # AdaBoostClassifier (StrongClassifier)
from .adaboost import DecisionStumpClassifier   # WeakClassifier

from . import HaarlikeFeature, HaarlikeType     # HaarlikeFeature가 찐이고, HaarlikeType는 그냥 Type만 지정

class BoostedCascade:
    """Boosted cascade proposed by Viola & Jones

    Parameters
    ----------
    Ftarget : float
        The target maximum false positvie rate.
        목적으로 두는 FPR (보통은 0.000001)

    f : float
        The maximum acceptable false positive rate per layer.
        한 strong classifier에서 수용가능한 FPR

    d : float
        The minimum acceptable detection rate per layer.
        한 strong classifier에서 수용가능한 detection...???

    validset_rate : float, optional
        The ratio of valid set in the whole training set.
        전체 training dataset에 대한 validation dataset의 비율

    CIsteps : float, optional.
        The steps of decreasing confidence threshold in each cascaded classifier.
        각 strong classifier에서마다의 confidence threshold의 감소하는 정도
    """

    # Haar-like features
    Haarlike = HaarlikeFeature()
    # Class of strong classifier, usu. AdaBoostClassifier
    SCClass = AdaBoostClassifier(weak_classifier_ = DecisionStumpClassifier(100))

    def __init__(self,
                 Ftarget, f, d,
                 validset_rate = 0.3,
                 CIsteps = 0.05):

        # Target false positive rate.
        self.Ftarget = Ftarget

        # The maximum acceptable false positive rate per layer.
        # 근데 사실, stages의 개수를 먼저 정하고 나서 각 stage에서의 acceptable FPR를 구해야 하지 않을까???
        # 나라면,
        # self.first_stages_fpr = 0.4 # 처음 FPR을 얼마로 할지는 내가 정함
        # self.n_stages = n_stages
        # self.f = (self.Ftarget / self.first_stages_fpr) ** (1 / (self.n_stages + 1))

        self.f = f

        # The minimum acceptable detection rate per layer
        # 얘는 뭐지? FNR인가??? (False Negative Rate: 얼굴이 아니라고 하는 비율)
        # FNR 맞음
        self.d = d

        # The ratio of valid set in the whole training set.
        self.validset_rate = validset_rate

        # The steps of decreasing confidence threshold in each cascaded classifier.
        # 이건 왜 필요할까??
        self.CIsteps = CIsteps

        # The window size of detector
        self.detectWndW, self.detectWndH = (-1, -1)


        # self.P?? self.N??
        self.P = np.array([]); self.N = np.array([])

        # validation에 대한 X,y를 나타내는 듯
        self.validX = np.array([]); self.validy = np.array([])

        # feature의 개수 카운트 (처음에는 -1로 두기)
        self.features_cnt = -1
        # feature generator를 일단 빈 공간의 np.array로 지정
        self.features_descriptions = np.array([])

        # self.SCs : list of self.SCClass (즉, Strong Classifier Class를 의미)
        #   The strong classifiers.
        self.SCs = []

        # self.thresholds : list of float
        #   The thresholds of each strong classifier.
        #   각 strong classifier에서의 threshold (이걸을 내가 정의하는 듯?!)
        self.thresholds = []

        # self.SCn : list of int
        #   The number of features used.
        #   사용된 feature 개수 (Strong classifier에서를 말하는 건가?)
        #                     (그럼 weak classifier의 개수를 말하는 건가??)
        self.SCn = []

    def getDetectWnd(self):
        # Detection Wiondow의 width, height를 내보냄
        return (self.detectWndH, self.detectWndW)

    def architecture(self):
        #  layer: Strong classifier
        #  weak classifier

        # 즉, ind번째 StrongClassifier에서의 weak_classifier으ㅐ 개수를 알려줌
        archi=""
        for ind, SC in enumerate(self.SCs):
            archi = archi + ("layer %d: %d weak classifier\n" % (ind, SC.nWC))

        # architecture 정보를 문자열로 계속 잇다가 해당 문자열을 리턴
        return archi

    def __str__(self):
        # 그냥 print()할 때 쓰는 친구
        return ("BoostedCascade(Ftarget=%f, "
                "f=%f, "
                "d=%f, "
                "validset_rate=%f, "
                "CIsteps=%f, "
                "detectWnd=%s, "
                "features_cnt=%d, "
                "n_strong_classifier=%d)") % (
                self.Ftarget,
                self.f,
                self.d,
                self.validset_rate,
                self.CIsteps,
                (self.detectWndH, self.detectWndW),
                self.features_cnt,
                len(self.SCs))

    # feature를 저장할 떄 필요 (학습하는 데 시간이 꽤 걸리는 듯)
    def savefeaturesdata(self, filename):
        os.makedirs(os.path.split(filename)[0], exist_ok=True)
        np.save(filename+'-variables', [self.detectWndH, self.detectWndW, self.features_cnt])
        np.save(filename+'-features_descriptions', self.features_descriptions)
        # 아...! P는 Positive, N은 Negative를 의미하는 건가?
        np.save(filename+'-P', self.P)
        np.save(filename+'-N', self.N)
        np.save(filename+'-validX', self.validX)
        np.save(filename+'-validy', self.validy)

    # feature를 로드할 때 필요
    def loadfeaturesdata(self, filename):
        detectWndH, detectWndW, features_cnt = \
            np.load(filename+'-variables.npy')
        self.detectWndH, self.detectWndW = int(detectWndH), int(detectWndW)
        self.features_cnt = int(features_cnt)
        self.features_descriptions = \
            np.load(filename+'-features_descriptions.npy')
        self.P = np.load(filename+'-P.npy')
        self.N = np.load(filename+'-N.npy')
        self.validX = np.load(filename+'-validX.npy')
        self.validy = np.load(filename+'-validy.npy')

    # 병렬로 Feature를 뽑아낸다: 24x24 size의 detection window(cropped face image) 위에서 feature를 뽑아냄
    def _parallel_translate(self, tid, range_, raw_data, result_output, schedule_output):
        assert type(range_) == tuple

        for n in range(range_[0], range_[1]):
            schedule_output.value = (n - range_[0]) / (range_[1] - range_[0])
            x = self.Haarlike.extractFeatures(raw_data[n], self.features_descriptions)
            # result_output은 아마도 'call by reference'인 듯
            result_output.put((n, x))

    # _parallel_translate애서 batch단위로 Feature를 뽑아낸다.
    #  _translate에서는 모든 feature를 저장
    def _translate(self, raw_data, verbose=False, max_parallel_process=8):
        n_samples, height, width = np.shape(raw_data)

        assert height == self.detectWndH and width == self.detectWndW, \
               "Height and width mismatch with current data."

        processes = [None] * max_parallel_process
        schedules = [None] * max_parallel_process
        results = mp.Queue()

        blocksize = math.ceil(n_samples / max_parallel_process)
        if blocksize <= 0: blocksize = 1
        for tid in range(max_parallel_process):
            schedules[tid] = mp.Value('f', 0.0)

            ##################################################
            blockbegin = blocksize * tid
            if blockbegin >= n_samples: break
            blockend = blocksize * (tid+1)
            if blockend > n_samples: blockend = n_samples
            ##################################################

            processes[tid] = mp.Process(target=__class__._parallel_translate,
                args=(self, tid, (blockbegin, blockend), raw_data, results, schedules[tid]))
            processes[tid].start()

        X = np.zeros((n_samples, self.features_cnt)) # 미리 n_samples x feature_cnt 크기의 공간을 생성
        while True:
            alive_processes = [None] * max_parallel_process
            for tid in range(max_parallel_process):
                alive_processes[tid] = processes[tid].is_alive()
            if sum(alive_processes) == 0:
                break

            while not results.empty():
                ind, x = results.get()
                X[ind] = x

            if verbose:
                for tid in range(max_parallel_process):
                    schedule = schedules[tid].value
                    print('% 7.1f%%' % (schedule * 100), end='')
                print('\r', end='', flush=True)

            time.sleep(0.2)

        sys.stdout.write("\033[K")

        # 결국에는 self.feature_cnt의 features를 가진
        # n_samples개의 sample을 얻어내는 듯
        return X

    def prepare(self, P_, N_, shuffle=True, verbose=False, max_parallel_process=8):
        """Prepare the data for training.

        Parameters
        ----------
        P_ : array-like of shape = [n_positive_samples, height, width]
            The positive samples.
            그치, 24x24 detection window을 학습하기 위해서는
                 24x24 Positive Sample이 필요함

        N_ : array-like of shape = [n_negetive_samples, height, width]
            The negetive samples.
            그치, 24x24 detection window을 학습하기 위해서는
                 24x24 Negative Sample이 필요함

        shuffle : bool
            Whether to shuffle the data or not.
        """

        # Positive Sample, Negative Sample의 형상은 서로 같아야 함 (모두 같아야 한다.)
        assert np.shape(P_)[1:3] == np.shape(N_)[1:3], "Window sizes mismatch."
        _, self.detectWndH, self.detectWndW = np.shape(P_)

        # 총 feature_cnt, feature_generator를 뽑아낸다.
        # 여기서는 각 Positive Sample, Negative Sample에 대해 따로 mean, std를 구하지 않는다.
        # mean, std는 이제 각 Sample에서 Normalize를 진행할 때 그때 같이 곁들어서 쓰이는 것이다.
        # 결론: determineFeature에서는 feature만 generate!!
        self.features_cnt, descriptions = \
            self.Haarlike.determineFeatures(self.detectWndW, self.detectWndH)
        # 그냥 거꾸로 돌린 것임
        self.features_descriptions = descriptions[::-1]

        if shuffle: # 뭐 사실 shuffle없어도 되기는 하지만, 있어도
            # If P_ is a list, this is faster than
            # P_ = np.array(skshuffle(P_, random_state=1))
            P_ = skshuffle(np.array(P_), random_state=1)
            N_ = skshuffle(np.array(N_), random_state=1)

        # 터미널에서 현재 Sample들이 준비가 되었는지를 보고 싶다면
        if verbose: print('Preparing positive data.')
        P = self._translate(P_, verbose=verbose, max_parallel_process=max_parallel_process)
        if verbose: print('Preparing negative data.')
        N = self._translate(N_, verbose=verbose, max_parallel_process=max_parallel_process)

        # validset_rate로 training dataset과 validation dataset를 나눈다.
        # overfitting을 방지하기 위해서
        divlineP = int(len(P)*self.validset_rate)
        divlineN = int(len(N)*self.validset_rate)

        # 그 비율(self.validset_rate)에 해당하는 Positive Sample과 Negative Sample를 담는다.)
        validset_X = np.concatenate(( P[0:divlineP], N[0:divlineN] ))
        # 그때에 정답지는 Positive는 1, Negative는 0으로 둔다.
        validset_y = np.concatenate(( np.ones(len(P[0:divlineP])), np.zeros(len(N[0:divlineN])) ))
        # validset_X, validset_y = skshuffle(validset_X, validset_y, random_state=1)

        # P,N은 이제 실제 Training dataset을 의미
        P = P[divlineP:len(P)]
        N = N[divlineN:len(N)]

        self.P = P
        self.N = N
        self.validX = validset_X
        self.validy = validset_y

    def train(self, is_continue=False, autosnap_filename=None, verbose=False):
        """Train the boosted cascade model."""

        # Preparing이 되어있지 않는다면
        assert self.detectWndW != -1 and self.detectWndH != -1 and \
               len(self.P) != 0 and len(self.N) != 0 and \
               self.features_cnt != -1 and \
               len(self.features_descriptions) != 0, \
               "Please call prepare first."

        # 멤버변수에서 Positive Sample과 Negative Sample을 꺼내 쓰기
        P = self.P
        N = self.N

        self._initEvaluate(self.validX, self.validy) # 이건 뭐지?

        # 음? 다시 self.P , self.N을 비운다고??? 비효율적인데?
        self.P = np.array([]); self.N = np.array([])
        self.validX = np.array([]); self.validy = np.array([])

        # 음? 이건 뭐지?
        f1 = 1.0    # 초기 FPR
        D1 = 1.0    # 얘는?

        if not is_continue: # 처음이면 Strong Classifier List, Threshold List, n_features 생성
            self.SCs = []
            self.thresholds = []
            self.SCn = []
        ###########################################################
        else:               # 처음이 아니면 (C++에서는 굳이 이 내용 필요없을 듯)
            yPred = self._predictRaw(N)
            N = N[yPred == 1]

            for ind in range(len(self.SCs)):
                ySync, f1, D1, _ = self._evaluate(ind)
                self._updateEvaluate(ySync)
        ###########################################################

        features_used = self.features_cnt # 총 feature 개수 (160,000)
        n_step = 1
        # n_step = int(self.features_cnt/400)
        # if n_step == 0: n_step = 1


        print('Begin training, with n_classes += %d, n_step = %d, Ftarget = %f, f = %f, d = %f'
            % (1, n_step, self.Ftarget, self.f, self.d))

        itr = 0
        while f1 > self.Ftarget: # desired FPR에 도달할 때까지 loop (초기 f1은 0.0)
            itr += 1
            f0 = f1
            D0 = D1
            n = 0
            # n = int(self.features_cnt/400)
            # n = self.features_cnt

            print('Training iteration %d, false positive rate = %f' % (itr, f1))

            training_X = np.concatenate(( P, N ))
            training_y = np.concatenate(( np.ones(len(P)), np.zeros(len(N)) ))
            training_X, training_y = skshuffle(training_X, training_y, random_state=1)

            classifier = copy.deepcopy(self.SCClass) # AdaBoostClassifier 생성

            self.SCs.append(copy.deepcopy(classifier)) # Stong Classifier List에 Append

            self.thresholds.append(1.0) # 초기에는 theshold를 1.0를 넣는다라...
            self.SCn.append(features_used)  # Scn에는 사용되는 feature개수를 append (아마도 n_weak_clasifers 아닐까 )

            while f1 > self.f * f0: # 음...좀 더 알아보기

                n = n_step # n += n_step

                # if n > self.features_cnt: n = self.features_cnt

                ind = len(self.SCs) - 1 # 현재 만들어진 Strong Classifier의 개수 - 1

                print('Itr-%d: Training %d-th AdaBoostClassifier, features count + %d, detection rate = %f, false positive rate = %f'
                    % (itr, ind, n, D1, f1))
                if verbose:
                    print('Aim detection rate : >=%f; Aim false positive rate : <=%f'
                        % (self.d * D0, self.f * f0))
                    print('Positive samples : %s; Negative samples : %s'
                        % (str(P.shape), str(N.shape)))

                # classifier = copy.deepcopy(self.SCClass)
                # classifier.mxWC = n
                # 그니까, n은 이제 weak_classifier의 개수를 의미
                # 여기서는 weak_classifier를 하나씩 생성
                # features_used로 사용할 feature 개수를 정하는 것 같은데, C++에서는 160,000개를 모두 사용하지 않을까???
                classifier.train(training_X[:, 0:features_used], training_y, n, is_continue=True, verbose=verbose)
                if verbose:
                    for a in range(classifier.nWC):
                        print('%d-th weak classifier select %s as its feature.'
                            % (a, str(self.features_descriptions[classifier.WCs[a].bestn])))

                # 현재 처리되고 있는 ind번째 Strong Classifier List 공간에 trained classifier를 저장
                self.SCs[ind] = copy.deepcopy(classifier)
                # 그때의 threshold도 1.0을 저장
                self.thresholds[ind] = 1.0
                #  ind번째 Strong Classifier <n_weak_classifiers>에 사용되었던 features의 개수 저장
                self.SCn[ind] = features_used

                # 내부에서 ind번째의 Strong Claaisifer를 가지고 evaluate
                # print를 보아하니, adjust threshold 및 FPR 및 FNR을 구하는 듯
                # D0=1.0, D1=FNR=0.6
                ySync, f1, D1, _ = self._evaluate(ind)
                print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f'
                    % (self.thresholds[ind], D1, f1))

                # D1보다 FNR * D0이 크다면
                # 이때 self.d는 The minimum acceptable detection rate per layer
                # (ex1) D0=1.0, D1=0.5, self.d=0.1이면
                # D1 < self.d * D0 => 0.5 < 0.1 (False) 이므로 loop 탈출

                # (ex2) D0=0.5, D1=0.002, self.d=0.1이면
                # D1 < self.d * D0 => 0.002 < 0.05 이므로 loop 돌아감 (self.d에 도달할 때까지)
                while D1 < self.d * D0:
                    # thesholds를 '일부러' 낮춤으로써, FNR을 조금씩 높임
                    self.thresholds[ind] -= self.CIsteps

                    # theshold 마지노선은 -1.0까지
                    if self.thresholds[ind] < -1.0: self.thresholds[ind] = -1.0

                    # 조정되 threshold를 가지고 다시 evaluate 진행
                    ySync, f1, D1, _ = self._evaluate(ind)
                    print('Threshold adjusted to %f, detection rate = %f, false positive rate = %f'
                        % (self.thresholds[ind], D1, f1))

            self._updateEvaluate(ySync)

            if f1 > self.Ftarget:
                yPred = self._predictRaw(N)
                N = N[yPred == 1]

            if autosnap_filename:
                self.saveModel(autosnap_filename)

        print('%d cascaded classifiers, detection rate = %f, false positive rate = %f'
            % (len(self.SCs), D1, f1))


    def _initEvaluate(self, validset_X, validset_y):
        """Initialize before evaluating the model.
        모델을 평가하기 전에 초기화하는 부분?
        Parameters
        ----------
        validset_X : np.array of shape = [n_samples, n_features]
            The samples of the valid set.

        validset_y : np.array of shape = [n_samples]
            The ground truth of the valid set.
        """
        # _initEvaluate에 미리 validset에 대한모든 정보를 담아놓늗다.
        class Eval: pass                                        # 클래스 생성?
        self._eval = Eval()                                     # self._eval에 Eval()을 호출?
        self._eval.validset_X = validset_X
        self._eval.validset_y = validset_y
        self._eval.PySelector = (validset_y == 1)               # 정답지가 1인 것만 싹다 가져옴: Positive Selector
        self._eval.NySelector = (validset_y == 0)               # 정답지가 0인 것만 싹다 가져옴: Negative Selector
        self._eval.cP = len(validset_y[self._eval.PySelector])  # Positive Selector로 valid_set의 Positive Sample만 취함
        self._eval.cN = len(validset_y[self._eval.NySelector])  # Negative Selector로 valid_set의 Negative Sample만 취함
        self._eval.ySync = np.ones(len(validset_y)) # All exist possible positive - 그니까 validset_y의 크기만큼에 원소는 모두 1인 numpy 배열을 생성
        pass

    def _evaluate(self, ind):
        """Evaluate the model, but won't update any parameter of the model.
        모델을 평가하는 용도이긴 하나, 이걸 가지고 파라미터를 업데이트하지는 않음

        Parameters
        ----------
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the valid set.
            valid_set의 입력

        y_ : np.array of shape = [n_samples]
            The ground truth of the valid set.
            valid_set의 정답지

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result.
            preditction한 결과

        f : float
            The false positive rate.

        d : float
            The detection rate.

        fp : np.array
            The false positives.
            FP: 벽을 보고 얼굴이라고 하는 경우
        """

        # 이미 self._eval에서 저장했던 validset_X와 validset_y를 가져옴
        X_ = self._eval.validset_X
        y_ = self._eval.validset_y

        # ySync역시 가져옴 (카피해서)
        ySync = self._eval.ySync.copy()

        # yiPred, CI = self.SCs[ind].predict(X_[ySync == 1][:, 0:self.SCn[ind]])
        # ind번째 Stong Classifier가 ind번째의 features를 가지고 prediction을 진행
        # 이때 Prediction과 Confidence가 같이 리턴됨
        yiPred, CI = self.SCs[ind].predict(X_[:, 0:self.SCn[ind]])
        CI[yiPred != 1] = -CI[yiPred != 1] # 1이 아닌 걸 왜 음수로 바꾸지??

        # Reject those whose confidences are less that thresholds
        # Strong Classifier의 threshold보다 낮은 confidence들은 모두 거부
        yiPred = (CI >= self.thresholds[ind]).astype(int) # 그렇게 걔네들만 True보냄

        # 얘는 뭐지?
        ySync[ySync == 1] = yiPred # Exclude all rejected

        fp = (ySync[self._eval.NySelector] == 1)
        dp = (ySync[self._eval.PySelector] == 1)
        f = (np.sum(fp) / self._eval.cN) if self._eval.cN != 0.0 else 0.0 # Negative Sample 중에서 Positive이라고 예측한 경우 (FPR)
        d = (np.sum(dp) / self._eval.cP) if self._eval.cP != 0.0 else 0.0 # Positive Sample 중에서 Negative이라고 예측한 경우 (FNR)


        # 여기 코드는 내가 좀 더 음미하고 나서 다시 짜봐야 할듯
        # 결국에는 C++로 코드를 짜야하므로

        return ySync, f, d, fp

    def _updateEvaluate(self, ySync):
        """Update the parameter of the evaluating model.

        Parameters
        ----------
        ySync : np.array of shape = [n_samples]
            The classifier result generated by function 'evaluate'.
        """
        self._eval.validset_X = self._eval.validset_X[ySync[self._eval.ySync == 1] == 1]
        self._eval.ySync = ySync # Update ySync

    def _weakPredict(self, wcself, X_):
        """Predict function for weak classifiers.

        Parameters
        ----------
        wcself : instance of WeakClassifier
            The weak classifier.
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted result of the testing samples.
        """

        description = self.features_descriptions[wcself.bestn]

        feature = np.zeros(len(X_))
        for ind in range(len(X_)):
            feature[ind] = self.Haarlike._getFeatureIn(
                X_[ind],        # integral image
                description[0], # haartype
                description[1], # x
                description[2], # y
                description[3], # w
                description[4]  # h
            )

        h = np.ones(len(X_))
        h[feature*wcself.bestd < wcself.bestp*wcself.bestd] = -1
        return h

    def _strongPredict(self, scself, X_):
        """Predict function for the strong classifier (AdaBoostClassifier).

        Parameters
        ----------
        scself : instance of self.SCClass
            The strong classifier (AdaBoostClassifier).
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        CI : np.array of shape = [n_samples]
            The confidence of each predict result.
        """
        hsum = 0
        for i in range(scself.nWC):
            hsum = hsum + scself.alpha[i] * self._weakPredict(scself.WCs[i], X_)

        yPred = np.sign(hsum)
        yPred[yPred == -1] = 0
        CI = abs(hsum) / np.sum(scself.alpha)

        return yPred, CI

    def translateToIntegralImage(self, image):
        return self.Haarlike._getIntegralImage(image)

    def predictIntegralImage(self, test_set_integral_images):
        X = test_set_integral_images
        yPred = np.ones(len(X))
        for ind in range(len(self.SCs)):
            yiPred, CI = self._strongPredict(self.SCs[ind], X[yPred == 1])
            CI[yiPred != 1] = -CI[yiPred != 1]
            yiPred = (CI >= self.thresholds[ind]).astype(int)
            # yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected

        return yPred

    def predict(self, test_set_):
        """Predict whether it's a face or not.

        Parameters
        ----------
        test_set_ : array-like of shape = [n_samples, height, width]
            The inputs of the testing samples.

        Returns
        -------
        yPred : np.array of shape = [n_samples]
            The predicted results of the testing samples.
        """
        X = np.zeros((len(test_set_), self.detectWndH+1, self.detectWndW+1))
        for i in range(len(test_set_)):
            X[i] = self.Haarlike._getIntegralImage(test_set_[i])
        return self.predictIntegralImage(X)



    def preparePredictRaw(self, P_, N_, verbose=False):
        # 나중에 외부에서 Test할때 이걸로 Test Positive Sample 및 Test Negative Sample을
        # self.P, self.N에 저장
        P = np.zeros((len(P_), self.features_cnt))
        N = np.zeros((len(N_), self.features_cnt))
        for i in range(len(P_)):
            if verbose: print('Preparing positive data NO.%d.' % i)
            P[i] = self.Haarlike.extractFeatures(P_[i], self.features_descriptions)
        for j in range(len(N_)):
            if verbose: print('Preparing negative data NO.%d.' % j)
            N[j] = self.Haarlike.extractFeatures(N_[i], self.featuers_descriptions)

        self.P = P
        self.N = N

    def _predictRaw(self, test_set_):
        yPred = np.ones(len(test_set_))
        for ind in range(len(self.SCs)):
            yiPred, CI = self.SCs[ind].predict(test_set_[yPred == 1][:, 0:self.SCn[ind]])
            CI[yiPred != 1] = -CI[yiPred != 1]
            yiPred = (CI >= self.thresholds[ind]).astype(int)
            # yiPred[yiPred == 1] = (CI[yiPred == 1] >= self.thresholds[ind]).astype(int)
            yPred[yPred == 1] = yiPred # Exclude all rejected

        return yPred

    def predictRaw(self):
        X = np.concatenate((self.P, self.N))
        yPred = self._predictRaw(X)
        return yPred[0:len(self.P)], yPred[len(self.P):len(X)]

    def saveModel(self, filename):
        np.save(filename+'-variables', [
            self.Ftarget, self.f, self.d, self.validset_rate, self.CIsteps,
            self.detectWndH, self.detectWndW, self.features_cnt,
        ])
        np.save(filename+'-features_descriptions',
            self.features_descriptions)
        np.save(filename+'-thresholds', self.thresholds)
        np.save(filename+'-SCs', self.SCs)
        np.save(filename+'-SCn', self.SCn)

    def loadModel(filename):
        Ftarget, f, d, validset_rate, CIsteps, \
        detectWndH, detectWndW, features_cnt   \
            = np.load(filename+'-variables.npy')
        model = BoostedCascade(Ftarget, f, d, validset_rate, CIsteps)
        model.detectWndH, model.detectWndW = int(detectWndH), int(detectWndW)
        model.features_cnt = int(features_cnt)

        model.features_descriptions = \
            np.load(filename+'-features_descriptions.npy')
        model.thresholds = list(np.load(filename+'-thresholds.npy'))
        model.SCs = list(np.load(filename+'-SCs.npy'))
        model.SCn = list(np.load(filename+'-SCn.npy'))
        return model

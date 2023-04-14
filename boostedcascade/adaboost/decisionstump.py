
# Author: Donny

import math
import time
import sys
import multiprocessing as mp
import numpy as np

class DecisionStumpClassifier:
    """A decision stump classifier

    Parameters
    ----------
    steps_ : int, optional
        The steps to train on each feature.
    """

    def __init__(self, steps_=400, max_parallel_processes_=8):
        # self.features : int
        #   Number of features. (about 160,000)

        # self.bestn : int
        #   Best feature.   (Best feature)

        # self.bestd : int of -1 or 1
        #   Best direction. (polarity)

        # self.bestp : float
        #   Best position. (threshold)

        # self.steps : int
        #   Count of training iterations.

        self.steps = steps_
        self.max_parallel_processes = max_parallel_processes_

    def train(self, X_, y_, W_, verbose=False):
        """Train the decision stump with the training set {X, y}

        Parameters
        ----------
        X_ : np.array of shape = [n_samples, n_features]
            The inputs of the training samples.

        y_ : np.array of shape = [n_samples]
            The class labels of the training samples.
            Currently only supports class -1 and 1.
            다만, 로봇비전응용의 PPT에서는 class 0,1로 지정

        W_ : np.array of shape = [n_samples]
            The weights of each samples.

        Returns
        -------
        err : float
            The sum of weighted errors.
        """

        # numpy array 형태로 바꾸기
        X = X_ if type(X_) == np.ndarray else np.array(X_)
        y = y_ if type(y_) == np.ndarray else np.array(y_)
        W = W_ if type(W_) == np.ndarray else np.array(W_)

        steps = self.steps # 현재 step을 저장 (time step인가?? 그럼 step번째 weakclassifier가 될 듯)

        n_samples, n_features = X.shape
        assert n_samples == y.size  # X의 개수와 y의 개수가 맞는 지를 확인

        # 왜 parallel_processes(병렬 처리)로 진행할까??
        processes = [None] * self.max_parallel_processes
        schedules = [None] * self.max_parallel_processes
        results = [None] * self.max_parallel_processes


        blocksize = math.ceil(n_features / self.max_parallel_processes) # math.ceil(): 올림 함수
        if blocksize <= 0: blocksize = 1

        for tid in range(self.max_parallel_processes):  # 병렬처리 단위만큼 iterate
            schedules[tid] = mp.Value('f', 0.0)
            results[tid] = mp.Queue()

            # block_start과 block_end를 지정 (tid가 아마 병렬처리의 한 단위인듯)
            blockbegin = blocksize * tid
            if blockbegin >= n_features: break; # Has got enough processes
                                                # (죽, block_start가 n_feautre를 넘긴 상황 - 이마 마지막 번지수의 n_feature도 이미 끝난 상황)
            blockend = blocksize * (tid+1)
            if blockend > n_features: blockend = n_features # block_end가 이미 feature 개수보다 넘으면 가장 마지막에 있는 n_feature로 지정

            # 그렇게 mp.Process로 병렬처리 시작
            # 160,000개를 일일히 돌리는 굉장히 부담이 가는 Task
            processes[tid] = mp.Process(target=__class__._parallel_optimize,
                # 여기서 160,000개의 features 중에서 optimal threshold를 구하고 그에 따른 error를 다시 갱신
                # target = __class__.parallel_optimize 함수를 호출 (자신의 클래스에서 호출)
                args=(self, tid, (blockbegin, blockend), results[tid], schedules[tid], X, y, W, steps))
            processes[tid].start()

        # 일단 verbose의 default가 False긴 하나,
        if verbose:
            while True:
                alive_processes = [None] * self.max_parallel_processes
                for tid in range(self.max_parallel_processes):
                    alive_processes[tid] = processes[tid].is_alive()    # tid번쩨 processes의 상태를 혹인하여
                if sum(alive_processes) == 0:   # 모든 processes가 alive하지 않는다면
                    break   # loop 빠져나옴

                for tid in range(self.max_parallel_processes):
                    schedule = schedules[tid].value # tid번째 schedules을 값을 모두 변수에 대입하여
                    print('% 7.1f%%' % (schedule*100), end='') # 터미널에 보고하기 (스케쥴링 보려고) -> 멀티스레드에서 스케쥴이 무슨 의미일까??
                print('\r', end='', flush=True)

                time.sleep(0.2)
            sys.stdout.write("\033[K") # Clear line

        bestn = 0
        bestd = 1
        bestp = 0
        minerr = W.sum()    # mp.Process()에서 이미 진행된 error의 총합을 구함
                            # e_t = sum(i,n)(w_t^i | h_t(x_i) - y_i |)

        for tid in range(self.max_parallel_processes):
            processes[tid].join()
            result = results[tid].get() # results의 tid번째를 가져옴
            if result['minerr'] < minerr:
                minerr = result['minerr']
                bestn = result['bestn'] # 160,000개 중에서 best feature 선정
                bestd = result['bestd'] # 해당 best feature에,
                                        # n_samples (얼굴 이미지 개수) 중에서 best threshold 선정
                bestp = result['bestp'] # polarity = 1 또는 polarity = -1 선정

        # 그렇게 n_features, best feature, optimal threshold, best polarity를 멤버 변수에 저장
        self.features = n_features
        self.bestn = bestn
        self.bestd = bestd
        self.bestp = bestp

        return minerr

    def _parallel_optimize(self, tid, range_, result_output, schedule, X, y, W, steps):
        assert type(range_) == tuple    # range_가 튜플인지를 파악

        bestn = 0   # best feature는 일단 0번지에서부터 시작
        bestd = 1   # best theshold도 일단 1번지에서터 시작 (index는 0번지에서 시작하는 거 아닌가?)
        bestp = 0   # 이 개발자는 polarity를 0 또는 1로 한 듯
        minerr = W.sum()    # 이전의 가중치들의 합


        # 이 for문을 loop하면서 결국에서는 best feature와 best threshold를 얻게 됨
        for n in range(range_[0], range_[1]):
            # setting and getting a float value is thread-safe
            schedule.value = (n - range_[0]) / (range_[1] - range_[0])
            # self._optimize에서 살펴보기
            # 여기서 n은 batch_size 개념으로 보면 됨
            # i,i+n까지 한꺼번에 연산 시작
            err, d, p = self._optimize(X[:, n], y, W, steps)

            # 그래서 새로 얻은 err가 이전에 내가 계산한 minerr(즉 이전 에러 가중치들의 합보다 작으면)
            # minerr를 갱신
            if err < minerr:
                minerr = err
                bestn = n
                bestd = d
                bestp = p

        result = dict()
        result['bestn'] = bestn
        result['bestd'] = bestd
        result['bestp'] = bestp
        result['minerr'] = minerr
        result_output.put(result)

    def _optimize(self, X, y, W, steps):
        """Get optimal direction and position to divided X.
        한 feature에서, optimal threshold와 그때의 polarity를 구해보자!!

        Parameters
        ----------
        X : np.array of shape = [n_samples]
            The inputs of a certain feature of the training samples.
            (예를 들어)
            - FDDB의 데이터 개수가 10,000개면
            - 10,000개 중에서 n번째 feature만 가져온 것임
            - 이때 feature라 함은 (x,y)에 위치한 haarlike(2h, 2v, 3h, 3v, 4)에 있는 feature

            좀 더 구체적으로 들어가보자.

            1. 10,000개의 이미지들을 Normalize를 진행한다. (Color image -> Gray Scale -> Histogram Equalized)
            2. 10,000개의 FDDB에서 얼굴만 crop하여 24 x 24 size의 detection window를 만든다.
            3. 각 detection window에 대해서 Normalize를 한 번 더 거치게 되는데,
             - S번째 detection window를 뽑았고, 현재 window에서 (i, j)번째에 Haarlikefeature2h [-1 | 1]를 적용한다면

                detection window의 mean과 std를 먼저 구하고 나서
                f2h' = (1/std)*(dectection_window[j][i+1] - dectection_window[j][i])

             - 즉, fh2'가 그때 사용되는 S번째 이미지의 feature가 되는 것이다!

            3. 따라서 S=1~10,000를 모두 진행하면 f2h'는 총 10,000개가 된다.

        y : np.array of shape = [n_samples]
            The class labels of the training samples.
        W : np.array of shape = [n_samples]
            The weights of each samples.
        steps : int
            Count of training iterations.

        Returns
        -------
        err : float
            The sum of weighted errors.
        d : int of value -1 or 1
            The optimal direction.
        p : float
            The optimal position.
        """

        X = X.flatten(1)    # 뽑힌 feature를일단 평평하게

        min_x, max_x = X.min(), X.max() # 특정 feature로 뽑힌 10,000개의 sample에서 MAX와 MIN 찾기
        len_x = max_x - min_x # MAX와 MIN 사이의 크기

        bestd = 1        # polarity는 일단 1로 지정
        bestp = min_x    # 제일 작은 수치를 가진 feature값부터
        minerr = W.sum() # minerr는 이전에 가지고 있던 W를 가지고 진행 (뭔가 아직까지는 W를 갱신하는 부분은 나오지 않았음)

        if len_x > 0.0:
            # steps로 나눠도 되는 걸까???
            for p in np.arange(min_x, max_x, len_x/steps):
                for d in [-1, 1]: # polarity: +1 또느 -1로 봄
                    gy = np.ones((y.size))
                    gy[X*d < p*d] = -1
                    err = np.sum((gy != y)*W) # 틀린 것들만 이전 가중치에 대해 모두 합함
                    if err < minerr: # 위에서 구한 err가 minerr보다 작으면,
                        # 모두 갱신
                        minerr = err
                        bestd = d
                        bestp = p

        # 아무튼, 그렇게 S번째 feature에 대해
        # minerr, optimal threshold, 그때에 해당하는 polarity를 반환
        return minerr, bestd, bestp

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
        """

        test_set = test_set_ if type(test_set_) == np.ndarray else np.array(test_set_)
        n_samples, n_features = test_set.shape # n_sample과 n_feature를 미리 구하고

        assert n_features == self.features # Train에서 구했던 n_features와 Test에서 진행해던 n_feature에 대해 서로 같은지를 판단

        single_feature = test_set[:, self.bestn] # 현재 단일 weak classifier에 대하여 best feature에 해당하는 samples를 뽑고
        h = np.ones((n_samples))    # hypothesis는 다 1로 만든다음에
        h[single_feature*self.bestd < self.bestp*self.bestd] = -1 # 해당 sample에 대해 이미 처리된 optimal threshold로 결과 -1 또는 1 반환
        return h

#
# facedetector.py
#   A face detector using BoostedCascade.
#
# Author : Donny
#

import time
import ctypes
import multiprocessing as mp
import numpy as np
import scipy as sp
from boostedcascade import BoostedCascade, HaarlikeFeature, HaarlikeType

class FaceDetector:

    ModelFile = 'models/model-100-l7/' + 'x5large-2'

    def __init__(self, max_parallel_process=8):
        # Cascade Strong Classifier 생성
        self.boostedCascade = BoostedCascade.loadModel(__class__.ModelFile)
        self.detectWnd = self.boostedCascade.getDetectWnd()
        self.signal = mp.Value(ctypes.c_bool, False)
        self.image_queue = mp.Queue()  # multiprocess Queue 생성
        self.result_queue = mp.Queue() # multiprocess Queue 생성
        self.setParallel(max_parallel_process)

    def stopParallel(self):
        if self.signal.value:
            self.signal.value = False

    def setParallel(self, max_parallel_process):
        if max_parallel_process <= 0: max_parallel_process = 1
        self.max_parallel_process = max_parallel_process
        if self.signal.value: self.signal.value = False
        self.signal = mp.Value(ctypes.c_bool, True)
        self.processes = [None] * max_parallel_process
        for tid in range(max_parallel_process):
            self.processes[tid] = mp.Process(target=__class__._parallel_detect,
                args=(self.signal, self.image_queue, self.result_queue, self.boostedCascade))
            self.processes[tid].start()

    def _transformToData(integral_image, wndW, wndH, padX, padY):
        """Scan the integral_image and get subimage of size = [wndW, wndH],
        padding of each subimage is [padX, padY].
        즉, image에 대해 detector를 x,y방향으로 move 시킨다.
        """
        h, w = integral_image.shape
        data = []
        for y in range(0,h-wndH,padY):
            for x in range(0,w-wndW,padX):
                data.append( [ integral_image[y:y+wndH+1, x:x+wndW+1], x, y, wndW, wndH ] )
        return np.array(data)

    def _parallel_detect(signal, image_queue, result_queue, boostedCascade):
        while signal.value:
            if image_queue.empty():
                time.sleep(0.2)
            else:
                data = image_queue.get()
                if type(data) != type(None):
                    pred = boostedCascade.predictIntegralImage(data[:, 0]) # 여기서 이제 예측된 값들을 받아오는 듯
                    result_queue.put([len(data), data[pred == 1, 1:]])

    # 얼굴 감지 코드
    def detect(self, image, min_size=0.0, max_size=1.0, step=0.5, detectPad=(12, 12), verbose=False, getTotalTiles=False):
        faces = []  # 얼굴을 담을 공간
        height, width = image.shape # 테스트할 이미지의 세로, 너비

        if min_size * min(width, height) < 24:
            min_size = 24.0 / min(width, height)

        if max_size < min_size:
            print("[WARNING] max_size < min_size")
            max_size = min_size
        assert step > 0.0 and step < 1.0

        total_items = 0
        given_items = 0
        done_items = 0

        si = max_size # scaling할 때의 마지노선

        while True:
            # 왜 여기에 대해 이미지를 scaled하지?? (음....이미지가 너무크면 연산이 너무 많으면 이미지 자체를 줄이는 건가?)
            scaledimg = sp.misc.imresize(image, size=(int(si*height), int(si*width)), mode='F')
            # integral이미지로 만듦
            integral_image = self.boostedCascade.translateToIntegralImage(scaledimg)

            # 모든 영역을 다 뽑아온다
            data = __class__._transformToData(
                integral_image,
                self.detectWnd[0],
                self.detectWnd[1],
                detectPad[0],
                detectPad[1]
            )

            # 뽑아온 data에 대한 전체 개수 -> n_samples에 넣는다.
            n_samples, _ = np.shape(data)
            if verbose: print("tiles count:", n_samples)
            total_items += n_samples

            # n_samples만큼 반복
            # scale된 이미지에 대해 scale된 x,y,Wndx,Wndy를 다시 넣어줌
            for ind in range(n_samples):
                data[ind] = [
                    data[ind, 0],
                    int(data[ind, 1]/si),
                    int(data[ind, 2]/si),
                    int(data[ind, 3]/si),
                    int(data[ind, 4]/si)
                ]

            for ind in range(0, n_samples, 100):
                blockbegin = ind
                blockend = min(ind+100, n_samples)
                self.image_queue.put(data[blockbegin:blockend, :])
                given_items += blockend - blockbegin

            if si <= min_size: break
            si = si * step
            if si < min_size: si = min_size


        if verbose: print("Total tiles:", total_items)
        assert given_items == total_items

        while True:
            if self.result_queue.empty():   # 큐가 비어있으면
                print(' %d/%d' % (done_items, total_items), end='\r') # 출력하고 (얼마만큼의 이미지가 처리되었는지)
                if self.image_queue.empty() and self.result_queue.empty(): # image queue와 result_queue에 모두 비어있고
                    if done_items == total_items: # 다 처리가 되었다면
                        break # loop 빠져나옴
                time.sleep(0.2)
            else:
                numbers, scaledfaces = self.result_queue.get() # result_queue에 결과를 가져옴
                done_items += numbers
                for x, y, w, h in scaledfaces:
                    faces.append([x, y, w, h])                # 예측된 얼굴에 대한 (x,y,w,h)를 넘겨줌

        if getTotalTiles:
            return np.array(faces), total_items
        else:
            return np.array(faces)

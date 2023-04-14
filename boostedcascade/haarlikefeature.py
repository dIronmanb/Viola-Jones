#
# haarlikefeature.py
#   Extract haar-like features from images.
#
# Author : Donny
#

import numpy as np
from enum import Enum

# Haarlikefeature: 2v, 2h, 3v, 3h, 4
class HaarlikeType(Enum):
    TWO_HORIZONTAL = 0
    TWO_VERTICAL = 1
    THREE_HORIZONTAL = 2
    THREE_VERTICAL = 3
    FOUR_DIAGONAL = 4
    TYPES_COUNT = 5

class HaarlikeFeature:
    """Extract haar-like features from images."""
    HaarWindow = [
        (2, 1),
        (1, 2),
        (3, 1),
        (1, 3),
        (2, 2)
    ]

    def __init__(self):
        # self.wnd_size = (0, 0)
        pass

    def determineFeatures(self, width, height):
        """Determine the features count while the window is (width, height),
           as well as giving the descriptions of each feature.
           그니까 features를 매번 count해서 현재 어디까지 진행되고 있는지를 알아보겠다는 거지??

        Call this function before calling extractFeatures.

        Parameters
        ----------
        width : int
            The width of the window. (Detection Window)
        height : int
            The height of the window. (Detection Windodw)

        Returns
        -------
        features_cnt : int
            The features count while the window is (width, height).
            지금까지 카운팅된 feature 번지수
        descriptions : list of shape = [features_cnt, [haartype, x, y, w, h]]
            The descriptions of each feature.
        """
        features_cnt = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype] # h2, v2, h3, v3, 4를 통해 window_x, window_y를 결정

            for x in range(0, width-wndx+1): # detection_window_width - window_x + 1
                for y in range(0, height-wndy+1): # detection_window_height - window_y + 1

                    features_cnt += int((width-x)/wndx)*int((height-y)/wndy)
                    # 마치 Conv filter 처럼 생각하면 되는데,
                    # detection_window_size = (24,24)
                    # h2: [-1, -1, 1, 1]
                    #   : [-1, -1, 1, 1]
                    # window_x = 4
                    # window_y = 2
                    # --> features_cnt = (24-4/4) * (24-2/2) =  5*11 = 55

                    # 마치 (x=4,,y=2) size가 있는데
                    # stride가 (x=4, y=2)로 진행하자는 것이다.
                    # x의 경우 (24  +2p -f)/stirde_x + 1 = (24 +0 - 4)/4 + 1= 20/4 + 1 = 6 (음??? 좀 더 생각해보자)


        descriptions = np.zeros((features_cnt, 5)) # feature_cnt개수만큼 미리 feature를 담을 공간 생성
        ind = 0
        for haartype in range(HaarlikeType.TYPES_COUNT.value):
            wndx, wndy = __class__.HaarWindow[haartype]

            for w in range(wndx, width+1, wndx):
                for h in range(wndy, height+1, wndy):
                    for x in range(0, width-w+1):
                        for y in range(0, height-h+1):

                            descriptions[ind] = [haartype, x, y, w, h] # 그렇게 24x24의 detection window에서 모든 feature를 담아낸다.
                            ind += 1 # ind를 계속 increase

        # print(features_cnt, descriptions.shape)
        # self.wnd_size = (height, width)
        return features_cnt, descriptions

    def extractFeatures(self, ognImage_, features_descriptions):
        """Extract features from an image.

        ****Please call determineFeatures first.****

        Parameters
        ----------
        ognImage_ : array-like of shape = [height, width]
            The original image.
            (정말로 이미지가 들어온다.)

        Returns
        -------
        features : np.array of shape = [features_cnt, val]
        """
        ognImage = np.array(ognImage_)  # numpy로 변환
        height, width = ognImage.shape  # 높이, 너비 저장

        # Call determineFeatures first.
        # 이미 호춣된<determineFeatures>를 다시 가져와서
        features_cnt = len(features_descriptions) # feature_descriptions의 개수를 저장
        descriptions = features_descriptions      # featute_descriptions을 가져옴

        # assert (height, width) == self.wnd_size

        features = np.zeros((int(features_cnt))) # 일단 모두 0번으로 만들고 (feature generator의 결과를 저장할 공간)

        itgImage = self._getIntegralImage(ognImage) # 원래 이미지를 -> Integral이미지로 만들고

        cnt = 0
        for description in descriptions:
            # 그렇게 모든 description에 대해 Feature 추출
            # Integral Image로 인하여 더 빠른 연산
            features[cnt] = self._getFeatureIn(
                itgImage,                           # Integral Image
                HaarlikeType(description[0]),       # haartype
                description[1],                     # x (시작점 x 좌표)
                description[2],                     # y (시작점 y 좌표)
                description[3],                     # w (끝점 x+w)
                description[4]                      # h (끝점 y+h)
            )
            cnt += 1                                # 그렇게 feature를 하나씩 차곡차곡 쌓아간다. (총 160,000개까지)

        return features


    def _getIntegralImage(self, ognImage):
        """Get the integral image.

        Integral Image:
        이렇게 저장하면, img(left-1,bottom), img(right, top-1), img(left-1, top-1) 등과 같이 'xxx - 1'와 같은 인덱싱 처리를 할 필요가 없어지게 된다.
        다만, 교수님께서 제공하신 코드로 그대로 이어나가자.
        + - - - - -        + -  -  -  -  -  -
        | 1 2 3 4 .        | 0  0  0  0  0  .
        | 5 6 7 8 .   =>   | 0  1  3  6 10  .
        | . . . . .        | 0  6 14 24 36  .
                           | .  .  .  .  .  .

        Parameters
        ----------
        _ognImage : np.array of shape = [height, width]
            The original image

        Returns
        -------
        itgImage : np.array of shape = [height+1, width+1]
            The integral image
        """
        h, w = ognImage.shape
        # print(w,h)
        itgImage = np.zeros((h+1, w+1))

        # 이런 방식도 좋고
        for y in range(1, h+1):
            for x in range(1, w+1):
                itgImage[y, x] = itgImage[y, x-1] + itgImage[y-1, x] - itgImage[y-1, x-1] + ognImage[y-1, x-1]

        # 다음과 같은 방식도 좋고(교수님이 제안하신 방식)
        # (굳이 넣지는 않음 - 어쩌피 C++로 코딩해야 하니)
        return itgImage

    def _getSumIn(self, itgImage, x, y, w, h):
        """
        intergral Image에서 원하는 4점을 가져와서 합을 구하게 된다.
        Get sum of image in rectangle [x, y, w, h]

        Parameters
        ----------
        itgImage : np.array of shape = [height+1, width+1]
            The integral image.
        x : int
            The starting column.
        y : int
            The starting row.
        w : int
            The width of the rectangle.
        h : int
            The height of the rectangle.

        Returns
        -------
        sum : int
            The sum of the pixels in the rectangle, excluding column w and row h.
        """
        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)
        return itgImage[h, w] + itgImage[y, x] - (itgImage[y, w] + itgImage[h, x])

    def _getFeatureIn(self, itgImage, feature_type, x, y, w, h):
        """Get haar feature in rectangle [x, y, w, h]

        Parameters
        ----------
        itgImage : np.array
            The integral image.
        feature_type : {HaarlikeType, number of HaarlikeType id}
            Tpye of the haar-like feature to extract.
        x : int
            The starting column.
        y : int
            The starting row.
        w : int
            The width of the rectangle.
        h : int
            The height of the rectangle.

        Returns
        -------
        diff : int
            The difference of white and black area, which represent the feature of the rectangle.
        """
        if not isinstance(feature_type, HaarlikeType):
            feature_type = HaarlikeType(feature_type)   # feature_type를 HaarlikeType으로 변환

        white = 0
        black = 0
        # 각 Haarlikefeature에 대한 처리
        if feature_type == HaarlikeType.TWO_HORIZONTAL:
            white = self._getSumIn(itgImage, x, y, w/2, h)
            black = self._getSumIn(itgImage, x + w/2, y, w/2, h)

        elif feature_type == HaarlikeType.TWO_VERTICAL:
            white = self._getSumIn(itgImage, x, y, w, h/2)
            black = self._getSumIn(itgImage, x, y + h/2, w, h/2)

        elif feature_type == HaarlikeType.THREE_HORIZONTAL:
            white = self._getSumIn(itgImage, x, y, w/3, h) + self._getSumIn(itgImage, x + w*2/3, y, w/3, h)
            black = self._getSumIn(itgImage, x + w/3, y, w/3, h)

        elif feature_type == HaarlikeType.THREE_VERTICAL:
            white = self._getSumIn(itgImage, x, y, w, h/3) + self._getSumIn(itgImage, x, y + h*2/3, w, h/3)
            black = self._getSumIn(itgImage, x, y + h/3, w, h/3)

        elif feature_type == HaarlikeType.FOUR_DIAGONAL:
            white = self._getSumIn(itgImage, x, y, w/2, h/2) + self._getSumIn(itgImage, x + w/2, y + h/2, w/2, h/2)
            black = self._getSumIn(itgImage, x + w/2, y, w/2, h/2) + self._getSumIn(itgImage, x, y + h/2, w/2, h/2)

        return white - black # 나중에 그냥 빼면 됨

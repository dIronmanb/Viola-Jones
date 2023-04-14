# 내가 구현해야 하는 것들

## 1. Decision Stamp

  - weak classifier를 만들어서 모든 feature에 대해:
  - optimal threshold / optimal feature를 구하기
  - 따로 weight를 조정하는 경우는 없음

## 2. Strong Classifier

  - weight를 갱신하는 건 여기서
  - beta와 alpha잘 구현하기

## 3. Feature Generator

  - Haarlikefeature를 만들었다면

## 4. Normalization Image

  - 모든 이미지들에 대해
  - Gray Scale  &rarr; Histogram Equalized 진행

## 5. How to extract to Feature
  - 24 x 24 window(임의의 이미지)에 대해 162,336개의 feature를 뽑아와보기
  - 4의 이미지에 대해
  - Normalized feature도 뽑아보기

## 6. Cascade Classifier

  - threshold를 조정을 어떻게 하는지 좀 더 파악하기
  - FPR, FNR 넘겨주는 것도
  - validation을 내부에 넣어주는 게 관건
  - Positive Sample, Negative Sample (24 x 24)를 다 넣어주어야 함


## 7. Overlap Problem

  - Confidence로 overlapping detection을 해결할 수 있을까??

## 8. Test

  -  테스트할 때는 임의의 테스트 이미지에 대해
  -  Scaled factor (1.25)를 사용해서 detection window를 키워나간다.
  -  detection window는 x축 방향으로 +1씩, y축 방향으로도 +1씩 키워나간다.
  -  그때에 해당하는 얼굴의 x,y,w,h를 뽑아와서 시각화 해보기

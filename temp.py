height, width = 24, 24


HaarWindow = [
  (2, 1),
  (1, 2),
  (3, 1),
  (1, 3),
  (2, 2)
]



'''나의 경우에는 그냥 바로 아래 코드만 사용하면 될 듯'''
ind = 0
for haartype in range(5):
    wndx, wndy = HaarWindow[haartype]
    for w in range(wndx, width+1, wndx):
        for h in range(wndy, height+1, wndy):
            for x in range(0, width-w+1):
                for y in range(0, height-h+1):
                    ind += 1 # ind를 계속 increase


features_cnt = 0
for haartype in range(5):
    wndx, wndy = HaarWindow[haartype] # h2, v2, h3, v3, 4를 통해 window_x, window_y를 결정
    for x in range(0, width-wndx+1): # detection_window_width - window_x + 1
        for y in range(0, height-wndy+1): # detection_window_height - window_y + 1
            features_cnt += int((width-x)/wndx)*int((height-y)/wndy)

print(f"ind: {ind}")
print(f"features_cnt: {features_cnt}")

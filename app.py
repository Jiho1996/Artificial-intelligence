import pandas as pd
import tensorflow as tf
import numpy as np

data = pd.read_csv('score.csv')

data = data.dropna()# 빈값있는 행은 제거해줌.

data['sat'] = (400*data['sat']/1000) # 400점 만점으로 값 변환

xdata = []
for i,rows in data.iterrows():
    xdata.append([ rows['sat'],rows['schoolgrade'],rows['rank'] ])

ydata = data['admit'].values



model = tf.keras.models.Sequential([
# layer 생성하기.
tf.keras.layers.Dense(64, activation='tanh'),# 노드 개수 64 ,activation은 parameter함수.
tf.keras.layers.Dense(128, activation='tanh'), # 노드개수 128
tf.keras.layers.Dense(1, activation='sigmoid')  # 결과물은 0 또는 1 , 결과는 0~1이어야 하기때문에, sigmoid

])
# 모델 만들기 sequential을 쓰면 신경망 레이어들 쉽게 만들어줌.

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#optimizer : learning rate조정.
#binary_crossentropy 를 loss함수로 사용 왜냐, 확률문제이기 때문입니다.

model.fit(np.array(xdata), np.array(ydata), epochs=300 ) # epoch = 학습횟수
# x = sat schoolgrade rank
# y = admit




#예측
predict_val =model.predict( [ [300, 3.70, 3], [100, 2.2, 1] ] )

for i in range(len(predict_val)):
    if (predict_val[i]<0.5 ) :
        print(i,"번째 입력자님은 불합격 가능성이 높습니다.\n 합격 가능성 :", predict_val[i])
    elif (0.5< predict_val[i]<0.7):
        print(i, "번째 입력자님은 소신 지원입니다.\n 합격 가능성 :", predict_val[i])
    else :
        print(i, "번째 입력자님은 안정 지원입니다.\n 합격 가능성 :", predict_val[i])



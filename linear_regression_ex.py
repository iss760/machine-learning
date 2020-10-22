import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


# sample data
x = np.array([845138, 811812, 730037, 638521, 614598])
y = np.array([3356, 2299, 2622, 2816, 2068])

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

# optimizer SGD 사용
# SGD(Stochastic Gradient Descent): 미니배치 사이즈만큼 일부 데이터만 계산해가며, 빠르게 학습
sgd = optimizers.SGD(lr=0.01)

# model compile
model.compile(optimizer=sgd, loss='mse', metrics=['mse'])   # loss function = MSE(Mean Squared Error)

# model train
model.fit(x, y, batch_size=1, epochs=100, shuffle=False)

# 예측값: 파란색 실선, 실제값: 검정색 점
plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()

print(model.predict([9.5]))

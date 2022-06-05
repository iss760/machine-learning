import wandb
from wandb.keras import WandbCallback
import tensorflow as tf


# wandb 설정
wandb.init(project='wandb-tutorial',
           config={
               'layer_1_activation': 'relu',
               'layer_1': 128,
               'learning_rate': 0.01,
               'dropout_rate': 0.2
           })
config = wandb.config

# load mnist datasets
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

# datasets 확인
print(train_x.shape, train_y.shape)
print(test_x.shape, test_y.shape)

# 데이터 정규화
train_x, test_x = train_x/255.0, test_x/255.0

# Sequential 모델 생성
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(config.layer_1, activation=config.layer_1_activation),
    tf.keras.layers.Dropout(config.dropout_rate),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 모델 학습
model.fit(train_x, train_y, epochs=5, callbacks=[WandbCallback()])

# 모델 평가
model.evaluate(test_x, test_y)


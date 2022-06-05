import tensorflow as tf
import wandb


# 학습 함수
def train():
    # config 기본 값
    config_defaults = {
        'layer1_size': 128,
        'dropout_rate': 0.2,
        'layer1_activation': 'relu',
        'optimizer': 'adam',
        'learning_rate': 0.01
    }
    # wandb 설정
    wandb.init(project='sweep-practice',
               config=config_defaults,
               magic=True)
    config = wandb.config

    # Load MNIST fashion datasets
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    print(train_images.shape)

    # 데이터 정규화
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Sequential 모델 생성
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(config.layer1_size,
                              activation=config.layer1_activation),
        tf.keras.layers.Dropout(config.dropout_rate),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    # Optimizer 설정
    if config.optimizer == 'rmsprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=config.learning_rate)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    # 모델 컴파일
    model.compile(optimizer=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # 모델 학습
    model.fit(train_images, train_labels, epochs=5,
              validation_data=(test_images, test_labels))


# 하이퍼파라미터 범위
sweep_config = {
    'method': 'grid',
    'parameters': {
        'layer1_size': {
            'values': [32, 64, 96, 128, 256]
        },
        'layer_activation': {
            'values': ['relu', 'sigmoid']
        },
        'dropout_rate': {
            'values': [0.1, 0.2, 0.3, 0.4, 0.5]
        },
        'learning_rate': {
            'values': [0.1, 0.01, 0.001]
        }
    }
}

# sweep id 설정
sweep_id = wandb.sweep(sweep_config, project='sweep-practice')

# 에이전트 실행
wandb.agent(sweep_id, function=train)

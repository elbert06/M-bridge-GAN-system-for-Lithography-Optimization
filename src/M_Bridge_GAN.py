import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df=pd.read_pickle("data/WM811K.pkl")

#list the field name of the structure
df.info()
def normalize_label(x):
    # 리스트, 튜플, 넘파이 배열이면 첫 번째 요소만 사용
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        return x[0]
    return x

# trainTestLabel 값을 전부 정규화
df['trainTestLabel'] = df['trainTestLabel'].apply(normalize_label)
#
# Select training and test data
trainIdx=df[df['trainTestLabel']=='Training'].index
testIdx=df[df['trainTestLabel']=='Test'].index

#show each failure type
trainFailureType=df.loc[trainIdx,'failureType']
testFailureType=df.loc[testIdx,'failureType']
uniqueType=df.loc[trainIdx,'failureType'].unique()
uniqueType.sort()

labelling = {}
for i,type in enumerate(uniqueType):
    labelling[type] = i
#Plot a wafer map
idx=trainFailureType[trainFailureType==uniqueType[0]].index
exampleIdx=idx[1]
# plt.imshow(df.iloc[exampleIdx]['waferMap'],)
# plt.show()
x_train = df.loc[trainIdx]['waferMap']
y_train = df.loc[trainIdx]['failureType'].map(labelling)
x_test = df.loc[testIdx]['waferMap']
y_test = df.loc[testIdx]['failureType'].map(labelling)


# 1. 목표 크기 설정 (데이터 중 넉넉하게 큰 사이즈로 잡음)
# WM-811K에서 보통 52~60 정도가 최대치 근처입니다. 넉넉히 60으로 잡죠.
import cv2

TARGET_H, TARGET_W = 60, 60

x_train_list = []
x_test_list = []
for wmap in x_train:
    h, w = wmap.shape
    x_train_list.append(cv2.resize(wmap, (TARGET_H, TARGET_W),interpolation=cv2.INTER_NEAREST))
for wmap in x_test:
    h, w = wmap.shape
    x_test_list.append(cv2.resize(wmap, (TARGET_H, TARGET_W),interpolation=cv2.INTER_NEAREST))


# 변환 완료
x_train_final = np.array(x_train_list).reshape(-1, TARGET_H, TARGET_W, 1)-1
y_train_final = np.array(y_train.tolist())
print(y_train_final)
# 모델 입력 사이즈도 (60, 60, 1)로 수정해야 함!
x_test_final = np.array(x_test_list).reshape(-1, TARGET_H, TARGET_W, 1)-1
y_test_final = np.array(y_test.tolist())
print("패딩 완료 shape:", x_train_final.shape)

#기존 모델 (basic)
import tensorflow as tf
import keras
from keras import layers
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
batch_size = 32
train_ds = tf.data.Dataset.from_tensor_slices((x_train_final, y_train_final)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test_final, y_test_final)).batch(batch_size)

model = keras.Sequential()
model.add(layers.Input(shape=(60, 60, 1)))
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(len(uniqueType), activation='softmax'))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#gan - adversarial - model
image_Input = keras.Input([100,])
label_Input = keras.Input(shape=(1,), dtype='int32')
mading_Input = keras.Input(shape=(60, 60, 1))

y = layers.Embedding(len(uniqueType), 50)(label_Input)   # (None, 1, 50)
y = layers.Flatten()(y)                              # (None, 50)
y = layers.Dense(15*15*128)(y)# 1. 입력: 100차원 노이즈 벡터
# Dense 층으로 기초 해상도(7x7)를 만듭니다. (목표가 28x28일 경우)
gen_layer = layers.Dense(15 * 15 * 128, use_bias=False)(image_Input)
gen_layer = layers.Concatenate()([gen_layer,y])
gen_layer = layers.BatchNormalization()(gen_layer)
gen_layer = (layers.LeakyReLU())(gen_layer)
gen_layer = (layers.Reshape((15, 15, 256)))(gen_layer)
# 2. 업샘플링 (이미지 크기 키우기): 15x15 -> 30x30
gen_layer = (layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))(gen_layer)
gen_layer = (layers.BatchNormalization())(gen_layer)
gen_layer = (layers.LeakyReLU())(gen_layer)
gen_layer = (layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))(gen_layer)
gen_layer = (layers.BatchNormalization())(gen_layer)
gen_layer = (layers.LeakyReLU())(gen_layer)
gen_output = (layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))(gen_layer)# 4. 이제 돌리세요(gen_layer)
generator = keras.models.Model([image_Input,label_Input],gen_output,name="generator")


x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(mading_Input)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)
# 출력: 1개의 값 (진짜일 확률)
feat_D = Flatten(name="feat_D")(x)
fakeornot = layers.Dense(1,activation='sigmoid')(feat_D)
discriminator = keras.models.Model(mading_Input,[fakeornot,feat_D])

x = layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same')(mading_Input)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)
x = layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')(x)
x = layers.LeakyReLU()(x)
x = layers.Dropout(0.3)(x)
# 출력: 1개의 값 (진짜일 확률)
feat_M = layers.Flatten(name="feat_M")(x)
labeled = layers.Dense(len(uniqueType),activation='softmax')(feat_M)
mediator = keras.models.Model(mading_Input,[labeled,feat_M])
mediator.summary()
discriminator.summary()
generator.summary()

dummy = tf.zeros((1, 60, 60, 1))
_, dummy_D_feat = discriminator(dummy)
_, dummy_M_feat = mediator(dummy)
D_dim = dummy_D_feat.shape[-1]
M_dim = dummy_M_feat.shape[-1]

feat_D_input = keras.Input(shape=(D_dim,), name="feat_D_input")
h = layers.Dense(64, activation='relu')(feat_D_input)   # 선택 사항
pred_label_from_D = layers.Dense(len(uniqueType), activation='softmax')(h)
M_head_on_D = keras.Model(feat_D_input, pred_label_from_D, name="M_head_on_D")

feat_M_input = keras.Input(shape=(M_dim,), name="feat_M_input")
h = layers.Dense(64, activation='relu')(feat_M_input)   # 선택 사항, 안 써도 됨
pred_rf_from_M = layers.Dense(1, activation='sigmoid')(h)
D_head_on_M = keras.Model(feat_M_input, pred_rf_from_M, name="D_head_on_M")


opt_G = tf.keras.optimizers.Adam(0.0002, 0.5)
opt_D = tf.keras.optimizers.Adam(0.0002, 0.5)
opt_M = tf.keras.optimizers.Adam(0.0002, 0.5)
opt_D_head = tf.keras.optimizers.Adam(0.0002, 0.5)
opt_M_head = tf.keras.optimizers.Adam(0.0002, 0.5)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
@tf.function
def train_step(real_x, real_y, noise_dim=100):
    BCE = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    
    # 헷갈리게 만들기 위한 정답지 (Binary: 0.5, Multi-class: 1/N)
    # 진짜인지 가짜인지 모르게 하려면 0.5가 목표값이어야 함
    target_confusion_binary = 0.5 * tf.ones([tf.shape(real_x)[0], 1])
    
    # 클래스를 모르게 하려면 균등 분포가 목표값 (예: [0.1, 0.1, ..., 0.1])
    n_classes = len(uniqueType)
    target_confusion_class = tf.ones([tf.shape(real_x)[0], n_classes]) / float(n_classes)

    batch_size = tf.shape(real_x)[0]
    z = tf.random.normal((batch_size, noise_dim))
    y_fake = tf.expand_dims(real_y, 1)

    with tf.GradientTape(persistent=True) as tape:
        # 1. Forward Pass (모든 모델 실행)
        fake_x = generator([z, y_fake], training=True)
        
        d_real_score, d_real_feat = discriminator(real_x, training=True)
        d_fake_score, d_fake_feat = discriminator(fake_x, training=True)
        
        m_real_label, m_real_feat = mediator(real_x, training=True)
        # m_fake_label, m_fake_feat = mediator(fake_x, training=True) # G 학습용

        # 2. Cross-Adversarial Heads 실행
        # M의 특징을 보고 D가 맞추려 함 (D_head가 M을 감시)
        pred_rf_from_M = D_head_on_M(m_real_feat, training=True) 
        # D의 특징을 보고 M이 라벨을 맞추려 함 (M_head가 D를 감시)
        pred_label_from_D = M_head_on_D(d_real_feat, training=True)

        # -----------------------------------------------------------
        # 3. Loss 계산
        # -----------------------------------------------------------
        
        # (A) Head들의 Loss (얘네는 정답을 맞춰야 함!)
        # D_head는 M이 진짜인지 가짜인지 맞춰야 함 -> 여기서는 real_x니까 1(True)이 정답
        loss_head_D_on_M = BCE(tf.ones_like(pred_rf_from_M), pred_rf_from_M)
        # M_head는 D의 특징으로 라벨을 맞춰야 함
        loss_head_M_on_D = SCE(real_y, pred_label_from_D)
        
        # (B) Main 모델들의 Loss (본업 + Head 속이기)
        
        # D Main: 진짜/가짜 구분
        loss_D_main = BCE(tf.ones_like(d_real_score), d_real_score) + BCE(tf.zeros_like(d_fake_score), d_fake_score)
        
        # D Adversarial: M_head(라벨 맞추기)를 헷갈리게 해야 함 (Entropy Maximization)
        # 방법: M_head의 예측 결과와 '균등 분포(찍기)' 사이의 거리를 줄임 -> + 부호 사용!
        loss_D_confuse = tf.keras.losses.CategoricalCrossentropy()(target_confusion_class, pred_label_from_D)
        
        loss_D_total = loss_D_main + 0.1 * loss_D_confuse  # 더하기(+)로 변경

        # M Main: 라벨 분류
        loss_M_main = SCE(real_y, m_real_label)
        
        # M Adversarial: D_head(진짜/가짜)를 헷갈리게 해야 함
        # 방법: D_head의 예측 결과가 0.5(모르겠다)가 되도록 유도 -> + 부호 사용!
        loss_M_confuse = BCE(target_confusion_binary, pred_rf_from_M)
        
        loss_M_total = loss_M_main + 0.1 * loss_M_confuse # 더하기(+)로 변경

        # G Loss (기존 유지)
        m_fake_label, _ = mediator(fake_x, training=True) # 위에서 계산 안했으면 여기서
        loss_G_adv = BCE(tf.ones_like(d_fake_score), d_fake_score)
        loss_G_sem = SCE(real_y, m_fake_label)
        loss_G_total = loss_G_adv + loss_G_sem

    # -----------------------------------------------------------
    # 4. Gradient 적용 (분리 적용이 핵심!)
    # -----------------------------------------------------------
    
    # [Step 1] Head들 업데이트 (똑똑해져라)
    # D_head_on_M만 업데이트
    grads_head_D = tape.gradient(loss_head_D_on_M, D_head_on_M.trainable_variables)
    opt_D_head.apply_gradients(zip(grads_head_D, D_head_on_M.trainable_variables))
    
    # M_head_on_D만 업데이트
    grads_head_M = tape.gradient(loss_head_M_on_D, M_head_on_D.trainable_variables)
    opt_M_head.apply_gradients(zip(grads_head_M, M_head_on_D.trainable_variables))

    # [Step 2] Main 모델들 업데이트 (본업 잘하고 + Head 헷갈리게 해라)
    # Discriminator 업데이트 (D_head 변수는 제외!)
    grads_D = tape.gradient(loss_D_total, discriminator.trainable_variables)
    opt_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

    # Mediator 업데이트 (M_head 변수는 제외!)
    grads_M = tape.gradient(loss_M_total, mediator.trainable_variables)
    opt_M.apply_gradients(zip(grads_M, mediator.trainable_variables))

    # Generator 업데이트
    grads_G = tape.gradient(loss_G_total, generator.trainable_variables)
    opt_G.apply_gradients(zip(grads_G, generator.trainable_variables))
    
    # tape 리소스 해제 (persistent=True 썼으므로)
    del tape

    return loss_D_total, loss_M_total, loss_G_total
@tf.function
def test_step(real_x, real_y):
    # Loss 함수들은 밖에서 정의된 걸 써도 되고 안에서 정의해도 됩니다.
    SCE = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    m_real_label, m_real_feat = mediator(real_x, training=False)
    # M Total Loss
    loss_M = SCE(real_y, m_real_label)
    return loss_M
EPOCHS = 5
history = {'Adversarial mediator Loss': [],'basic model Loss': []}
for epoch in range(EPOCHS):
    D_loss_epoch = 0.0
    M_loss_epoch = 0.0
    G_loss_epoch = 0.0
    n_batches = 0

    for real_x, real_y in train_ds:
        d_loss, m_loss, g_loss = train_step(real_x, real_y)
        D_loss_epoch += d_loss
        M_loss_epoch += m_loss
        G_loss_epoch += g_loss
        n_batches += 1
        model.train_on_batch(real_x, real_y)
    D_loss_epoch /= n_batches
    M_loss_epoch /= n_batches
    G_loss_epoch /= n_batches

    print(f"Epoch {epoch+1}: "
          f"D={float(D_loss_epoch):.4f}, "
          f"M={float(M_loss_epoch):.4f}, "
          f"G={float(G_loss_epoch):.4f}")
    basic_mode_loss=0.0
    Adversarial_mediator_loss=0.0
    f = 0
    for x_test,y_test in test_ds:
        basic_mode_loss += float(model.test_on_batch(x_test, y_test)[0])
        Adversarial_mediator_loss += float(test_step(x_test, y_test))
        f += 1
    basic_mode_loss /= f
    Adversarial_mediator_loss /= f
    history['basic model Loss'].append(basic_mode_loss) 
    history['Adversarial mediator Loss'].append(Adversarial_mediator_loss)
    print(f" Test Loss - basic model: {basic_mode_loss:.4f}, Adversarial mediator: {Adversarial_mediator_loss:.4f}")
# plt는 basic model 과의 비교 목적    
plt.plot(range(1,len(history['basic model Loss'])+1),history['basic model Loss'], label='basic model Loss')
plt.plot(range(1,len(history['Adversarial mediator Loss'])+1),history['Adversarial mediator Loss'], label='Adversarial mediator Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

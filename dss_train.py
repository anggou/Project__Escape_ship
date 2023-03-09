import copy
import pylab
import random
import numpy as np
# from dss_environment import Env
from small_env import Env
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 딥살사 인공신경망
class DeepSARSA(tf.keras.Model):  # 6
    def __init__(self, action_size):  # 6
        super(DeepSARSA, self).__init__()
        self.fc1 = Dense(30, activation='relu')
        self.fc2 = Dense(30, activation='relu')
        self.fc_out = Dense(action_size)

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        q = self.fc_out(x)
        return q


# 그리드월드 예제에서의 딥살사 에이전트
class DeepSARSAgent:
    def __init__(self, state_size, action_size):  # 12, 6
        # 상태의 크기와 행동의 크기 정의
        self.state_size = state_size
        self.action_size = action_size

        # 딥살사 하이퍼 파라메터
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.
        self.epsilon_decay = .9999
        self.epsilon_min = 0.01
        self.model = DeepSARSA(self.action_size)  # 출력이 6개를 지정
        self.optimizer = Adam(learning_rate=self.learning_rate)

    # 입실론 탐욕 정책으로 행동 선택
    def get_action(self, state):
        if np.random.rand() <= self.epsilon: #0~1 사이 실수 입실론 값이 작아질수록 탐험 적게
            return random.randrange(self.action_size)
        else:
            q_values = self.model(state)  # ANN 사용해서 Q테이블 불러오기
            # print(q_values[0])
            return np.argmax(q_values[0])

    # <s, a, r, s', a'>의 샘플로부터 모델 업데이트
    def train_model(self, state, action, reward, next_state, next_action, done):
        if self.epsilon > self.epsilon_min:   # 스텝이 진행될수록,
            self.epsilon *= self.epsilon_decay

        # 학습 파라메터
        model_params = self.model.trainable_variables  ##가중치
        with tf.GradientTape() as tape:
            tape.watch(model_params)
            predict = self.model(state)[0]  # state=12개 , 출력은 q-value 6개
            one_hot_action = tf.one_hot([action], self.action_size)  # action_size = 6, 액션에 해당하는 곳만 1 [010000]
            predict = tf.reduce_sum(one_hot_action * predict, axis=1) # q 함수 핫코딩 형식 [040000] > 4 곱한뒤 더함

            # done = True 일 경우 에피소드가 끝나서 다음 상태가 없음
            next_q = self.model(next_state)[0][next_action] #  값 하나,
            target = reward + (1 - done) * self.discount_factor * next_q
            # print(target)
            # MSE 오류 함수 계산
            loss = tf.reduce_mean(tf.square(target - predict))
        #오류함수를 구할때, 강화학습알고리즘을 사용
        # 오류함수를 줄이는 방향으로 모델 업데이트
        grads = tape.gradient(loss, model_params)
        self.optimizer.apply_gradients(zip(grads, model_params))


if __name__ == "__main__":
    # 환경과 에이전트 생성
    env = Env(render_speed=0.1) # 0.001
    state_size = 12
    action_space = [0, 1, 2, 3, 4, 5]
    action_size = len(action_space)
    agent = DeepSARSAgent(state_size, action_size) #12,
    scores, episodes = [], []

    EPISODES = 1000
    for e in range(EPISODES):
        done = False
        score = 1
        # env 초기화
        state = env.reset()  # 12개 변수
        state = np.reshape(state, [1, state_size])  # state_size = 12개

        while not done:
            # 현재 상태에 대한 행동 선택
            action = agent.get_action(state)  # 모델을 통해 q 함수 최댓값 가져옴

            # 선택한 행동으로 환경에서 한 타임스텝 진행 후 샘플 수집
            next_state, reward, done = env.step(action)  # 여기서 state
            next_state = np.reshape(next_state, [1, state_size])
            next_action = agent.get_action(next_state)  # ANN을 통한 q함수 출력

            # 샘플로 모델 학습

            agent.train_model(state, action, reward, next_state, next_action, done) # 12,1,1,12,1,1
            if done == 1 : # 불만났을때 추가
                score += reward
            state = next_state

            if done:
                # 에피소드마다 학습 결과 출력
                print("episode: {:3d} | score: {:3d} | epsilon: {:.3f}".format(
                    e, score, agent.epsilon))

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.xlabel("episode")
                pylab.ylabel("score")
                pylab.savefig("../Escape_ship/save_graph/graph.png")

        # 100 에피소드마다 모델 저장
        if e % 1 == 0:
            agent.model.save_weights('save_model/model', save_format='tf')

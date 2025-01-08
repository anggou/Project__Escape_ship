import numpy as np

class PPOAgent:
    def __init__(self, state_size, action_size, actor_model, critic_model):
        self.state_size = state_size
        self.action_size = action_size
        self.actor_model = actor_model  # 정책 네트워크
        self.critic_model = critic_model  # 가치 함수 네트워크
        self.gamma = 0.99
        self.lam = 0.95
        self.epsilon = 0.2
        self.epochs = 10
        self.batch_size = 64

    def get_action(self, state):
        probs = self.actor_model.predict(state)
        action = np.random.choice(self.action_size, p=probs[0])  # 정책에 따라 행동 선택
        return action, probs[0][action]

    def compute_advantages(self, rewards, dones, values):
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * (1 - dones[t]) * values[t + 1] - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return np.array(advantages)

    def train(self, states, actions, advantages, rewards, old_probs):
        for _ in range(self.epochs):
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            for i in range(0, len(states), self.batch_size):
                idx = indices[i:i + self.batch_size]
                s_batch = states[idx]
                a_batch = actions[idx]
                adv_batch = advantages[idx]
                r_batch = rewards[idx]
                old_p_batch = old_probs[idx]

                # Update actor (policy network)
                with tf.GradientTape() as tape:
                    probs = self.actor_model(s_batch)
                    entropy = -np.sum(probs * np.log(probs + 1e-10), axis=-1)
                    ratio = tf.exp(tf.math.log(probs + 1e-10) - tf.math.log(old_p_batch + 1e-10))
                    clipped_ratio = tf.clip_by_value(ratio, 1 - self.epsilon, 1 + self.epsilon)
                    policy_loss = -tf.reduce_mean(tf.minimum(ratio * adv_batch, clipped_ratio * adv_batch) + 0.01 * entropy)
                actor_grads = tape.gradient(policy_loss, self.actor_model.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

                # Update critic (value network)
                with tf.GradientTape() as tape:
                    values = self.critic_model(s_batch)
                    critic_loss = tf.reduce_mean((r_batch - values) ** 2)
                critic_grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))


# 환경과 에이전트 초기화
state_size = env.state_size
action_size = env.action_size
agent = PPOAgent(state_size, action_size, actor_model, critic_model)

for episode in range(total_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    score = 0

    states, actions, rewards, dones, old_probs = [], [], [], [], []
    while not done:
        # 행동 선택
        action, old_prob = agent.get_action(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])

        # 경험 저장
        states.append(state[0])
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_probs.append(old_prob)

        state = next_state
        score += reward

    # 에피소드 종료 후 학습
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards)
    dones = np.array(dones)
    old_probs = np.array(old_probs)

    # 가치 함수 예측
    values = agent.critic_model.predict(states)
    values = np.append(values, [0])  # 마지막 값 추가

    # Advantage 계산
    advantages = agent.compute_advantages(rewards, dones, values)
    discounted_rewards = advantages + values[:-1]

    # 모델 학습
    agent.train(states, actions, advantages, discounted_rewards, old_probs)
    print(f"Episode: {episode}, Score: {score}")

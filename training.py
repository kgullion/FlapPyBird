import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

class DQN:
    def __init__(self):
        self.memory  = deque(maxlen=2000)
        self.gamma = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.1
        self.tau = .125
        self.model = self.create_model()

    def create_model(self):
        model   = Sequential()
        state_shape = 3
        model.add(Dense(8, input_dim=state_shape, activation="relu"))
        # model.add(Dense(48, activation="relu"))
        model.add(Dense(4, activation="relu"))
        model.add(Dense(2))
        model.compile(loss="mean_squared_error",
            optimizer=Adam(lr=self.learning_rate))
        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        if np.random.random() < self.epsilon:
            return int(random.randint(0,9)==0)
        actions = self.model.predict(state)[0]
        print(actions)
        return np.argmax(actions)

    def remember(self, state, action, reward, new_state):
        self.memory.append([state, action, reward, new_state])

    def replay(self):
        batch_size = 256
        if len(self.memory) < batch_size: 
            samples = self.memory
        else:
            samples = random.sample(self.memory, batch_size)
        states, actions, rewards, new_states = zip(*samples)
        states = np.concatenate(states)
        new_states = np.concatenate(new_states)
        targets = self.model.predict(states)
        futures = np.max(self.model.predict(new_states), axis=1)
        learned = rewards + self.gamma*futures
        indices = range(len(actions))
        choosen = targets[indices, actions]
        choosen = (1 - self.learning_rate)*choosen + self.learning_rate*learned
        targets[indices,actions] = choosen
        self.model.fit(states, targets, epochs=1, verbose=0)

    def save_model(self, fn):
        self.model.save(fn)

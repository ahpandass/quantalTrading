# -*- coding: utf-8 -*-

import logging

import click

import numpy as np
import pandas as pd
import keras
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tradingENV_synsmpl_tracking_btc as tradingENV_synsmpl
import os
from datetime import datetime

PLOT_AFTER_ROUND = 1

log = logging.getLogger(__name__)
logging.basicConfig()
log.setLevel(logging.INFO)
log.info('%s logger started.', __name__)
# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size, model_path = False):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24
        
        self.states, self.actions, self.rewards = [], [], []
        # create model for policy network        
        if model_path:
            self.model = keras.models.load_model(model_path)
            print('Model loaded')            
        # lists for the states, actions and rewards
        else:
            self.model = self.build_model()
            print('Model built')
    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=self.state_size))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(self.hidden1,  activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a).
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=self.learning_rate))
        return model
    # using the output of policy network, pick action stochastically
    #def get_action(self, state):
        #policy = self.model.predict(state, batch_size=1).flatten()
        #random choice return size length (1) from action_size, and the random pr rate is follow array p. there is another arg replace=True/False, to identify if 1 number could be picked multiple times
        #return np.random.choice(self.action_size, 1, p=policy)[0] 
        
    def get_action(self, state,batch_size,action_size):
        policy = self.model.predict(state, batch_size=batch_size).flatten()
        #random choice return size length (1) from action_size, and the random pr rate is follow array p. there is another arg replace=True/False, to identify if 1 number could be picked multiple times
        self.actions =[]
        for (i,pr) in enumerate(policy):
          if i % self.action_size == 0:
            action_pr_ls = []
            for j in range(action_size):
              action_pr_ls.append(policy[i+j])
            action_pr_ls /= np.sum(action_pr_ls)
            self.actions.append(np.random.choice(self.action_size, 1, p=action_pr_ls)[0])
        return self.actions
    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards
    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        training_size = list(self.state_size)
        training_size.insert(0,episode_length)
        update_inputs = np.zeros(tuple(training_size))
        advantages = np.zeros((episode_length, self.action_size))
        for i in range(episode_length):
            update_inputs[i] = np.array(self.states[i]).reshape(self.state_size)
            advantages[i][self.actions[i]] = discounted_rewards[i]
        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []
    def save_model(self, fn):
        self.model.save(fn)

@click.command()
@click.option(
    '-s',
    '--symbol',
    default='BNBUSDT',
    show_default=True,
    help='given stock code ',
)
@click.option(
    '-b',
    '--begin',
    default='2021-05-01',
    show_default=True,
    help='The begin date of the train.',
)

@click.option(
    '-e',
    '--end',
    default='2022-02-20',
    show_default=True,
    help='The end date of the train.',
)

@click.option(
    '-d',
    '--interval',
    default='30m',
    help='train interval',
)

@click.option(
    '-t',
    '--train_round',
    type=int,
    default=1000000,
    help='train round',
)

@click.option(
    '-m',
    '--model_path',
    default='.',
    show_default=True,
    help='trained model save path.',
)

@click.option(
    '-m',
    '--freq_cnt',
    type=int,
    default=100,
    show_default=True,
    help='Sample duration',
)

@click.option(
    '-m',
    '--window',
    type=int,
    default=20,
    show_default=True,
    help='tracking last ? minutes steps',
)

def execute(symbol, begin, end, interval, train_round,  model_path, freq_cnt, window):
    # In case of CartPole-v1, you can play until 500 time step
    env = tradingENV_synsmpl.TradingEnv(symbol=symbol, interval=interval, start=begin, end=end, freq_cnt=freq_cnt, window = window)
    EPISODES = train_round
    # get size of state and action from environment
    state_size = (window,env.observation_space,1)
    action_size = len(env.action_space)

    # make REINFORCE agent
    model_name = os.path.join(model_path, env.src.symbol + ".model")
    #agent = REINFORCEAgent(state_size, action_size, model_name)
    agent = REINFORCEAgent(state_size, action_size)

    scores, episodes = [], []
    simrors = np.zeros(EPISODES)
    mktrors = np.zeros(EPISODES)
    victory = False
    for episode in range(EPISODES):
        if victory:
            break
        done = False
        state = env.reset()
        #state = np.reshape(state, [1, state_size])
        step_ls=[]
        while not done:            
            step_ls.append(state)
            next_state, done = env._step()
            state = next_state
            if done:
                #prepare the feature data:
                feature_ls = []
                for i,step in enumerate(step_ls):
                    if (i < (window)) or (i+1  == len(step_ls)):
                        continue
                    feature = step_ls[i-int(window):i]
                    feature_ls.append(feature)
                
                #feature_ls = prepare_feature(step_ls, window)
                #print('Done, at {}'.format(datetime.now()))

                action_ls = agent.get_action(feature_ls[:-1],len(feature_ls),action_size)
                agent.rewards = env._rewards_ls(feature_ls,action_ls)
                agent.states = feature_ls[:-1]                
                df = env.sim.to_df()
                simrors[episode] = df.bod_nav.values[-1] - 1  # compound returns
                mktrors[episode] = df.mkt_nav.values[-1] - 1
                if episode % 100 == 0:
                    log.info('year #%6d, sim ret: %8.4f, mkt ret: %8.4f, net: %8.4f', episode,
                             simrors[episode], mktrors[episode], simrors[episode] - mktrors[episode])
                    log.info("100 episode Completed in %d trials , save it as %s", episode,
                            os.path.join(model_path, env.src.symbol + ".model"))
                    agent.save_model(os.path.join(model_path, env.src.symbol + ".model"))
                    if episode > 100:
                        vict = pd.DataFrame({'sim': simrors[episode - 100:episode],
                                             'mkt': mktrors[episode - 100:episode]})
                        vict['net'] = vict.sim - vict.mkt
                        log.info('vict:%f', vict.net.mean())
                        if vict.net.mean() > 0.2:
                            victory = True
                            log.info('Congratulations, Warren Buffet!  You won the trading game ', )
                            break
                # every episode, agent learns from sample returns
                agent.train_model()
                episodes.append(episode)

    
    log.info("Completed in %d trials , save it as %s", episode,
             os.path.join(model_path, env.src.symbol + ".model"))
    agent.save_model(os.path.join(model_path, env.src.symbol + ".model"))


if __name__ == "__main__":
    execute()

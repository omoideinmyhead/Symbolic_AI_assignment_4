#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic Programming 
Practical for course 'Symbolic AI'
2020, Leiden University, The Netherlands
By Thomas Moerland
"""

import numpy as np
from world import World

class Dynamic_Programming:

    def __init__(self):
        self.V_s = None # will store a potential value solution table
        self.Q_sa = None # will store a potential action-value solution table
        
    def value_iteration(self, env, gamma = 1.0, theta=0.001):
        ''' Executes value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Value Iteration (VI)")
        # initialize value table
        V_s = np.zeros(env.n_states)
    
        ## IMPLEMENT YOUR VALUE ITERATION ALGORITHM HERE
        delta = 1
        while delta > theta:
            delta = 0
            for j in range(env.n_states):
                former_value = V_s[j]
                state_up, r_up = env.transition_function(j,'up')
                state_down, r_down = env.transition_function(j,'down')
                state_left, r_left = env.transition_function(j,'left')
                state_right, r_right = env.transition_function(j,'right')
                V_s[j] = max(r_up + V_s[state_up], r_down + V_s[state_down], r_left + V_s[state_left], r_right+V_s[state_right])
                delta = max(delta, abs(former_value - V_s[j]))
                print(delta)
        self.V_s = V_s
        return

    def Q_value_iteration(self, env,
                          gamma = 1.0, theta=0.001):
        ''' Executes Q-value iteration on env. 
        gamma is the discount factor of the MDP
        theta is the acceptance threshold for convergence '''

        print("Starting Q-value Iteration (QI)")
        # initialize state-action value table
        Q_sa = np.zeros([env.n_states,env.n_actions])

        ## IMPLEMENT YOUR Q-VALUE ITERATION ALGORITHM HERE
        delta = 1
        while delta > theta:
            delta = 0
            for s in range(env.n_states):
                for a in range(env.n_actions):
                    former_value = Q_sa[s][a]
                    #this list is a trick to translate the numbers into action strings
                    actions = ['up', 'down', 'left', 'right']
                    #whatever the action is finds the reward and new state
                    state, r = env.transition_function(s, actions[a])
                    #applies the Q-iteration
                    Q_sa[s][a] = r+gamma*max([Q_sa[state][i] for i in range(4)])
                    delta = max(delta, abs(former_value - Q_sa[s][a]))
                    print(delta)
        self.Q_sa = Q_sa
        return
                
    def execute_policy(self,env,table='V'):
        ## Execute the greedy action, starting from the initial state
        env.reset_agent()
        print("Start executing. Current map:") 
        env.print_map()
        while not env.terminal:
            current_state = env.get_current_state() # this is the current state of the environment, from which you will act
            available_actions = env.actions
            # Compute action values
            if table == 'V' and self.V_s is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.V_s HERE !!!
                action_values = np.zeros(len(available_actions))
                i = -1
                for action in available_actions:
                    i += 1
                    next_state, reward = env.transition_function(current_state, action)
                    action_values[i] = reward + self.V_s[next_state]
                greedy_action_index = get_greedy_index(action_values)
                greedy_action = available_actions[greedy_action_index] # replace this!
            elif table == 'Q' and self.Q_sa is not None:
                ## IMPLEMENT ACTION VALUE ESTIMATION FROM self.Q_sa here !!!
                action_values = self.Q_sa[current_state]
                greedy_action_index = get_greedy_index(action_values)
                greedy_action = available_actions[greedy_action_index]
            else:
                print("No optimal value table was detected. Only manual execution possible.")
                greedy_action = None

            # ask the user what he/she wants
            while True:
                if greedy_action is not None:
                    print('Greedy action= {}'.format(greedy_action))    
                    your_choice = input('Choose an action by typing it in full, then hit enter. Just hit enter to execute the greedy action:')
                else:
                    your_choice = input('Choose an action by typing it in full, then hit enter. Available are {}'.format(env.actions))
                    
                if your_choice == "" and greedy_action is not None:
                    executed_action = greedy_action
                    env.act(executed_action)
                    break
                else:
                    try:
                        executed_action = your_choice
                        env.act(executed_action)
                        break
                    except:
                        print('{} is not a valid action. Available actions are {}. Try again'.format(your_choice,env.actions))
            print("Executed action: {}".format(executed_action))
            print("--------------------------------------\nNew map:")
            env.print_map()
        print("Found the goal! Exiting \n ...................................................................... ")
    

def get_greedy_index(action_values):
    ''' Own variant of np.argmax, since np.argmax only returns the first occurence of the max. 
    Optional to uses '''
    return np.where(action_values == np.max(action_values))
    
if __name__ == '__main__':
    env = World('prison.txt') 
    DP = Dynamic_Programming()

    # Run value iteration
    input('Press enter to run value iteration')
    optimal_V_s = DP.value_iteration(env)
    input('Press enter to start execution of optimal policy according to V')
    DP.execute_policy(env, table='V') # execute the optimal policy


    # Once again with Q-values:
    input('Press enter to run Q-value iteration')
    optimal_Q_sa = DP.Q_value_iteration(env)
    input('Press enter to start execution of optimal policy according to Q')
    DP.execute_policy(env, table='Q') # execute the optimal policy


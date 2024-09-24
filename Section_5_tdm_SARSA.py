import numpy as np
import time
import pygame
from random import randint
from maze import *

env = Maze()
state = env.reset()
env.render()

pygame.font.init()
font = pygame.font.SysFont('Times New Roman', 24)

q = {}
for row in range(5):
    for col in range(5):
        state = (row, col)
        for action in range(4):
            if state == (4,4):
                q[(state, action)] = 0
            else:
                q[(state, action)] = -20* np.random.rand()
print(q)

def pi(state, epsilon=0.1):
    if np.random.rand() < epsilon:
        action  = env.action_space.sample()
    else:
        action_values = {action: q[(state, action)] for action in range(4)}
        action = max(action_values, key=action_values.get)
    return action

def generateOneEpisode(wantToView=False, episode_num=1, epsilon=0.1, alpha=0.1, gamma=0.9):
    state = env.reset()
    action = pi(state, epsilon)
    done = False
    episode = []

    while not done:
        next_state, reward, done, _ = env.step(action)
        next_action = pi(next_state, epsilon)
        
        if (state, action) not in q:
            q[(state, action)] = 0

        if (next_state, next_action) not in q:
            q[(next_state, next_action)] = 0

        q[(state, action)] += alpha * (reward + gamma * q[(next_state, next_action)] - q[(state, action)])
        
        episode.append([state, action, reward])

        state = next_state
        action = next_action
        
        if wantToView:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            env.render()

            episode_text = font.render(f'Episode: {episode_num}', True, (255, 223, 0))
            env.screen.blit(episode_text, (10, 10))
            pygame.display.update()
            time.sleep(0.01)

    return episode

def sarsa (episodes=20, alpha=0.1, gamma=0.9, wantToView=False):
    for i in range(1, episodes + 1):
        env.current_episode = i
        generateOneEpisode(wantToView=wantToView, episode_num=i, epsilon=1/(1+i), alpha=alpha, gamma=gamma)
    
    print("Done")

sarsa(wantToView=True)

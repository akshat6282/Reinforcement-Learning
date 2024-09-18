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
            q[(state, action)] = randint(-10, 10)
print(q)

def pi(state,epsilon=0.9):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    else:
        action_values = {action: q[(state, action)] for action in range(4)}
        action = max(action_values, key=action_values.get)
        return action

def generateOneEpisode(wantToView=False,episode_num=1,epsilon = 0):
    episode = []
    state = env.reset() 
    done = False
    
    while not done:
        action = pi(state,epsilon=epsilon) 
        newState, reward, done, _ = env.step(action) 
        
        episode.append([state, action, reward])  
        state = newState 
        
        if (state, action) not in q:
            q[(state, action)] = 0  

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

def on_policy_mc_control(episodes=20, gamma=0.9, wantToView=False):
    Q = {} 
    
    for i in range(1, episodes + 1):
        env.current_episode = i
    
        episode = generateOneEpisode(wantToView=wantToView,episode_num=i,epsilon = 1/(1+i))
        
        G = 0 
        for step in range(len(episode) - 1, -1, -1): 
            state, action, reward = episode[step]
            G = reward + gamma * G 
            
            if (state, action) not in Q:
                Q[(state, action)] = [] 
            
            Q[(state, action)].append(G)
            
            q[(state, action)] = np.mean(Q[(state, action)])

    print("Training complete")

on_policy_mc_control(wantToView=True)
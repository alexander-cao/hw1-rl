import numpy as np
import matplotlib.pyplot as plt

"parameters"
K = 15
all_A_t = np.arange(1,6)
prob_A_t = 1 / len(all_A_t)
c_f = 100
c_h = 2
T = 500
max_people_in_station = 200
gamma = 0.95
num_actions = 2

"action a=0: don't dispatch shuttle"
"action a=1: dispatch shuttle"

def new_state(s, A_t, a):
    if a == 0:
        new = np.minimum(s + A_t, max_people_in_station)
    elif a == 1:
        new = np.minimum(np.maximum(0, s-K)+A_t, max_people_in_station)
    return new

def reward(s, a):
    if a == 0:
        r = -c_h * s
    elif a == 1:
        num_people_left_in_station = np.maximum(0, s - K)
        r = -c_f - c_h * num_people_left_in_station
    return r
    
V = np.zeros((T+1, max_people_in_station+1))
rhs = np.zeros(num_actions)
for t in range(T, -1, -1):
    for s in range(max_people_in_station+1):
        if t == T:
            for a in range(num_actions):
                rhs[a] = reward(s, a)
            V[t, s] = np.amax(rhs)
        else:
            for a in range(num_actions):
                E = 0
                for A_t in all_A_t:
                    E += prob_A_t * V[t+1, new_state(s, A_t, a)]
                rhs[a] = reward(s, a) + gamma*E
            V[t, s] = np.amax(rhs)
            
plt.figure(figsize = (8, 6))
plt.plot(V[0,:])
plt.xlabel('s = customers waiting', fontsize = 20)
plt.ylabel('V(t=0, s)', fontsize = 20)
plt.savefig('problem_1a-enumeration.png')
plt.close()

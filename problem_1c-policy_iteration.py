import numpy as np
import matplotlib.pyplot as plt

"parameters"
K = 15
all_A_t = np.arange(1,6)
prob_A_t = 1 / len(all_A_t)
c_f = 100
c_h = 2
max_people_in_station = 200
gamma = 0.95
num_actions = 2
tol = 1e-6

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

V_old = np.zeros(max_people_in_station+1)    
V_new = np.zeros(max_people_in_station+1)
pi_old = np.zeros(max_people_in_station+1)
pi_new = np.zeros(max_people_in_station+1)
rhs = np.zeros(num_actions)

counter = 0
pi_norm = 99
while pi_norm > tol:    
    V_norm = 99
    while V_norm > tol:
        for s in range(max_people_in_station+1):
            E = 0
            for A_t in all_A_t:
                E += prob_A_t * V_old[new_state(s, A_t, pi_old[s])]     
            V_new[s] = reward(s, pi_old[s]) + gamma*E
            
        V_norm = np.linalg.norm(V_old-V_new, ord=np.inf)
        V_old =  np.copy(V_new)
        
    for s in range(max_people_in_station+1):
        for a in range(num_actions):
            E = 0
            for A_t in all_A_t:
                E += prob_A_t * V_new[new_state(s, A_t, a)]   
            rhs[a] = reward(s, a) + gamma*E
        pi_new[s] = np.argmax(rhs) 
        
    pi_norm = np.linalg.norm(pi_old-pi_new, ord=2)
    pi_old = np.copy(pi_new)
    
    counter += 1
    print(counter, pi_norm)
    
plt.figure(figsize = (8, 6))
plt.plot(V_new)
plt.xlabel('s = customers waiting', fontsize = 20)
plt.ylabel('V(s)', fontsize = 20)
plt.show()

plt.figure(figsize = (8, 6))
plt.plot(pi_new)
plt.xlabel('s = customers waiting', fontsize = 20)
plt.ylabel('a = $\pi$(s)', fontsize = 20)
plt.savefig('problem_1c-policy_iteration.png')
plt.close()

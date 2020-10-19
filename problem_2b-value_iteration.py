"i simplified the problem by assuming the following:"
"the policy should be such that the most expensive customers (type 5, then 4, etc.) should get on the shuttle first"
"that way we don't have to worry about the order of customers/combinatorials in the state representation"
"doing so would mean the state is exponentially large and thus intractable to solve with these methods"

"all that being said, this code still takes infeasibly long to run"

import numpy as np
import matplotlib.pyplot as plt

"parameters"
K = 30
c_f = 100
c_h = np.asarray([1.0, 1.5, 2.0, 2.5, 3.0])
Ai_t_range = np.arange(1,6)
prob_Ai_t = 1.0 / (len(Ai_t_range)**len(c_h))
max_customer_type_in_station = 100
gamma = 0.95
num_actions = 2
num_types_customers = len(c_h)
tol = 1e-6

"action a=0: don't dispatch shuttle"
"action a=1: dispatch shuttle"

def after_shuttle(s):
    leftover = K
    for i in range(num_types_customers-1, -1, -1):
        if leftover > 0:
            if s[i] >= leftover:
                s[i] -= leftover
                leftover = 0
            else:
                leftover -= s[i]
                s[i] = 0
    return s

def new_state(s, A_t, a):
    if a == 1:
        s = after_shuttle(s)
    if a == 0 or a == 1:
        for i in range(num_types_customers):
            s[i] = np.minimum(s[i] + A_t[i], max_customer_type_in_station)
    return s

def reward(s, a):
    if a == 0:
        r = -np.dot(c_h, s)
    elif a == 1:
        s = after_shuttle(s)
        r = -c_f - np.dot(c_h, s)
    return r

N = max_customer_type_in_station+1
V_old = np.zeros((N, N, N, N, N))
V_new = np.zeros((N, N, N, N, N))
rhs = np.zeros(num_actions)

counter = 0
error = 99
while error > tol:
    for s1 in range(N):
        for s2 in range(N):
            for s3 in range(N):
                for s4 in range(N):
                    for s5 in range(N):
                        s = np.asarray([s1, s2, s3, s4, s5])
                        for a in range(num_actions):
                            E = 0
                            for A1_t in Ai_t_range:
                                for A2_t in Ai_t_range:
                                    for A3_t in Ai_t_range:
                                        for A4_t in Ai_t_range:
                                            for A5_t in Ai_t_range:
                                                A_t = np.asarray([A1_t, A2_t, A3_t, A4_t, A5_t])
                                                s_prime = new_state(s, A_t, a)
                                                E += prob_Ai_t * V_old[s_prime[0], s_prime[1], s_prime[2], s_prime[3], s_prime[4]]
                            rhs[a] = reward(s, a) + gamma*E
                        V_new[s1, s2, s3, s4, s5] = np.amax(rhs)

    error = np.linalg.norm(np.ndarray.flatten(V_new - V_old), ord=np.inf)
    V_old = np.copy(V_new)

    counter += 1
    print(counter, error)
            
plt.figure(figsize = (8, 6))
plt.plot(V_new[:,0,0,0,0])
plt.xlabel('$s_1$ = type 1 customers waiting', fontsize = 20)
plt.ylabel('V($s_1$, $s_2$=0, $s_3$=0, $s_4$=0, $s_5$=0)', fontsize = 20)
plt.savefig('problem_2b-value_iteration.png')
plt.close()

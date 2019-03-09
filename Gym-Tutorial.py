import gym

'''
    - ENV : Environment
    + `gym.make()`         : Argument : Env_id  | Creates an environment
    + `env.reset()`        : Resets             | Returns : Initial Observation
    + `env.step()`         : Argument : Action  | Returns : Observation, Reward, Done?, Info (debugging info)
    + `env.render()`       : Renders a frame. Can be taken as the representation vector for the state.
    + `env.action_space`   : Returns action vector (set of valid actions you can take). Clases : {0, ... num. of actions}.

    - env.step(1) is possible. It takes action whose "class" is 1.
    + Spaces : Tells valid observations and actions possible.
      - <S, A> of MDP
      - A : env.action_space
      - S : env.observation_space
      

'''

env = env.make('CartPole-v0')
env.reset()

for _ in range(0, episode):
  s_0, s_t = env.reset(), s_0
  done, ret_list = False, []

  for t in range(0, T):
    if(done == True):
      break
    env.render()
    action = env.action_space[greedy_Q(epsillon)]
    s_t, reward, done, info = env.step(action)
    ret_list.append((gamma ** t) * reward)

  # To get return of the ith state
  return_i = sum(ret_list[i:]) / (gamma ** t)
  print(return_i)

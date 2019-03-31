import gym
import numpy as np
import tensorflow as tf

env = gym.make('CartPole-v0')

episode_no, timesteps = 100, 1000
epsillon, gamma = 0.01, 0.5

obv = tf.placeholder(tf.float32, shape=[1, 4]) # CartPole-v0 observation is a vector of 4 elements
q_target = tf.placeholder(tf.float32, shape=[1, 2]) # And has 2 actions

W = tf.Variable(tf.random_normal(shape=[4, 2]))
b = tf.Variable(tf.random_normal(shape=[2]))
qval = tf.matmul(obv, W) + b
a_max = tf.argmax(qval, 1)

loss = tf.reduce_mean(tf.square(q_target - qval))
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for _ in range(0, episode_no):
        s_t = env.reset()
        for t in range(0, timesteps):
            action, qvalue = sess.run([a_max, qval], feed_dict={obv : s_t})
            if(np.random.rand() < epsillon):
                action[0] = env.action_space.sample()

            s_t1, reward, done, _ = env.step(action[0])   # Take an action with epsillon-greedy policy
            q_t1 = sess.run(qval, feed_dict={obv : s_t1}) # Run it for the next state to get the target
            max_qt1 = np.max(q_t1)
            target_q = qvalue
            target_q[0, action[0]] = reward + gamma * max_qt1

            loval, _ = sess.run([loss, train], feed_dict={obv : s_t, q_target : target_q})
            print(loval, end="\r")
            s_t = s_t1

            if(done == True):
                break

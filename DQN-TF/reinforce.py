import gym
import argpase
import numpy as np
import tensorflow as tf

def model(x, sizes, activ=tf.nn.relu, output_activ=None):
    for size in sizes[:-1]:
        x = tf.layers.dense(x, units=size, activation=activ)
    return tf.layers.dense(x, units=sizes[-1], activation=output_activ)

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=0.001, 
            epochs=50, batch_size=5000, render=False):
    
    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    obs_ph = tf.placeholder(tf.float32, shape=[None, obs_dim])
    return_ph = tf.placeholder(tf.float32, shape=[None, ])

    logits = model(obs_ph, sizes=hidden_sizes + n_acts)
    action = tf.squeeze(tf.multinomial(logits=logits, num_samples=1), axis=1)
    loss = - tf.reduce_mean(return_ph * tf.nn.log_softmax(logits))
    train = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for e in range(0, epochs):
            obv_memory, reward_memory, action_memory, return_memory = [], [], [], []
            s_t, done = env.reset(), False

            while done == False:
                if(render):
                    env.render()

                act = sess.run(action, feed_dict={obs_ph:s_t})
                s_t1, reward, done, _ = env.step(act)
                obv_memory.append(s_t)
                action_memory.append(action)
                reward_memory.append(reward)
                s_t = s_t1

            return_memory[len(reward_memory) - 1] = reward_memory[-1]
            for i in range(len(reward_memory) - 2, 0, -1):
                return_memory = reward_memory[i] + gamma * return_memory[i + 1]

            obv_memory, return_memory = np.asarray(obv_memory), np.asarray(return_memory)
            lo, _ = sess.run([loss, train], feed_dict={obs_ph : obv_memory, return_ph : return_memory})

if __name__ == '__main__':
    train()

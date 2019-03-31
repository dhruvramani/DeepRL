import gym
import argpase
import numpy as np
import tensorflow as tf
from model import *

parser = argpase.ArgumentParser(description='DQN Tensorflow Agent')
parser.add_argument('-e', '--epsillon', type=float, default=0.01)
parser.add_argument('-ne', '--no_episodes', type=int, defaut=100, help='Number of Episodes')
parser.add_argument('-ts', '--timesteps', type=int, defaut=100, help='Timesteps per Episode')
parser.add_argument('-basz', '--batch_size', type=int, defaut=64, help='Batch Size')
parser.add_argument('-busz', '--buffer_size', type=int, defaut=5000, help='Buffer Size')
parser.add_argument('-noba', '--no_batches', type=int, default=10, help='Number of batches to be randomly sampled from the memory')
parser.add_argument('--resume', type=int, defaut=0, help='Resume (0/1)')
parser.add_argument('-g', '--gamma', type=float, help='Value of Gamma')
parser.add_argument('-ut', '--update_target', type=int, help='step to update target network')
parser.add_argument('--ckpt_dir', defaut="../save/", help='Path to save the model')

args = parser.parse_args()
env = gym.make('CartPole-v1')

def train():
    mainDQN = DQNAgent(2)
    targetDQN = DQNAgent(2)

    obsv = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    targets = tf.placeholder(tf.float32, shape=[None, 1])
    targval = targetDQN.forward(obsv)
    qvalue = mainDQN.forward(obsv)
    maxaction = tf.argmax(qvalue, 1)
    loss = tf.reduce_mean(tf.square(qvalue - targets))
    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for e in range(0, args.no_episodes):
            s_t = env.reset()
            replay = ExperienceReplay(args.buffer_size)
            
            for t in range(0, args.timesteps):
                env.render()
                action, qv = sess.run([maxaction, qvalue], feed_dict={obsv: s_t})
                if(np.random.rand(1) < args.epsillon):
                    action[0] = env.action_space.sample()
                s_t1, reward, done, _ = env.step(action[0])
                replay.add((s_t, action[0], s_t1, reward))

                for ne in range(0, args.no_batches):
                    batch = replay.sample(args.batch_size)
                    s, s_1, rewards = batch[:, 0], batch[:, 2], batch[:, 3] # Collection of current and next states for the current batch
                    
                    q_s1 = sess.run(targval, feed_dict={obsv: s_1})
                    maxq_s1 = np.max(q_s1, 1)
                    target = rewards + args.gamma * maxq_s1
                    
                    loss, _ = sess.run([loss, train], feed_dict={obsv: s, targets: target})
                    print("Episode : {}, Timestep :{}, Batch : {} - Loss = {}".format(e, t, ne, loss), end="\r")
                
                if(t % args.update_target == 0):
                    targetDQN = mainDQN # Might have to load a saved model of mainDQN

                s_t1 = s_t

if __name__ == '__main__':
    train()
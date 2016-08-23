#!/usr/bin/env python
import argparse
import logging
import sys

import gym
import env_runner

# In modules, use `logger = logging.getLogger(__name__)`
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))

class RandomAgent(object):
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation, reward, done):
        return self.action_space.sample()

def complete(results):
    return len(results['episode_lengths']) == 250

def run_random(env):
    episode_count = 250
    agent = RandomAgent(env.action_space)

    for i in xrange(episode_count):
        ob = env.reset()
        reward = done = None

        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break

def first_ten(id):
    return id < 10

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--base-dir', help='Set base dir.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    runner = env_runner.EnvRunner('random-v3', run_random, complete, base_dir=args.base_dir, video_callable=first_ten)
    runner.run()

    return 0

if __name__ == '__main__':
    sys.exit(main())

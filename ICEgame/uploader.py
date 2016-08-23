#!/usr/bin/env python
import argparse
import logging
import os
import sys

import gym
import env_runner

# In modules, use `logger = logging.getLogger(__name__)`
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))

class Uploader(object):
    def __init__(self, base_dir, algorithm_id, writeup):
        self.base_dir = base_dir
        self.algorithm_id = algorithm_id
        self.writeup = writeup

    def run(self):
        for entry in os.listdir(self.base_dir):
            if entry in ['.', '..']:
                continue
            training_dir = os.path.join(self.base_dir, entry)
            if not os.path.isdir(training_dir):
                logger.info('Skipping: {}'.format(training_dir))
                continue
            gym.upload(training_dir, algorithm_id=self.algorithm_id, writeup=self.writeup)

def main():
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-b', '--base-dir', required=True, help='Set base dir.')
    parser.add_argument('-a', '--algorithm_id', required=True, help='Set the algorithm id.')
    parser.add_argument('-w', '--writeup', help='Writeup to attach.')
    parser.add_argument('-v', '--verbose', action='count', dest='verbosity', default=0, help='Set verbosity.')
    args = parser.parse_args()

    if args.verbosity == 0:
        logger.setLevel(logging.INFO)
    elif args.verbosity >= 1:
        logger.setLevel(logging.DEBUG)

    runner = Uploader(base_dir=args.base_dir, algorithm_id=args.algorithm_id, writeup=args.writeup)
    runner.run()

    return 0

if __name__ == '__main__':
    sys.exit(main())
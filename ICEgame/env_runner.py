import logging
import multiprocessing
import os
import shutil
import signal
import tempfile

import gym
from gym import monitoring

logger = logging.getLogger(__name__)
pool = None

class EnvRunner(object):
    def __init__(self, algorithm_id, training_callable, complete_callable, base_dir=None, video_callable=None, processes=None, env_ids=None):
        global pool
        self.base_dir = base_dir or tempfile.mkdtemp()
        self.training_callable = training_callable
        self.complete_callable = complete_callable
        self.algorithm_id = algorithm_id
        self.video_callable = video_callable

        if env_ids is not None:
            self.specs = [gym.spec(env_id) for env_id in env_ids]
        else:
            self.specs = gym.envs.registry.all()
        self.selected_specs = None

        processes = processes or max(1, multiprocessing.cpu_count() - 1)
        if not pool:
            pool = multiprocessing.Pool(processes)

    def run(self):
        self.select_specs()
        self.train()

    def train(self):
        work = []
        for i, (spec, training_dir) in enumerate(self.selected_specs):
            work.append((self, i, spec, training_dir))

        try:
            pool.map(run_training, work)
        except KeyboardInterrupt:
            pool.terminate()
            pool.join()
            raise

    def select_specs(self):
        specs = self.specs
        selected_specs = []
        for i, spec in enumerate(specs):
            training_dir = self.env_dir(spec.id)
            results = monitoring.load_results(training_dir)
            if results and self.complete_callable(results):
                logger.info('Skipping already-processed %s', spec.id)
                continue
            elif os.path.exists(training_dir):
                shutil.rmtree(training_dir)
            selected_specs.append((spec, training_dir))
        self.selected_specs = selected_specs

    def env_dir(self, id):
        return os.path.join(self.base_dir, id)

# Actually run the training (in the worker)
def run_training((self, i, spec, training_dir)):
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger.info('i=%s id=%s total=%s', i, spec.id, len(self.selected_specs))
    env = spec.make()
    env.monitor.start(training_dir,
                      video_callable=self.video_callable)
    self.training_callable(env)
    # Dump monitor info to disk
    env.monitor.close()

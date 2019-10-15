#
# DQN Helpers
#

import numpy as np
import random
import tensorflow as tf

from collections import deque


class ActionGetter:
    def __init__(self, n_actions, exp_init, exp_final, exp_final_frame):

        self.n_actions = n_actions
        self.exp_init = exp_init
        self.exp_final = exp_final
        self.exp_final_frame = exp_final_frame

        self.exp = self.exp_init
        self.slope = (exp_init - exp_final) / exp_final_frame

        self.action = 0

    def get_best_action(self, sess, frame_number, state, main_dqn,
                        evaluation=False, pre_pop=False):

        if evaluation:
            output = sess.run(
                main_dqn.output,
                feed_dict={main_dqn.input: np.expand_dims(state, 0)})[0]

            self.action = np.argmax(output)
        else:
            exp_tradeoff = np.random.rand()

            if self.exp >= exp_tradeoff:
                self.action = random.randint(0, self.n_actions - 1)
            else:
                output = sess.run(
                    main_dqn.output,
                    feed_dict={main_dqn.input: [state]})

                # output = sess.run(
                #    main_dqn.output,
                #    feed_dict={main_dqn.input: np.expand_dims(state, 0)})[0]

                self.action = np.argmax(output)

            if not pre_pop and frame_number <= self.exp_final_frame:
                self.exp = self.exp - self.slope

        return self.action

class FrameStacker:
    def __init__(self, stack_size=4, frame_width=84, frame_height=84):
        self.stack_size = stack_size
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.stacked_frames = deque(
            [np.zeros((self.frame_width, self.frame_height), dtype=np.uint8)
             for i in range(self.stack_size)],
            maxlen=self.stack_size)
        self.stacked_state = None

    def stack_frames(self, state, is_reset):
        # frame = preprocess_frame(state)
        frame = state

        if is_reset:
            self.stacked_frames = deque([np.zeros(
                (self.frame_width, self.frame_height), dtype=np.uint8) for i in
                range(self.stack_size)],
                maxlen=self.stack_size)

            for i in range(self.stack_size):
                self.stacked_frames.append(frame)

            self.stacked_state = np.stack(self.stacked_frames)  # ,axis=2

        else:
            self.stacked_frames.append(frame)
            self.stacked_state = np.stack(self.stacked_frames)

        return self.stacked_state


class ReplayBuffer:
    def __init__(self, max_size):
        self.buffer = [None] * (max_size + 1)
        self.start = 0
        self.end = 0
        self.current = self.end

    def add(self, experience):
        self.buffer[self.end] = experience
        self.end = (self.end + 1) % len(self.buffer)
        self.current += 1

        if self.end == self.start:
            self.start = (self.start + 1) % len(self.buffer)

    def sample(self, batch_size):
        if self.current < len(self.buffer):
            indexes = random.sample(range(0, self.current), batch_size)
        else:
            indexes = random.sample(range(0, len(self.buffer)), batch_size)

        return [self.buffer[i] for i in indexes]


class ModelParametersCopier:
    def __init__(self, estimator1, estimator2):
        e1_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator1.scope)]
        e1_params = sorted(e1_params, key=lambda v: v.name)
        e2_params = [t for t in tf.trainable_variables() if
                     t.name.startswith(estimator2.scope)]
        e2_params = sorted(e2_params, key=lambda v: v.name)

        self.update_ops = []
        for e1_v, e2_v in zip(e1_params, e2_params):
            op = e2_v.assign(e1_v)
            self.update_ops.append(op)

    def update_target_graph(self, sess):
        sess.run(self.update_ops)

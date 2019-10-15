from env.scheduling_env import *
from env.utils import *
from env.network import *
from env.dqhelpers import *
from matplotlib import style
import matplotlib.pyplot as plt


class Agent:
    def __init__(self, env_name, gamma, buffer_size, batch_size, buffer_start,
                 target_update_freq, update_freq, learning_rate,
                 max_episodes, max_steps, exp_init, exp_final,
                 exp_final_frame, training=True,
                 use_target=True, restore_session=False, write_summary=True):

        self.plot1 = None
        self.plot2 = None

        self.gamma = gamma
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.buffer_start = buffer_start
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.learning_rate = learning_rate
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.training = training
        self.use_target = use_target
        self.restore_session = restore_session
        self.exp_final_frame = exp_final_frame
        self.write_summary = write_summary

        self.env = SchedulingEnv()
        self.best_env = copy.deepcopy(self.env)
        self.mean_rewards = []
        self.max_reward = 0
        # self.env = TaskDay()

        self.action_size = self.env.action_space.n
        print("action_size: ", self.action_size)
        self.state_shape = self.env.observation_space.shape
        print("state_shape: ", self.state_shape)

        self.replay_buffer = ReplayBuffer(max_size=buffer_size)

        self.action_getter = ActionGetter(self.action_size, exp_init,
                                          exp_final, exp_final_frame)

        self.dqn_network = None
        self.target_network = None
        self.network_copier = None
        self.summary_writer = None

        tf.reset_default_graph()

    def train(self):
        with tf.Session() as sess:

            with tf.variable_scope('dqn_network'):
                self.dqn_network = DQNetwork(84, 84, self.state_shape,
                                             self.action_size,
                                             self.learning_rate, scope="dqn_network")

            if self.use_target:
                with tf.variable_scope('target_network'):
                    self.target_network = DQNetwork(84, 84, self.state_shape,
                                                    self.action_size,
                                                    self.learning_rate,
                                                    scope="target_network")

                self.network_copier = ModelParametersCopier(self.dqn_network,
                                                            self.target_network)

            if self.write_summary:
                self.summary_writer = tf.summary.FileWriter('./log')
            sess.run(tf.global_variables_initializer())
            episode = 0
            # TODO: restore previous session

            state = self.env.reset()

            # In this loop the replay buffer will be pre-populated with
            # experiences by performing random actions
            for i in range(self.buffer_start):
                if i % 100 == 0:
                    print("Filling Buffer... ", (i * 100) / self.buffer_start, "%")

                action = random.randint(0, self.action_size - 1)

                next_state, reward, done, _ = self.env.step(action)

                self.replay_buffer.add((state, action,
                                        reward, next_state, done))
                if done:
                    state = self.env.reset()
                else:
                    state = next_state

            if self.use_target:
                self.network_copier.update_target_graph(sess)

            print("Starting Training!")
            # Training starts now
            all_rewards = []
            self.max_reward = -999999

            best_env = self.env
            conflicts = 0
            episode_conflicts = []
            best_episode_conflicts = 0

            c_before = 0
            c_after = 0
            a_before = 0
            a_after = 0
            while episode < self.max_episodes:
                epoch_frame = 0
                state = self.env.reset()
                sum_rewards = 0
                for step in range(self.max_steps):
                    action = self.action_getter.get_best_action(
                        sess,
                        episode,
                        state,
                        self.dqn_network,
                        pre_pop=False
                    )
                    next_state, reward, done, _ = self.env.step(action)
                    sum_rewards += reward
                    self.replay_buffer.add((state, action,
                                            reward, next_state, done))

                    all_rewards.append(reward)

                    if episode % self. update_freq == 0:
                        batch = self.replay_buffer.sample(self.batch_size)

                        states_b = np.array([each[0] for each in batch])
                        actions_b = np.array([each[1] for each in batch])
                        rewards_b = np.array([each[2] for each in batch])
                        next_states_b = np.array(
                            [each[3] for each in batch])
                        terminals_b = np.array([each[4] for each in batch])

                        if self.use_target:
                            q_vals_target = sess.run(
                                self.target_network.output,
                                feed_dict={
                                    self.target_network.input: next_states_b
                                }
                            )
                            target_q = rewards_b + np.invert(
                                terminals_b).astype(
                                np.float32) * self.gamma * np.amax(
                                q_vals_target, axis=1)
                        else:
                            q_vals_target = sess.run(
                                self.dqn_network.output,
                                feed_dict={
                                    self.dqn_network.input: next_states_b
                                }
                            )
                            target_q = rewards_b + np.invert(
                                terminals_b).astype(
                                np.float32) * self.gamma * np.amax(
                                q_vals_target, axis=1)  # (...q_vals_target, axis=1)

                        """target_qs = []
                        for i in range(0, self.batch_size):
                            terminal = terminals_b[i]

                            if terminal:
                                target_qs.append(rewards_b[i])
                            else:
                                target = rewards_b[i] + self.gamma * np.max(next_states_b[i])
                                target_qs.append(target)"""

                        loss, _, summaries = sess.run(
                            [self.dqn_network.loss, self.dqn_network.update, self.dqn_network.summaries],
                            feed_dict={
                                self.dqn_network.input: states_b,
                                self.dqn_network.target_Q: target_q,
                                self.dqn_network.actions: actions_b
                            })
                        self.summary_writer.add_summary(summaries, episode)

                    if (self.use_target and
                            episode % self.target_update_freq == 0):
                        self.network_copier.update_target_graph(sess)

                    episode += 1
                    epoch_frame += 1

                    conflicts += 1

                    if done:
                        print("Training... ", (episode * 100) / self.max_episodes, "%")
                        print("Conflicts: ", self.env.conf_number)
                        print("A-check conflicts: ", self.env.a_check_conf_number)
                        print("C-check conflicts: ", self.env.c_check_conf_number)

                        r = self.env.reward_calendar()
                        self.mean_rewards.append(r)
                        if r > self.max_reward:
                            self.max_reward = r
                            best_episode_conflicts = conflicts
                            c_before = self.env.c_tasks_before
                            c_after = self.env.c_tasks_after
                            a_before = self.env.a_tasks_before
                            a_after = self.env.a_tasks_after
                            self.best_env = copy.deepcopy(self.env)

                        print("Reward: ", r)
                        print("::::::::::::::::::::::::::::::::::::::::::::")

                        episode_conflicts.append(conflicts)
                        conflicts = 0

                        break

            print("Max Reward", self.max_reward)

            print("Number of conflicts in best episode: ", best_episode_conflicts)
            print("Mean number of conflicts: ", sum(episode_conflicts) / len(episode_conflicts))

            self.plot2 = self.mean_rewards

            render_results_excel(self, self.mean_rewards, self.max_reward, episode_conflicts, best_episode_conflicts, c_before,
                                 c_after, a_before, a_after)

            render_calendar_excel(self.best_env, 'calendar.xlsx')
            create_json(self.best_env)
            task_oriented_json(self.best_env)
            maintenance_plan_json(self.best_env)


plot1 = None
plot2 = None


if __name__ == '__main__':
    tf.enable_eager_execution()

    dqn_agent = Agent("Seaquest-v0",
                      gamma=0.9,
                      buffer_size=10000,
                      batch_size=32,
                      buffer_start=10000,
                      target_update_freq=10000,
                      update_freq=4,
                      learning_rate=0.00025,
                      max_episodes=7500,
                      max_steps=200000000,
                      exp_init=1.0,
                      exp_final=0.05,
                      exp_final_frame=500000)
    dqn_agent.train()

    plot1 = dqn_agent.plot1
    plot2 = dqn_agent.plot2

    style.use('fivethirtyeight')

    plt.figure(figsize=(8, 6))

    plt.xlabel('Episode', fontsize=20)
    plt.ylabel('Reward', fontsize=20)

    plt.plot(plot2, color='#ee9b0b')

    plt.show()
    print("Max", max(plot2))

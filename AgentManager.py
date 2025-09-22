import math

import torch
import torch.nn as nn
import numpy as np
import pygame
import torch.nn.functional as F
from torch import optim

from torch.distributions import Normal
from config import config

class NeuralNetwork(nn.Module):
    def __init__(self, inputs):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(inputs, 4)
        self.fc2 = nn.Linear(4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.tanh(self.fc2(x))
        return x


class AgentManager:
    def __init__(self, screen, num_agents, size, energy, rays_range, positions, spatial_hash, environment):
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.screen = screen
        self.num_agents = num_agents
        self.env = environment
        self.max_speed = config['max speed']
        self.max_angular_speed = 10
        self.sensing_capability = 3  # agent can see n objects around it

        self.start_pos = positions
        self.start_energy = energy

        self.positions = np.array(positions, dtype=float)  # (num_agents, 2)
        self.energies = np.ones(num_agents, dtype=float) * energy
        self.speeds = np.ones(num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, num_agents).astype(np.float32)
        self.rewards = np.zeros((num_agents, 1), dtype=int)

        self.start_size = size
        self.size = np.random.uniform(self.start_size, self.start_size, num_agents)

        self.fov = math.radians(config['fov'])

        self.team = np.zeros(num_agents, dtype=int)
        self.team[:num_agents // 2] = -1  # team 1
        self.team[num_agents // 2:] = 1  # team 2

        self.colors = np.zeros((num_agents, 3), dtype=int)
        self.colors[self.team == -1] = [0, 0, 255]  # team 1
        self.colors[self.team == 1] = [255, 0, 0]  # team 2

        # pointing error 1, distance 1, size 1, team 1, pointing error 2, distance 2, ...
        self.nn_inputs = self.sensing_capability * 4
        self.neural_networks = nn.ModuleList([NeuralNetwork(self.nn_inputs).to(self.device) for _ in range(num_agents)])

        self.sensing_range = rays_range

        self.spatial_hash = spatial_hash
        self.alive = (self.energies > 0)
        self.file_path = 'weights.npy'

    def move(self, dt):
        dx = self.speeds[self.alive] * np.cos(self.directions[self.alive]) * dt
        dy = self.speeds[self.alive] * np.sin(self.directions[self.alive]) * dt
        self.positions[self.alive] += np.stack([dx, dy], axis=1)
        self.positions[:, 0] = self.positions[:, 0] % self.env.width
        self.positions[:, 1] = self.positions[:, 1] % self.env.height
        self.energies[self.alive] -= dt

    def decide_actions_for_agents(self, sensing_data_batch, dt):
        inputs_tensor = torch.tensor(sensing_data_batch, dtype=torch.float32).to(self.device)
        inputs_tensor = inputs_tensor.view(self.num_agents, -1)

        outputs = torch.cat([self.neural_networks[i](inputs_tensor[i].unsqueeze(0)) for i in range(self.num_agents)],
                            dim=0)
        outputs = outputs.squeeze()
        #actions = torch.argmax(outputs, dim=1)

        #speeds = np.zeros(self.num_agents, dtype=np.float16)
        #angular_speeds = np.zeros(self.num_agents, dtype=np.float16)

        '''
        for i, action in enumerate(actions.cpu().numpy()):
            if action == 1: # dx
                self.directions[i] += self.max_angular_speed * dt
            elif action == 2: # sx
                self.directions[i] -= self.max_angular_speed * dt
        '''

        outputs = outputs.squeeze()
        speeds = outputs[:, 0] * self.max_speed

        angular_speeds = outputs[:, 1] * self.max_angular_speed

        angular_speeds_np = angular_speeds.cpu().detach().numpy()

        angular_speeds = np.array(angular_speeds_np, dtype=np.float16)
        self.directions += angular_speeds * dt
        self.directions = self.directions % (2 * np.pi)
        self.speeds = np.array(speeds.cpu().detach().numpy(), dtype=np.float16)

    def update(self, dt, sensing_data):
        # sensing_data_batch = self.get_inputs_for_neural_networks(sensing_data)
        self.alive = (self.energies > 0)
        self.decide_actions_for_agents(sensing_data, dt)
        self.move(dt)

    def reset(self, new_weigths):
        if new_weigths is not None:
            self.import_weights(new_weigths)
        self.positions = np.array(self.start_pos, dtype=np.float32)  # (num_agents, 2)
        self.energies = np.ones(self.num_agents, dtype=np.float32) * self.start_energy
        self.speeds = np.ones(self.num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, self.num_agents).astype(np.float32)
        self.size = np.ones(self.num_agents, dtype=np.float32) * self.start_size



    def export_weights(self):
        all_weights = []
        for nn in self.neural_networks:
            nn_weights = []
            for param in nn.parameters():
                nn_weights.append(param.data.cpu().numpy().flatten())
            all_weights.append(np.concatenate(nn_weights))
        return np.array(all_weights, dtype=object)

    def import_weights(self, weights_array):
        for nn, nn_weights in zip(self.neural_networks, weights_array):
            offset = 0
            for param in nn.parameters():
                num_weights = param.numel()
                param_weights = nn_weights[offset:offset + num_weights]
                param_weights = np.array(param_weights, dtype=np.float32)

                param.data = torch.tensor(param_weights.reshape(param.shape), dtype=torch.float32,
                                          device=param.device)
                offset += num_weights

    def draw(self, i):
        if self.energies[i] > 0:
            pygame.draw.circle(self.screen, self.colors[i], (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])

            line_start = (int(self.positions[i, 0]), int(self.positions[i, 1]))
            line_end = (int(self.positions[i][0] + self.size[i] * 2 * np.cos(self.directions[i])),
                        int(self.positions[i][1] + self.size[i] * 2 * np.sin(self.directions[i])))

            pygame.draw.line(self.screen, self.colors[i], line_start, line_end, int(self.size[i] / 2))
            '''
            theta = self.directions[i] - self.fov
            line_end = (int(self.positions[i][0] + self.sensing_range * np.cos(theta)),
                        int(self.positions[i][1] + self.sensing_range * np.sin(theta)))

            pygame.draw.line(self.screen, self.colors[i], line_start, line_end, 1)

            theta = self.directions[i] + self.fov
            line_end = (int(self.positions[i][0] + self.sensing_range * np.cos(theta)),
                        int(self.positions[i][1] + self.sensing_range * np.sin(theta)))
            pygame.draw.line(self.screen, self.colors[i], line_start, line_end, 1)'''
        else:
            pygame.draw.circle(self.screen, (80, 80, 80), (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])

        # x,y = self.positions[i]
        # rect = pygame.Rect(x - self.sensing_range, y - self.sensing_range, self.sensing_range*2, self.sensing_range*2)
        # pygame.draw.rect(self.screen, (0,0,0), rect, 1)  # Disegna solo il contorno della cella

    def save(self):
        all_weights = self.export_weights()
        np.save(self.file_path, all_weights)
        print(f"Saved in {self.file_path}")

    def load_weights_from_file(self, file_path):
        all_weights = np.load(file_path, allow_pickle=True)
        print(f"Loading from {self.file_path}")
        return all_weights

    def load(self):
        loaded_weights = self.load_weights_from_file('ga_best_weights.npy')
        self.import_weights(loaded_weights)


class RolloutBuffer:
    def __init__(self, batch_size = 64, buffer_size = 5000):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.batch_size = batch_size
        self.inserted_vals = 0
        self.buffer_size = buffer_size

    def add(self, state, action, logprob, reward, done, value):
        if self.inserted_vals < self.buffer_size:
            self.states.append(state)
            self.actions.append(action)
            self.logprobs.append(logprob)
            self.rewards.append(reward)
            self.dones.append(done)
            self.values.append(value)
            self.inserted_vals += 1

    def generate_batches(self):
        batch_start = np.arange(0, self.inserted_vals, self.batch_size)
        indices = np.arange(self.inserted_vals, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.logprobs),\
                np.array(self.rewards),\
                np.array(self.dones),\
                np.array(self.values),\
                batches

    def clear(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.inserted_vals = 0

    def print_element(self, i):
        print('state', self.states[i])
        print('action', self.actions[i])
        print('logp', self.logprobs[i])
        print('reward', self.rewards[i])
        print('done', self.dones[i])
        print('val', self.values[i])

class ActorNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 4)
        self.fc2 = nn.Linear(4, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))  # Learnable std deviation

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        mean = torch.tanh(self.fc2(x))  # Mean of the Gaussian
        std = torch.exp(self.log_std)  # Standard deviation
        return mean, std


class CriticNetwork(nn.Module):
    def __init__(self, input_dims):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dims, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        value = self.fc2(x)
        return value


class AgentManagerPPO:
    def __init__(self, screen, num_agents, size, energy, rays_range, positions, spatial_hash, environment):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.screen = screen
        self.num_agents = num_agents
        self.env = environment
        self.max_speed = config['max speed']
        self.max_angular_speed = 10
        self.sensing_capability = 3

        self.start_pos = positions
        self.start_energy = energy

        self.positions = np.array(positions, dtype=float)
        self.energies = np.ones(num_agents, dtype=float) * energy
        self.speeds = np.ones(num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, num_agents).astype(np.float32)
        self.rewards = np.zeros((num_agents, 1), dtype=int)

        self.start_size = size
        self.size = np.random.uniform(self.start_size, self.start_size, num_agents)

        self.fov = math.radians(config['fov'])

        self.team = np.zeros(num_agents, dtype=int)
        self.team[:num_agents // 2] = -1
        self.team[num_agents // 2:] = 1

        self.colors = np.zeros((num_agents, 3), dtype=int)
        self.colors[self.team == -1] = [0, 0, 255]
        self.colors[self.team == 1] = [255, 0, 0]

        self.sensing_range = config['agents range']
        self.spatial_hash = spatial_hash
        self.alive = (self.energies > 0)
        self.buffer = RolloutBuffer()

        self.nn_inputs = self.sensing_capability * 4
        self.actor = ActorNetwork(self.nn_inputs, n_actions=2).to(self.device)
        self.critic = CriticNetwork(self.nn_inputs).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.total_rewards = []
        self.epoch_rewards = 0

    def decide_actions_for_agents(self, sensing_data_batch, dt):
        inputs_tensor = torch.tensor(sensing_data_batch, dtype=torch.float32).to(self.device)
        mean, std = self.actor(inputs_tensor)
        dist = Normal(mean, std)

        actions = dist.sample()

        logprobs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(inputs_tensor)

        self.epoch_rewards += np.sum(self.rewards)

        for i in range(self.num_agents):
            if self.alive[i]:
                    reward = self.rewards[i]
                    self.buffer.add(
                        state=inputs_tensor[i].cpu().numpy(),
                        action=actions[i].detach().cpu().numpy(),  # Detach!
                        logprob=logprobs[i].detach().item(),  # Detach!
                        reward=reward,
                        done=not self.alive[i],
                        value=values[i].detach().item()  # Detach!
                    )

            else:
                if (self.buffer.inserted_vals > 0) and (self.buffer.dones[-1] == False): # if the agent has just died
                    self.buffer.add(
                        state=inputs_tensor[i].cpu().numpy(),
                        action=actions[i].detach().cpu().numpy(),
                        logprob=logprobs[i].detach().item(),
                        reward=self.rewards[i],
                        done=True,
                        value=values[i].detach().item()
                    )

            self.rewards[i] = 0

        speeds = np.clip(actions[:, 0].cpu().numpy() * self.max_speed, -self.max_speed, self.max_speed)
        angular_speeds = actions[:, 1].cpu().numpy() * self.max_angular_speed

        self.directions += angular_speeds * dt
        self.directions %= (2 * np.pi)
        self.speeds = speeds


    def update_policy(self, epochs=5, clip_epsilon=0.2, gamma=0.98, gae_lambda=0.9):
        state_arr, action_arr, old_prob_arr, reward_arr, dones_arr, vals_arr, batches = self.buffer.generate_batches()
        reward_arr = np.array(reward_arr, dtype=np.float32)
        dones_arr = np.array(dones_arr, dtype=np.float32)
        vals_arr = np.array(vals_arr, dtype=np.float32)
        values = vals_arr  # alias

        advantage = np.zeros_like(reward_arr, dtype=np.float32)
        last_adv = 0.0
        N = len(reward_arr)


        # advantage
        for t in reversed(range(N - 1)):
            #print(t, end = ',')
            delta = reward_arr[t] + gamma * values[t + 1] * (1 - dones_arr[t]) - values[t]
            last_adv = delta + gamma * gae_lambda * (1 - dones_arr[t]) * last_adv
            advantage[t] = last_adv

        advantage = torch.tensor(advantage, dtype=torch.float32, device=self.device).view(-1)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device).view(-1)

        for _ in range(epochs):
            for batch in batches:
                states = torch.tensor(np.array(state_arr)[batch], dtype=torch.float32, device=self.device)
                actions = torch.tensor(np.array(action_arr)[batch], dtype=torch.float32, device=self.device)
                old_log_probs = torch.tensor(np.array(old_prob_arr)[batch], dtype=torch.float32, device=self.device)

                # returns = advantage + values
                batch_advantage = advantage[batch].detach()
                batch_returns = (advantage + values_tensor)[batch].detach()

                mean, std = self.actor(states)
                dist = Normal(mean, std)

                new_log_probs = dist.log_prob(actions)
                if new_log_probs.ndim > 1:
                    new_log_probs = new_log_probs.sum(dim=-1)

                # rt(theta) = pi_theta/pi_theta_old
                prob_ratio = torch.exp(new_log_probs - old_log_probs)

                # rt(theta) * At
                weighted_probs = batch_advantage * prob_ratio
                # clipping
                weighted_clipped = batch_advantage * torch.clamp(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)
                actor_loss = -torch.min(weighted_probs, weighted_clipped).mean()

                # critic loss = (V(st) - Rt)^2
                critic_value = self.critic(states).view(-1)
                critic_loss = (batch_returns - critic_value).pow(2).mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.buffer.clear()

    def move(self, dt):
        dx = self.speeds[self.alive] * np.cos(self.directions[self.alive]) * dt
        dy = self.speeds[self.alive] * np.sin(self.directions[self.alive]) * dt
        self.positions[self.alive] += np.stack([dx, dy], axis=1)
        self.positions[:, 0] = self.positions[:, 0] % self.env.width
        self.positions[:, 1] = self.positions[:, 1] % self.env.height
        self.energies[self.alive] -= dt

    def update(self, dt, sensing_data):
        self.alive = (self.energies > 0)
        self.decide_actions_for_agents(sensing_data, dt)
        self.move(dt)

    def reset(self, update):
        self.total_rewards.append(self.epoch_rewards)
        if len(self.total_rewards) % 10 == 0:
            print("Reward totale:", self.total_rewards)
        self.epoch_rewards = 0

        if update:
            print('updating with data retrieved: ', self.buffer.inserted_vals)
            self.update_policy()
        self.positions = np.array(self.start_pos, dtype=np.float32)  # (num_agents, 2)
        self.energies = np.ones(self.num_agents, dtype=np.float32) * self.start_energy
        self.alive = (self.energies > 0)
        self.speeds = np.ones(self.num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, self.num_agents).astype(np.float32)
        self.size = np.ones(self.num_agents, dtype=np.float32) * self.start_size



    def export_weights(self):
        return [param.data.cpu().numpy() for param in self.ppo_network.parameters()]

    def import_weights(self, weights_array):
        for param, weight in zip(self.ppo_network.parameters(), weights_array):
            param.data = torch.tensor(weight, dtype=torch.float32).to(self.device)

    def draw(self, i):
        if self.energies[i] > 0:
            pygame.draw.circle(self.screen, self.colors[i], (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])

            line_start = (int(self.positions[i, 0]), int(self.positions[i, 1]))
            line_end = (int(self.positions[i][0] + self.size[i] * 2 * np.cos(self.directions[i])),
                        int(self.positions[i][1] + self.size[i] * 2 * np.sin(self.directions[i])))

            pygame.draw.line(self.screen, self.colors[i], line_start, line_end, int(self.size[i] / 2))
        else:
            pygame.draw.circle(self.screen, (80, 80, 80), (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])

        # x,y = self.positions[i]
        # rect = pygame.Rect(x - self.sensing_range, y - self.sensing_range, self.sensing_range*2, self.sensing_range*2)
        # pygame.draw.rect(self.screen, (0,0,0), rect, 1)  # Disegna solo il contorno della cella

    def save(self, path_actor='actor.pth', path_critic='critic.pth'):
        torch.save(self.actor.state_dict(), path_actor)
        torch.save(self.critic.state_dict(), path_critic)

    def load(self, path_actor='actor.pth', path_critic='critic.pth'):
        try:
            self.actor.load_state_dict(torch.load(path_actor))
            self.critic.load_state_dict(torch.load(path_critic))
            self.actor.eval()
            self.critic.eval()
            print(f"Modelli caricati da {path_actor} e {path_critic}")
        except FileNotFoundError:
            print(f"Errore: i file {path_actor} o {path_critic} non sono stati trovati.")



class AgentManagerGA_PPO:
    def __init__(self, screen, num_agents, size, energy, rays_range, positions, spatial_hash, environment):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.screen = screen
        self.num_agents = num_agents
        self.env = environment
        self.max_speed = 150
        self.max_angular_speed = 10
        self.sensing_capability = 3

        self.start_pos = positions
        self.start_energy = energy

        self.positions = np.array(positions, dtype=float)
        self.energies = np.ones(num_agents, dtype=float) * energy
        self.speeds = np.ones(num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, num_agents).astype(np.float32)
        self.rewards = np.zeros((num_agents, 1), dtype=int)

        self.start_size = size
        self.size = np.random.uniform(self.start_size, self.start_size, num_agents)

        self.fov = math.radians(config['fov'])

        self.team = np.zeros(num_agents, dtype=int)
        self.team[:num_agents // 2] = -1
        self.team[num_agents // 2:] = 1

        self.colors = np.zeros((num_agents, 3), dtype=int)
        self.colors[self.team == -1] = [0, 0, 255]
        self.colors[self.team == 1] = [255, 0, 0]

        self.sensing_range = rays_range
        self.spatial_hash = spatial_hash
        self.alive = (self.energies > 0)

        self.nn_inputs = self.sensing_capability * 4

        self.neural_networks = nn.ModuleList([NeuralNetwork(self.nn_inputs).to(self.device) for _ in range(num_agents//2)])

        self.actor = ActorNetwork(self.nn_inputs, n_actions=2).to(self.device)
        self.critic = CriticNetwork(self.nn_inputs).to(self.device)

        self.load_GA()
        self.load_PPO()

    def decide_actions_for_agents_PPO(self, sensing_data_batch, dt):
        inputs_tensor = torch.tensor(sensing_data_batch, dtype=torch.float32).to(self.device)
        mean, std = self.actor(inputs_tensor)
        dist = Normal(mean, std) # std = 0 buon idea?

        actions = dist.sample()

        speeds = torch.clamp((actions[:, 0]+1) * self.max_speed, 0, self.max_speed).cpu().numpy()
        angular_speeds = torch.clamp(actions[:, 1] * self.max_angular_speed, -self.max_angular_speed,
                                     self.max_angular_speed).cpu().numpy()

        self.directions[self.num_agents//2:] += angular_speeds * dt
        self.directions[self.num_agents//2:] %= (2 * np.pi)
        self.speeds[self.num_agents//2:] = speeds

    def decide_actions_for_agents_GA(self, sensing_data_batch, dt):
        inputs_tensor = torch.tensor(sensing_data_batch, dtype=torch.float32).to(self.device)
        inputs_tensor = inputs_tensor.view(self.num_agents//2, -1)

        outputs = torch.cat([self.neural_networks[i](inputs_tensor[i].unsqueeze(0)) for i in range(self.num_agents//2)],
                            dim=0)
        outputs = outputs.squeeze()
        speeds = (outputs[:, 0] + 1) * self.max_speed

        angular_speeds = torch.clamp(outputs[:, 1] * self.max_angular_speed, -self.max_angular_speed,
                                     self.max_angular_speed)

        angular_speeds_np = angular_speeds.cpu().detach().numpy()

        angular_speeds = np.array(angular_speeds_np, dtype=np.float16)
        self.directions[0:self.num_agents//2] += angular_speeds * dt
        self.directions[0:self.num_agents//2] %= (2 * np.pi)
        self.speeds[0:self.num_agents//2] = np.array(speeds.cpu().detach().numpy(), dtype=np.float16)


    def move(self, dt):
        dx = self.speeds[self.alive] * np.cos(self.directions[self.alive]) * dt
        dy = self.speeds[self.alive] * np.sin(self.directions[self.alive]) * dt
        self.positions[self.alive] += np.stack([dx, dy], axis=1)
        self.positions[:, 0] = self.positions[:, 0] % self.env.width
        self.positions[:, 1] = self.positions[:, 1] % self.env.height
        self.energies[self.alive] -= dt

    def update(self, dt, sensing_data):
        self.alive = (self.energies > 0)
        self.decide_actions_for_agents_GA(sensing_data[0:self.num_agents//2], dt)
        self.decide_actions_for_agents_PPO(sensing_data[self.num_agents//2:], dt)
        self.move(dt)

    def reset(self):
        self.positions = np.array(self.start_pos, dtype=np.float32)
        self.energies = np.ones(self.num_agents, dtype=np.float32) * self.start_energy
        self.speeds = np.ones(self.num_agents, dtype=int) * self.max_speed
        self.directions = np.random.uniform(0, 2 * np.pi, self.num_agents).astype(np.float32)
        self.size = np.ones(self.num_agents, dtype=np.float32) * self.start_size
        #self.processed = np.zeros(self.num_agents, dtype=bool)


    def draw(self, i):
        if self.energies[i] > 0:
            pygame.draw.circle(self.screen, self.colors[i], (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])

            line_start = (int(self.positions[i, 0]), int(self.positions[i, 1]))
            line_end = (int(self.positions[i][0] + self.size[i] * 2 * np.cos(self.directions[i])),
                        int(self.positions[i][1] + self.size[i] * 2 * np.sin(self.directions[i])))

            pygame.draw.line(self.screen, self.colors[i], line_start, line_end, int(self.size[i] / 2))
        else:
            pygame.draw.circle(self.screen, (80, 80, 80), (int(self.positions[i, 0]), int(self.positions[i, 1])),
                               self.size[i])


    def load_PPO(self, path_actor='actor.pth', path_critic='critic.pth'):
        try:
            self.actor.load_state_dict(torch.load(path_actor))
            self.critic.load_state_dict(torch.load(path_critic))
            self.actor.eval()
            self.critic.eval()
            print(f"Modelli caricati da {path_actor} e {path_critic}")
        except FileNotFoundError:
            print(f"Errore: i file {path_actor} o {path_critic} non sono stati trovati.")

    def load_weights_from_file(self, file_path):
        all_weights = np.load(file_path, allow_pickle=True)
        print(f"Loading from {file_path}")
        return all_weights

    def load_GA(self):
        loaded_weights = self.load_weights_from_file('ga_best_weights.npy')
        self.import_weights(loaded_weights)

    def import_weights(self, weights_array):
        for nn, nn_weights in zip(self.neural_networks, weights_array):
            offset = 0
            for param in nn.parameters():
                num_weights = param.numel()
                param_weights = nn_weights[offset:offset + num_weights]
                param_weights = np.array(param_weights, dtype=np.float32)

                param.data = torch.tensor(param_weights.reshape(param.shape), dtype=torch.float32,
                                          device=param.device)
                offset += num_weights

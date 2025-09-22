import concurrent


from SpatialHash import SpatialHash
from AgentManager import AgentManager, AgentManagerPPO, AgentManagerGA_PPO
import numpy as np
from GeneticAlgorithm import GeneticAlgorithm
import math
import random
import pygame
from config import config

def get_distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)




class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius

    def draw(self, screen):
        pygame.draw.circle(screen, (0, 0, 0), (int(self.x), int(self.y)), self.radius)


class Food:
    def __init__(self, screen, id, x, y):
        self.screen = screen
        # print(f'new food of id {id}')
        self.id = id
        self.x = x
        self.y = y
        self.size = 5
        self.rect = pygame.Rect(self.x - self.size, self.y - self.size, self.size, self.size)

    def draw(self):
        pygame.draw.circle(self.screen, (0, 80, 0), (self.x, self.y), self.size)


class Environment:
    def __init__(self, screen, testing):
        self.screen = screen
        self.width = config['map width']
        self.height = config['map height']
        self.agents_range = config['agents range']
        self.agent_starting_energy = config['agents starting energy']
        self.agents_blue = []
        self.agents_red = []
        self.foods = []
        self.num_agents = config['num agents']
        self.energy_gain = config['energy gain']
        self.num_foods = config['num foods']
        self.num_obstacles = config['num obstacles']
        self.obstacles = []
        self.rewards = np.zeros((self.num_agents, 1), dtype=int)
        self.best = []

        self.ga = GeneticAlgorithm()

        self.spatial_hash = SpatialHash(cell_size=50)


        positions = self.make_world()

        self.generations = 0
        self.testing = testing
        if testing:
            print('testing config')
            self.am = AgentManagerGA_PPO(screen, self.num_agents, 10, self.agent_starting_energy, 200,
                                         positions, self.spatial_hash, self)

        else:
            if config['use ppo']:
                self.am = AgentManagerPPO(screen, self.num_agents, 10, self.agent_starting_energy, 200,
                                          positions, self.spatial_hash, self)
            else:
                self.am = AgentManager(screen, self.num_agents, 10, self.agent_starting_energy, 200,
                                       positions, self.spatial_hash, self)

    def generate_random_pos(self, num, positions, dist):
        my_pos = []
        for i in range(0, num):
            ok = False
            while not ok:
                valid = True
                x = random.randint(10, self.width-10)
                y = random.randint(10, self.height-10)
                for pos in positions:
                    if math.dist((x, y), pos) <= dist:
                        valid = False
                        break
                if valid:
                    ok = True
            my_pos.append([x,y])
        return my_pos

    def make_world(self):
        obst = self.generate_obst()
        food = self.generate_food(obst)
        self.food_pos = food
        return self.generate_starting(np.vstack((obst, food)))

    def generate_starting(self, obst):
        pos = self.generate_random_pos(self.num_agents, obst, 20)
        return pos

    def generate_obst(self):
        self.obstacles = []
        my_pos = []

        for i in range(self.num_obstacles):
            x = random.randint(100, self.width - 100)
            y = random.randint(100, self.height - 100)

            obstacle = Obstacle(x, y, 30)
            my_pos.append([x, y])
            self.obstacles.append(obstacle)
            self.spatial_hash.insert(i, -2, x, y, obstacle.radius)
        return my_pos

    def generate_food(self, positions):
        self.foods = []
        my_pos = self.generate_random_pos(self.num_foods, positions, 100)

        for i in range(len(my_pos)):
            x = my_pos[i][0]
            y = my_pos[i][1]
            food = Food(self.screen, self.num_obstacles + i, x, y)
            self.foods.append(food)
            self.spatial_hash.insert(food.id, 0, x, y, 5)
        return my_pos

    def draw(self, screen):
        for food in self.foods:
            food.draw()

        for obstacle in self.obstacles:
            obstacle.draw(screen)

        # self.spatial_hash.draw(self.screen)

    def food_eaten(self, food_id):
        x, y = 10000, 10000
        real_id = food_id - self.num_obstacles
        self.foods[real_id].x = x
        self.foods[real_id].y = y
        self.spatial_hash.insert(food_id, 0, x, y, 5)

    def load(self):

        if not self.testing:
            self.am.load()
            self.load_gen()

    def update(self, dt):
        self.am.update(dt, self.get_inputs())

    def save(self):
        if not self.testing:
            self.am.save()
        # self.statistics.save()

    import concurrent.futures

    def reset(self):
        if not self.testing:
            self.spatial_hash.clear()

            # Esegui make_world e am.reset in parallelo
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_make_world = executor.submit(self.make_world)
                if config['use ppo']:
                    future_reset = executor.submit(self.am.reset, True)
                else:
                    weights = self.am.export_weights()
                    new_w, self.best = self.ga.do_stuff(weights, self.am.rewards)
                    future_reset = executor.submit(self.am.reset, new_w)

                self.am.positions = future_make_world.result()
                future_reset.result()

        else:
            self.am.reset()

        for i in range(0, len(self.foods)):
            self.foods[i].x = self.food_pos[i][0]
            self.foods[i].y = self.food_pos[i][1]
            self.spatial_hash.insert(self.foods[i].id, 0, self.foods[i].x, self.foods[i].y, 5)

        self.am.rewards = np.zeros((self.num_agents, 1), dtype=int)
        self.generations += 1
        self.save_gen()

    def get_inputs(self):
        input_batch = np.zeros((self.num_agents, self.am.nn_inputs))

        for i in range(self.am.num_agents):
            self.am.draw(i)  # draw agent
            start_index = self.num_foods + self.num_obstacles
            if self.am.alive[i]:
                self.spatial_hash.insert(start_index + i, self.am.team[i], int(self.am.positions[i, 0]),
                                         int(self.am.positions[i, 1]),
                                         self.am.size[i])
                input_batch[i][:] = np.array(self.sensing(i), dtype=np.float32)
            else:
                self.spatial_hash.remove(start_index + i)
                input_batch[i][:] = (np.zeros(self.am.nn_inputs, dtype=np.float32))
        return input_batch

    def sensing(self, agent_idx):
        x_agente, y_agente = self.am.positions[agent_idx]
        sensing_range = self.am.sensing_range
        rect = pygame.Rect(x_agente - sensing_range, y_agente - sensing_range, sensing_range * 2, sensing_range * 2)

        nearby_objects = self.spatial_hash.query(rect)

        sensing_data = []
        for obj in nearby_objects:

            obj_id, obj_team, obj_x, obj_y, obj_size = obj  # (id, team, x, y, size)

            agent_id = obj_id - self.num_obstacles - self.num_foods
            if agent_id == agent_idx and obj_team == self.am.team[agent_idx]:  # agent itself
                continue

            dist_to_obj = get_distance(x_agente.item(), y_agente.item(), obj_x, obj_y)
            pointing_error = \
                ((math.atan2(obj_y - self.am.positions[agent_idx][1], obj_x - self.am.positions[agent_idx][0]) \
                  - self.am.directions[agent_idx]) + np.pi) % (2 * np.pi) - np.pi

            if dist_to_obj < self.am.sensing_range and (-self.am.fov < pointing_error < self.am.fov):

                if obj_team == self.am.team[agent_idx]:  # Compagno di squadra
                    obj_type = 1
                    perceived_size = 0
                elif obj_team == 0:  # Cibo
                    obj_type = -1
                    perceived_size = 1
                    if dist_to_obj <= self.am.size[agent_idx]:
                        self.am.rewards[agent_idx] += 1
                        self.am.energies[agent_idx] += self.energy_gain
                        self.am.size[agent_idx] += 1
                        self.food_eaten(obj_id)  # remove food
                elif obj_team == -2:  # ostacolo
                    obj_type = -1  # same as enemies
                    perceived_size = -1
                    if dist_to_obj <= self.am.size[agent_idx] + obj_size:
                        self.am.rewards[agent_idx] -= 1

                        self.am.energies[agent_idx] = 0
                else:  # Nemico
                    obj_type = -1
                    perceived_size = 1 if self.am.size[agent_idx] > self.am.size[agent_id] else -1 if self.am.size[
                                                                                                          agent_idx] < \
                                                                                                      self.am.size[
                                                                                                          agent_id] else 0

                    if (dist_to_obj <= self.am.size[agent_idx]) and (self.am.size[agent_idx] > self.am.size[agent_id]):
                        self.am.rewards[agent_idx] += 2
                        self.am.energies[agent_idx] += self.energy_gain
                        self.am.size[agent_idx] += 1

                        self.am.energies[agent_id] = 0  # enemy dies
                        self.am.rewards[agent_id] -= 1

                sensing_data.append([pointing_error / np.pi,
                                     (dist_to_obj - self.am.size[agent_idx] - obj_size) / self.am.sensing_range,
                                     perceived_size,
                                     obj_type])

        sensing_data.sort(key=lambda obj: (obj[1]))  # order by distance
        sensing_data = sensing_data[:self.am.sensing_capability]
        sensing_data.sort(key=lambda obj: (obj[0]))  # order by angle

        while len(sensing_data) < self.am.sensing_capability:
            sensing_data.append([0, 0, 0, 0])  # if less than 5 obj in the range

        # sensing_data.insert(0, [self.am.energies[agent_idx] / self.am.start_energy])

        sensing_data_flat = [item for obj in sensing_data for item in obj]
        sensing_data_array = np.array(sensing_data_flat, dtype=np.float32)
        # print(agent_idx, sensing_data)
        self.draw_sensing(agent_idx, sensing_data)
        # self.draw_test((x_agente - 100, y_agente - self.am.size[agent_idx] - 20), f'{math.degrees(sensing_data_array[0]),math.degrees(sensing_data_array[4]), math.degrees(sensing_data_array[8]) }')

        return sensing_data_array

    def draw_sensing(self, agent, data):
        pos = self.am.positions[agent]
        direction = self.am.directions[agent]
        for elem in data:
            x_nemico = pos[0] + elem[1] * self.am.sensing_range * math.cos(elem[0] * np.pi + direction)
            y_nemico = pos[1] + elem[1] * self.am.sensing_range * math.sin(elem[0] * np.pi + direction)

            pygame.draw.line(self.screen, (0, 0, 0), (pos[0], pos[1]), (x_nemico, y_nemico), 1)

    def draw_sensing2(self, pos, theta, distanza, direction):
        x_nemico = pos[0] + distanza * math.cos(theta + direction)
        y_nemico = pos[1] + distanza * math.sin(theta + direction)

        pygame.draw.line(self.screen, (0, 0, 0), (pos[0], pos[1]), (x_nemico, y_nemico), 1)

    def draw_test(self, pos, text):
        font = pygame.font.Font(None, 40)
        text_surface = font.render(text, True, (0, 0, 0))
        self.screen.blit(text_surface, pos)

    def save_gen(self):
        with open('gen.txt', 'w') as f:
            f.write(str(self.generations))

    def load_gen(self):
        with open('gen.txt', 'r') as f:
            self.generations = int(f.readline())

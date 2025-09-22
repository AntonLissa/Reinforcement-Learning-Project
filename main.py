import time

import pygame
import  numpy as np
from Environment import Environment
from config import config
MAP_WIDTH, MAP_HEIGHT = config['map width'], config['map height']
WINDOW_WIDTH, WINDOW_HEIGHT = 1000, 800

pygame.init()

def game_loop(new_sim, testing):
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    map_surface = pygame.Surface((MAP_WIDTH, MAP_HEIGHT))

    environment = Environment(map_surface, testing)
    if not new_sim:  # load weights
        environment.load()

    running = True
    paused = False
    start_time = pygame.time.get_ticks()

    previous_time = pygame.time.get_ticks()

    SIM_STEP_LENGTH = 20 * 1000
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            clock.tick(60)

            map_surface.fill((220, 220, 220))
            current_time = pygame.time.get_ticks()
            dt = (current_time - previous_time) / 1000.0
            previous_time = current_time
            environment.update(dt)
            environment.draw(map_surface)

            elapsed_time = pygame.time.get_ticks() - start_time
            if np.all(environment.am.alive == False) or (elapsed_time >= SIM_STEP_LENGTH):
                environment.reset()
                environment.save()
                start_time = pygame.time.get_ticks()
                previous_time = start_time

            fps = clock.get_fps()

            pygame.display.set_caption(f"generation {environment.generations} - FPS: {fps:.2f} - {elapsed_time}")

            scaled_map = pygame.transform.scale(map_surface, (WINDOW_WIDTH, WINDOW_HEIGHT))
            screen.blit(scaled_map, (0, 0))
        else:
            clock.tick(10)

        if environment.generations >= 2000:
            print('2000 epochs of training reached... stopping simulation')
            running = False

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    action = input('Insert 0 for loading weights \ninsert 1 for a new simulation\ninsert 2 for a PPO vs GA\n')
    if action == '1':
        action2 = input("If you run a new simulation the weights will be overwritten, insert 1 again if you're sure\n")
        if action2 == '1':
            game_loop(True, False)
    elif action == '0':
        game_loop(False, False)
    elif action == '2':
        game_loop(False, True)

import pygame
from environment import SnakeEnv
from pygame.surfarray import array3d

BLACK = pygame.Color(0,0,0)
WHITE = pygame.Color(255,255,255)
RED = pygame.Color(255, 0, 0)
GREEN = pygame.Color(0, 255, 0)

# Initialize the environment
snake_env = SnakeEnv(600, 600)

# speed of the snake
difficulty = 10

fps_controller = pygame.time.Clock()
check_errors = pygame.init()

pygame.display.set_caption("Snake Game")

while True:

    # Human input
    for event in pygame.event.get():

        snake_env.action = snake_env.human_step(event)

    # Check direction
    snake_env.direction = snake_env.change_direction(snake_env.action, snake_env.direction)
    snake_env.snake_pos = snake_env.move(snake_env.direction, snake_env.snake_pos)

    # Check if we ate food
    snake_env.snake_body.insert(0, list(snake_env.snake_pos))
    if snake_env.eat():
        snake_env.score += 1
        snake_env.food_spawn = False
    else:
        snake_env.snake_body.pop()

    # Check if spawn new food
    if not snake_env.food_spawn:

        snake_env.food_pos = snake_env.spawn_food()
    snake_env.food_spawn = True

    # Drawing the snake
    snake_env.game_window.fill(BLACK)
    for pos in snake_env.snake_body:
        pygame.draw.rect(snake_env.game_window, GREEN, pygame.Rect(pos[0], pos[1],10, 10))

    # Drawing of food
    pygame.draw.rect(snake_env.game_window, WHITE, pygame.Rect(snake_env.food_pos[0], snake_env.food_pos[1], 10, 10))

    # Check if end game
    snake_env.game_over()

    # Refresh game screen
    snake_env.display_score(WHITE, 'consolas',20)

    pygame.display.update()
    fps_controller.tick(difficulty)
    img = array3d(snake_env.game_window)
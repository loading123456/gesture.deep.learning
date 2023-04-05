import cv2
import pygame
import torch
from khmt import BreakException, camera, predict
from random import choice
from pygame import Surface
from threading import Thread
from typing import Tuple, Optional

pygame.init()

# Set up the screen
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Define colors
WHITE, BLACK = (255, 255, 255), (0, 0, 0)
RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)

# Set up the ball
BALL_SIZE = 20
ball = pygame.Rect(SCREEN_WIDTH // 2 - BALL_SIZE // 2,
                   SCREEN_HEIGHT // 2 - BALL_SIZE // 2, BALL_SIZE, BALL_SIZE)
ball_vel = [choice([-5, 5]), 5]

# Set up the paddle
PADDLE_WIDTH = 100
PADDLE_HEIGHT = 20
paddle = pygame.Rect(SCREEN_WIDTH // 2 - PADDLE_WIDTH // 2,
                     SCREEN_HEIGHT - PADDLE_HEIGHT - 10, PADDLE_WIDTH, PADDLE_HEIGHT)
PADDLE_SPEED = 8

# Set up the bricks
BRICK_WIDTH = 70
BRICK_HEIGHT = 30
BRICK_PADDING = 10
BRICKS_PER_ROW = 5
NUM_ROWS = 5
brick_color = choice([RED, GREEN, BLUE])
bricks = []

for row in range(NUM_ROWS):
    color = brick_color
    for brick in range(BRICKS_PER_ROW):
        brick_rect = pygame.Rect(
            brick * (BRICK_WIDTH + BRICK_PADDING) + BRICK_PADDING,
            row * (BRICK_HEIGHT + BRICK_PADDING) + BRICK_PADDING + 50,
            BRICK_WIDTH,
            BRICK_HEIGHT
        )
        bricks.append((brick_rect, color))
    brick_color = choice([RED, GREEN, BLUE])

# Set up the score
font = pygame.font.Font(pygame.font.get_default_font(), 36)

clock = pygame.time.Clock()
net = torch.jit.load('./gesture.pt')  # type: ignore
net.eval()


def game_ctrl(_, hand_detect: Tuple[bool, cv2.Mat]):
    global current_image, move
    has_hand, hand_image = hand_detect
    if has_hand:
        conf, predicted = predict(net, hand_image)
        if conf > 0.8:
            move = predicted
        hand_image = cv2.cvtColor(hand_image, cv2.COLOR_BGR2RGB)
        current_image = pygame.surfarray.make_surface(
            hand_image.transpose((1, 0, 2)))
    else:
        current_image = None

    if not running:
        raise BreakException('Game exited...')


current_image: Optional[Surface] = None
score = 0
move = 2
running = True

t = Thread(target=camera, args=(game_ctrl,))
t.start()

while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            t.join()

    # Move the ball
    ball.left += ball_vel[0]
    ball.top += ball_vel[1]

    # Check for collisions with the walls
    if ball.left < 0 or ball.right > SCREEN_WIDTH:
        ball_vel[0] *= -1
    if ball.top < 0:
        ball_vel[1] *= -1

    # Check for collisions with the paddle
    if ball.colliderect(paddle):
        ball_vel[1] *= -1

    # Check for collisions with the bricks
    for brick in bricks:
        if ball.colliderect(brick[0]):
            bricks.remove(brick)
            ball_vel[1] *= -1
            score += 10

    if move == 0 and paddle.left > 0:
        paddle.left -= PADDLE_SPEED
    elif move == 1 and paddle.right < SCREEN_WIDTH:
        paddle.right += PADDLE_SPEED

    # Clear the screen
    screen.fill(WHITE)

    # Draw the ball, paddle, and bricks
    pygame.draw.circle(screen, BLACK, (ball.left + BALL_SIZE //
                                       2, ball.top + BALL_SIZE // 2), BALL_SIZE // 2)
    pygame.draw.rect(screen, BLACK, paddle)
    for brick in bricks:
        pygame.draw.rect(screen, brick[1], brick[0])

    # Draw the score
    score_text = font.render("Score: {}".format(score), True, BLACK)
    screen.blit(score_text, (10, 10))

    # Draw the detection area
    if isinstance(current_image, Surface):
        hand = pygame.transform.scale(current_image, (50, 50))
        screen.blit(hand, (350, 0))

    # Update the screen
    pygame.display.update()

    # Check for game over
    if ball.bottom > SCREEN_HEIGHT:
        # Game over
        # running = False
        # game_over_text = font.render("Game Over!", True, BLACK)
        # screen.blit(game_over_text, (SCREEN_WIDTH // 2 - game_over_text.get_width() // 2, SCREEN_HEIGHT // 2 - game_over_text.get_height() // 2))
        # pygame.display.update()
        # pygame.time.delay(3000)
        ball_vel[1] *= -1

    # Show the FPS in the window title
    fps = int(clock.get_fps())
    pygame.display.set_caption("My Pygame Game (FPS: {})".format(fps))

    # Limit the FPS to 16
    clock.tick(16)

# Quit the game
pygame.quit()

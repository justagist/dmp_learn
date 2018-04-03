import pygame
import time

screen = pygame.display.set_mode((1000, 600))

pygame.display.set_caption('Mouse Position Capture')

running = True
start_record = False
while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.MOUSEBUTTONDOWN:
            start_record = True
        elif event.type == pygame.MOUSEBUTTONUP:
            start_record = False

    if start_record:
        (mouseX, mouseY) = pygame.mouse.get_pos()
        print mouseX, mouseY

    time.sleep(0.5)


# class MouseTracker:

#     def __init__:
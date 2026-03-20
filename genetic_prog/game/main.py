import pygame
import sys
from class_game import LabGame


def main():
    pygame.init()

    # Initialize our custom environment
    env = LabGame(width=21, height=21, cell_size=30)

    # We use this dictionary to map keyboard inputs to our AI Action numbers (0, 1, 2, 3)
    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_RIGHT: 1,
        pygame.K_DOWN: 2,
        pygame.K_LEFT: 3
    }

    running = True
    while running:
        action = None

        # Event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            # If playing manually via keyboard
            if event.type == pygame.KEYDOWN:
                if event.key in key_to_action:
                    action = key_to_action[event.key]
                if event.key == pygame.K_r:  # Press 'R' to reset maze
                    env.reset()

        # Execute action if one was chosen
        if action is not None:
            state, reward, done, kill = env.step(action)
            print(f"Moved to State: {state} | Reward: {reward} | Done: {done}")

            if done and kill:
                print("You reached the goal! Press 'R' to generate a new maze.")
                #exit()
            elif not done and kill:
                print("You reached the wall, game is over")

        # Draw the frame
        env.render()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
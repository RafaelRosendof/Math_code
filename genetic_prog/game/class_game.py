import pygame
import random

# This class gonna be a Labyrinth game
class LabGame:
    def __init__(self, width=30, height=30, cell_size=30):
        self.width = width
        self.height = height
        self.cell_size = cell_size

        self.screen_wid = self.width * self.cell_size
        self.screen_hi = self.height * self.cell_size

        self.screen = pygame.display.set_mode((self.screen_wid, self.screen_hi))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption("Figas Game")

        self.BLACK = (0, 0, 0)  # Walls
        self.WHITE = (255, 255, 255)  # Paths
        self.RED = (255, 0, 0)  # Agent
        self.GREEN = (0, 255, 0)

        self.reset()
        self.maze = self._generate_maze()

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)

        self.maze = self._generate_maze()

        self.agent_pos = [1,1]
        self.goal_pos = [self.width - 2, self.height - 2]
        self.done = False
        self.kill = False

        return self._get_state()

    def _generate_maze(self):
        maze = [ [1 for _ in range(self.width)] for _ in range(self.height) ]

        def carve_passage(cx, cy):
            directions = [ (0,-2), (0,2), (2,0), (-2,0)]

            random.shuffle(directions)
            for dx, dy in directions:
                nx, ny = cx + dx , cy + dy

                if 1 <= ny < self.height -1 and 1 <= nx < self.width - 1 and maze[ny][nx] == 1:
                    maze[cy + dy // 2][cx + dx // 2] = 0
                    maze[ny][nx] = 0
                    carve_passage(nx, ny)
        maze[1][1] = 0
        carve_passage(1,1)
        return maze


    def _get_state(self):
        return tuple(self.agent_pos)

    def step(self, action):
        """
        0 = up, 1 = right, 2 = down, 3 = left
        """

        if self.done:
            return self._get_state(), 0, self.done, self.kill

        if self.kill:
            return self._get_state(), -1, self.done, self.kill

        movements = {
            0: (0,-1), #up
            1: (1,0), #rig
            2: (0,1), #down
            3: (-1,0) #left
        }

        dx, dy = movements.get(action, (0,0))
        new_x = self.agent_pos[0] + dx
        new_y = self.agent_pos[1] + dy

        if self.maze[new_y][new_x] == 0:
            self.agent_pos = [new_x, new_y]

        reward = -0.01

        if self.maze[new_x][new_y] == 1:
            reward = -10.0
            #self.done = True
            self.kill = True
            # return self._get_state(), reward, self.done, self.kill

        if self.agent_pos == self.goal_pos:
            reward = 1.0
            self.done = True

        return self._get_state(), reward, self.done, self.kill

    def render(self):
        self.screen.fill(self.BLACK)

        for y in range(self.height):
            for x in range(self.width):
                if self.maze[y][x] == 0:
                    rect = pygame.Rect(x * self.cell_size, y * self.cell_size, self.cell_size, self.cell_size)
                    pygame.draw.rect(self.screen, self.WHITE, rect)

        goal_rect = pygame.Rect(self.goal_pos[0] * self.cell_size,
                                self.goal_pos[1] * self.cell_size,
                                self.cell_size, self.cell_size
                                )
        pygame.draw.rect(self.screen, self.GREEN, goal_rect)

        agent_rect = pygame.Rect(self.agent_pos[0] * self.cell_size,
                                 self.agent_pos[1] * self.cell_size,
                                 self.cell_size,self.cell_size
                                 )
        pygame.draw.rect(self.screen, self.RED, agent_rect)

        pygame.display.flip()
        self.clock.tick(60)

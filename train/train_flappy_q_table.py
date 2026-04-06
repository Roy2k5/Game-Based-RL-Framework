from env.flappy_bird.flappy_bird import main
from src.q_learning.q_table import QTableAgent

agent = QTableAgent(4, 2, 0.9, 0.5, 0.999, 1e-3, True)
main(agent)

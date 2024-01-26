import json

from loguru import logger

from Network import Network

# inputs:
#   speed
#   size
#   energy
#   dash
#   sensing_side 4 nodes (left, right, top, bottom)

# outputs:
#   up
#   down
#   left
#   right
#   dash
nn = Network([8, 10, 5])

logger.info(json.dumps(nn.as_dict(), indent=2))

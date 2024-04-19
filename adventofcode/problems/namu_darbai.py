from dataclasses import dataclass, field
from queue import SimpleQueue

from typing_extensions import Self


@dataclass
class Node:
	is_explored: bool = False
	relationships: list[Self] = field(default_factory=list)


def dfs_explore(start: Node) -> None:
	start.is_explored = True
	for node in start.relationships:
		if not node.is_explored:
			dfs_explore(node)


def bfs_explore(start: Node) -> None:
	frontier = SimpleQueue()  # type: ignore
	start.is_explored = True
	frontier.put(start)
	while not frontier.empty():
		current = frontier.get()
		for node in current.relationships:
			if node.is_explored:
				continue
			node.is_explored = True
			frontier.put(node)

""" Task 2
Given 6 airports: A, B, C, D, E, F.

Planes fly between them on the following routes:

A -> B
A -> C
B -> C
B -> F
B -> A
F -> D
C -> D
D -> B
D -> E

Find all the routes from A to B. On a single route,
each airport can be visited only once.
"""


def find_routes(start: str, finish: str, graph: list) -> list[list[str]]:
	routes = []

	for route in graph:
		if list(route.keys())[0] == start:
			v = list(route.values())[0]
			r = [list(route.keys())[0]]
			lookup_rec(v, finish, graph, r)

			if r[-1] == finish:
				routes.append(r)

	return routes


def lookup_rec(start: str, finish: str, graph: list, route_path: list):
	if start == finish:
		return route_path.append(finish)

	for route in graph:
		if list(route.keys())[0] == start:
			k = list(route.keys())[0]
			v = list(route.values())[0]
			graph.remove(route)
			route_path.append(k)

			return lookup_rec(v, finish, graph, route_path)


graph = [
	{"A": "B"},
	{"A": "C"},
	{"B": "C"},
	{"B": "F"},
	{"B": "A"},
	{"F": "D"},
	{"C": "D"},
	{"D": "B"},
	{"D": "E"},
]

print(find_routes('A', 'B', graph.copy()))
print(find_routes('A', 'D', graph.copy()))
print(find_routes('F', 'B', graph.copy()))
print(find_routes('A', 'W', graph.copy()))

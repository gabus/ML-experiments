from time import time


def get_element(d: dict, flat_key: str, flat: dict):
	for d_key, value in d.items():
		k = '.'.join([flat_key, d_key]) if flat_key else d_key

		if type(value) != dict:
			flat[k] = value
			continue

		get_element(d[d_key], k, flat)


data = {"a": {"b": {"c": 1, "f": 2}, "g": 3}, "d": 2}
flat_data = {}
get_element(data, '', flat_data)
print(flat_data)


def get_fib(max_counter: int, n1: int = 0, n2: int = 1, counter: int = 0):
	if max_counter <= counter:
		return n1

	return get_fib(max_counter, n2, n1 + n2, counter + 1)


def fib_recursive(n: int) -> int:
	if n < 1:
		raise ValueError
	if n == 1:
		return 0
	if n == 2:
		return 1
	return fib_recursive(n - 2) + fib_recursive(n - 1)


a = time()
print(get_fib(997))
print("done in: {}".format(time() - a))

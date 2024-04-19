""" Task 3
For a pair of words, find the shortest path by changing one letter
at a time. For example,

find_path(dictionary, "bun", "van")

should return ["bun", "ban", "van"] for such a dictionary:
["bun", "ban", "van", "dam", "gun"]

An English dictionary is first provided as a list of words.
All words in the path must come from this dictionary.
"""


def find_path(start: str, finish: str, dictionary: list[str]) -> list[str]:
	pass


dictionary = ['bun', 'ban', 'van', 'dam', 'gun']
assert find_path('bun', 'van', dictionary) == ['bun', 'ban', 'van']

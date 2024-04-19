""" Task 1
Given an array of integers nums and an integer target,
return indices of the two numbers such that they add up to target.

You may assume that each input would have exactly one solution,
and you may not use the same element twice.

You can return the answer in any order.
"""


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
	for i, num_i in enumerate(nums):
		for j, num_j in enumerate(nums):
			if i == j:
				continue

			if num_i + num_j == target:
				return i, j


def two_sum_2(nums: list[int], target: int) -> tuple[int, int] | None:
	seen: dict[int, int] = {}
	for i in range(len(nums)):
		num = target - nums[i]
		if num in seen:
			return seen[num], i
		seen[nums[i]] = i
	return None


assert two_sum_2([1, 2, 32, 4, 5, 0, 56], 2) == (1, 5)
assert two_sum_2([1, 2, 32, 4, 5, 56], 5) == (0, 3)
assert two_sum_2([1, 2, 32, 4, 5, 56], 8) == None
assert two_sum_2([1, 2, 32, 4, 5, 56], 37) == (2, 4)
assert two_sum_2([1, 2, 32, 4, 5, 56], 57) == (0, 5)

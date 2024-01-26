import plotly.graph_objects as go
from numpy import arange


def objective(x: float):
	return x ** 2.0


# define range for input
r_min, r_max = -5.0, 5.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max, 0.1)
# compute targets
results = objective(inputs)

# df = px.data.gapminder().query("continent=='Oceania'")
# logger.info(df)


fig = go.Figure()

fig.add_trace(go.Scatter(
	x=inputs,
	y=results,
	name='parabola',
	line=dict(color='firebrick', width=3))
)

fig.add_trace(go.Scatter(
	x=[1, 2, 3],
	y=[1, 2, 3],
	name='line',
	line=dict(color='green', width=3))
)

fig.show()


# # create a scatter plot of input vs result
# pyplot.plot(inputs, results)
# pyplot.plot([-2, 2], [0, 2], color='g')
# pyplot.scatter(0, 0, color='r')
# # show the plot
# pyplot.show()
#

# approximate slope of the function objective at x
def find_slope(x: float):
	h = 0000.1

	# positive - going up, negative - going down
	slope_direction = objective(x + h) - objective(x)

	# If the slope is very steep, make bigger intervals for the next point check
	# When slope evens out, that means you're close the minimum, so take more samples
	slope_steepness = slope_direction / h

from collections import deque

import plotly.graph_objects as go
from dash import html, Dash, callback, Output, Input, dcc
from numpy import arange

from Slope import Slope


def objective(x: float):
	# return x ** 2.0
	return 0.2 * pow(x, 4) + 0.1 * pow(x, 3) - pow(x, 2) + 2


def initial_parabola():
	return go.Scatter(
		x=inputs,
		y=results,
		name='parabola',
		line=dict(color='blue', width=3)
	)


# define range for input
r_min, r_max = -3.0, 3.0
# sample input range uniformly at 0.1 increments
inputs = arange(r_min, r_max, 0.1)
results = objective(inputs)

X = deque(maxlen=20)
Y = deque(maxlen=20)

slope = Slope(objective)


@callback(Output('graph', 'figure'), Input('graph-interval', 'n_intervals'))
def update_graph(n_intervals):
	# X.append(n_intervals + 2)
	# Y.append(n_intervals + 2)

	parabola = initial_parabola()

	line = slope.get_next_solve_x()

	# line_frame = go.Scatter(
	# 	x=list(line['x']),
	# 	y=list(line['y']),
	# 	name='line',
	# 	line=dict(color='firebrick', width=3),  
	# 	mode='lines+markers',
	# )

	p = go.Scatter(
		x=slope.all_x,
		y=slope.all_y,
		name='points',
		line=dict(color='green', width=8),
		mode='markers',
	)

	graph_margin = 5
	layout = go.Layout(
		xaxis=dict(range=[min(inputs) - graph_margin, max(inputs) + graph_margin]),
		yaxis=dict(range=[min(results) - graph_margin, max(results) + graph_margin]),
	)

	return {
		'data': [parabola, p],
		# 'layout': layout
	}


app = Dash(__name__)

layout = html.Div(
	[
		# dcc.Slider(-5, 10, 1, value=-3),
		html.Div([
			dcc.Graph(id="graph", animate=True),
			dcc.Interval(id='graph-interval', interval=1000, n_intervals=0, max_intervals=-1),
		]),
	],
)

app.layout = layout

if __name__ == '__main__':
	app.run(debug=True)

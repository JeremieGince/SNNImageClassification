import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
# import statsmodels.api as sm
# from tabulate import tabulate
import numpy as np

plot_layout = dict(
	paper_bgcolor='rgb(243, 243, 243)',
	plot_bgcolor='rgb(243, 243, 243)',
	legend=dict(font_size=35, borderwidth=3, ),
	xaxis=dict(
		tickfont_size=28,
		zeroline=False,
		showgrid=False,
		title_font_size=24,
		showline=True,
		linewidth=4,
		linecolor='black',
	),
	yaxis=dict(
		showgrid=False,
		tickfont_size=42,
		zeroline=False,
		title_font_size=40,
		showline=True,
		linewidth=4,
		linecolor='black',
	)
)

dict_param_name = {
	'hidden_layer_type': 'Dynamique',
	"use_recurrent_connection": "Connections récurrentes",
	"to_spikes_use_periods": 'Temps en période',
	"n_hidden_neurons": 'Taille de la couche cachée'
}


def load_results(file_path='tr_data/results.csv') -> pd.DataFrame:
	"""
	Loads the results from the given file path.
	"""
	return pd.read_csv(file_path, index_col=0)


def _plot_bar_result(figure: go.Figure, results: pd.DataFrame, dataset: str, y_axis: str, **kwargs):
	"""
	Plots a histogram of the given results.
	"""
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset]
	dataset_results = dataset_results.sort_values(
		by=['hidden_layer_type', "use_recurrent_connection", "to_spikes_use_periods", "n_hidden_neurons"],
		ignore_index=True)
	y_data = dataset_results[y_axis]*100
	y_data = y_data.tolist()
	xlabel = [
		f'{dataset_results.loc[i, "hidden_layer_type"].split(".")[-1]}<br>'
		f'REC {"[✓]" if dataset_results.loc[i, "use_recurrent_connection"] else "[X]"}<br>'
		f'P {"[✓]" if dataset_results.loc[i, "to_spikes_use_periods"] else "[X]"}<br>'
		f'HN:{dataset_results.loc[i, "n_hidden_neurons"]}<br>'
		for i in range(dataset_results.shape[0])

	]
	figure.add_trace(go.Bar(
		y=y_data,
		x=xlabel,
		name=y_axis,
		text=list(map(lambda a: round(a, 2), y_data)),
		textposition='auto',
		textangle=90,
		textfont_size=32,
		**kwargs
	)
	)
	return figure


def plot_bar_result(results: pd.DataFrame, dataset_name: str, list_col_names: list):
	"""
	Plots a histogram of the given results.
	"""
	figure = go.Figure()
	palette = sns.color_palette("rocket", len(list_col_names))
	for i, y_axis in enumerate(list_col_names):
		color = f'rgb{tuple(map(lambda a: int(a * 255), palette[i]))}'
		figure = _plot_bar_result(figure, results, dataset_name, y_axis, marker_color=color)
	figure.update_layout(plot_layout)
	figure.update_layout(
		barmode='group',
		legend=dict(
			x=0.98,
			y=0.95,
			xanchor='right',
			yanchor='top',
			borderwidth=3,
		),
		# xaxis_tickangle=70,
		uniformtext=dict(mode="hide", minsize=18),
	)
	figure.update_xaxes(
		ticks="outside",
		tickwidth=4,
		tickcolor='black'
	)
	figure.update_yaxes(
		title=dict(text='Accuracy [%]'),
		range=[0, 100],
	)
	return figure


def make_data_for_box_plot(results: pd.DataFrame, dataset_name: str, ydata: str):
	"""
	Returns the data for the box plot.
	"""
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset_name]
	y_data = dataset_results[ydata]*100
	box_plot_data = pd.DataFrame()
	for param in dict_param_name.keys():
		for param_value in dataset_results[param].unique():
			if param == 'hidden_layer_type':
				param_value_name = param_value.split('.')[-1]
			elif param == 'use_recurrent_connection':
				param_value_name = 'REC [✓]' if param_value else 'REC [X]'
			elif param == 'to_spikes_use_periods':
				param_value_name = 'P [✓]' if param_value else 'P [X]'
			elif param == 'n_hidden_neurons':
				param_value_name = f'HN {param_value}'
			else:
				param_value_name = param_value
			box_plot_data[f'{param_value_name}'] = y_data[dataset_results[param] == param_value].tolist()
	return box_plot_data


def box_plot_accuracy(results: pd.DataFrame, dataset_name: str):
	box_plot_data = make_data_for_box_plot(results, dataset_name, 'test_accuracy')
	figure = go.Figure()
	palette = sns.color_palette("tab10", 4)
	Legendg = ['Dynamique', 'Recurrence', 'Période', 'N neurones']
	for i in range(box_plot_data.shape[1]):
		color = f'rgb{tuple(map(lambda a: int(a * 255), palette[int(i / 2)]))}'
		figure.add_trace(
			go.Box(
				y=box_plot_data.iloc[:, i],
				name=box_plot_data.columns[i],
				boxpoints='all',
				pointpos=0,
				marker=dict(
					color=color,
					size=12
					),
				legendgroup=f'{Legendg[i//2]}',
			)
		)
	figure.update_layout(plot_layout)
	figure.update_xaxes(
		ticks="outside",
		tickwidth=4,
		tickcolor='black',
		tickfont_size=38,

	)
	figure.update_yaxes(
		title=dict(text='Accuracy [%]'),
		range=[0, 100],
	)
	return figure


def make_data_for_stat(results: pd.DataFrame, dataset_name: str, ydata: str):
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset_name]
	y_data = dataset_results[ydata]
	stat_data = pd.DataFrame()
	stat_data['to_spikes_use_periods'] = dataset_results['to_spikes_use_periods'].map({True: 1, False: 0}) #If True replace by 1, else replace by 0
	stat_data['hidden_layer_type'] = dataset_results['hidden_layer_type'].map({'LayerType.LIF': 0, 'LayerType.ALIF': 1})
	stat_data['use_recurrent_connection'] = dataset_results['use_recurrent_connection'].map({True: 1, False: 0})
	stat_data['n_hidden_neurons'] = dataset_results['n_hidden_neurons'].map({100: 0, 200: 1})
	return stat_data, y_data.tolist()


# def statistical_analysis_model(results: pd.DataFrame, dataset_name: str):
# 	X, Y = make_data_for_stat(results, dataset_name, 'test_accuracy')
# 	X2 = sm.add_constant(X)
# 	return sm.OLS(Y, X2).fit()


def make_pairwise_data(results: pd.DataFrame, dataset_name: str, param_name: str,ydata_name: str):
	dataset_results = results[results['dataset_id'] == 'DatasetId.' + dataset_name]
	list_params = list(dict_param_name.keys())
	list_params.remove(param_name)
	list_params = [param_name] + list_params
	dataset_results = dataset_results.sort_values(by=list_params, ignore_index=True)
	pairwise_data = pd.DataFrame()
	for index, param in enumerate(dataset_results[param_name].unique()):
		ydata = dataset_results[dataset_results[param_name] == param]
		if index == 0:
			pairwise_data[list_params[1:]] = ydata[list_params[1:]]
		pairwise_data[f'{param_name}={param}'] = ydata[ydata_name].tolist()
	return pairwise_data.iloc[:, [-2, -1]]


def pairwise_comparison(results: pd.DataFrame, dataset_name: str):
	list_params = list(dict_param_name.keys())
	list_mean = []
	list_std = []
	for param_name in list_params:
		pairwise_data = make_pairwise_data(results, dataset_name, param_name, 'test_accuracy')
		diff = pairwise_data.diff(axis=1).iloc[:, -1].to_numpy()
		list_mean.append(np.abs(np.mean(diff)))
		list_std.append(np.std(diff)/3)
	fig = go.Figure()
	fig.add_trace(
		go.Bar(
			x=list(map(lambda a: dict_param_name[a],list_params)),
			y=list_mean,
			marker_color='crimson',
		)
	)
	fig.update_layout(plot_layout)
	fig.update_xaxes(
		tickfont_size=38,
	)
	fig.update_yaxes(
		title=dict(text='Différence couplée moyenne'),
		range=[0, 0.45],
	)
	return fig


if __name__ == '__main__':
	result = load_results('tr_results/results.csv')
	# box_plot_accuracy(result, 'MNIST').show()
	# box_plot_accuracy(result, 'FASHION_MNIST').show()
	plot_bar_result(result, 'MNIST', ['test_accuracy']).show()
	# plot_bar_result(result, 'FASHION_MNIST', ['test_accuracy', 'val_accuracy']).show()
	# print(statistical_analysis_model(result, 'FASHION_MNIST').summary())
	# pairwise_comparison(result, 'MNIST').show()
	# pairwise_comparison(result, 'FASHION_MNIST').show()
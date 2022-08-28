import streamlit as st

import ipywidgets as wg
from IPython.display import display, HTML

import pandas as pd
import numpy as np

import sys, os
import time
import datetime
import re

import plotly as py
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.colors import n_colors
from plotly.subplots import make_subplots

st.set_page_config(
    layout="wide",
    page_title="Plotly & Widgets",
    page_icon="📊",
    menu_items={
        "Get help": "https://www.perfectlynormal.co.uk/",
        "About": "##### This was created to demo the Python libraries of Plotly and IPyWidgets, and show how they can be combined to create interactive output in Python notebooks."
    }
)

streamlit_style = """
			<style>
			a, a:link, a:visited, a:active {
                color: black;
                text-decoration: none;
			}
            a:hover {
                color: black;
                text-decoration: underline;
            }
			</style>
			"""
st.markdown(streamlit_style, unsafe_allow_html=True)

def parse_title(title):
    title = title.replace("(", "").replace(")", "").split(" ")
    title = [word.lower() for word in title if word != "&"]
    title = "#" + "-".join(title)
    return title

all_titles = [
    "Scatter plots",
    "Line charts",
    "Bar charts",
    "Box plots",
    "Violin plots",
    "Histograms",
    "Heatmaps",
    "Treemaps",
    "Marginal plots",
    "Scatterplot matrix",
    "Tables"
]

num_blocks = [10, 5, 9, 4, 4, 5, 1, 3, 2, 3]

with st.sidebar:
    for title in all_titles:
        st.markdown(f"[{title}]({parse_title(title)})", unsafe_allow_html=True)








code_1 = """df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()"""

df = px.data.iris()

fig_1 = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])

code_2 = """N = 100
random_x = np.linspace(0, 1, N)
random_y = [np.random.randn(N) + 5, np.random.randn(N), np.random.randn(N) - 5]

trace_info = ["markers", "lines+markers", "lines"]
fig = go.Figure(data=[go.Scatter(x=random_x, y=y, mode=t, name=t) for (y, t) in zip(random_y, trace_info)])

fig.show()"""

N = 100
random_x = np.linspace(0, 1, N)
random_y = [np.random.randn(N) + 5, np.random.randn(N), np.random.randn(N) - 5]

trace_info = ["markers", "lines+markers", "lines"]
fig_2 = go.Figure(data=[go.Scatter(x=random_x, y=y, mode=t, name=t) for (y, t) in zip(random_y, trace_info)])

code_3 = """fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker_size=[40, 60, 80, 100],
    marker_color=[0, 1, 2, 3]))

fig.show()"""

fig_3 = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker_size=[40, 60, 80, 100],
    marker_color=[0, 1, 2, 3]))


code_4 = """t = np.linspace(0, 10, 100)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=t, y=np.sin(t),
    name='sin',
    marker_color='rgba(152, 0, 0, .8)'
))

fig.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='cos',
    marker_color='rgba(255, 182, 193, .9)'
))

# Set options common to all traces with fig.update_traces
fig.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig.update_layout(title='Styled Scatter',
                  yaxis_zeroline=True, 
                  xaxis_zeroline=False)


fig.show()"""

t = np.linspace(0, 10, 100)

fig_4 = go.Figure()

fig_4.add_trace(go.Scatter(
    x=t, y=np.sin(t),
    name='sin',
    marker_color='rgba(152, 0, 0, .8)'
))

fig_4.add_trace(go.Scatter(
    x=t, y=np.cos(t),
    name='cos',
    marker_color='rgba(255, 182, 193, .9)'
))

# Set options common to all traces with fig_4.update_traces
fig_4.update_traces(mode='markers', marker_line_width=2, marker_size=10)
fig_4.update_layout(title='Styled Scatter',
                  yaxis_zeroline=True, 
                  xaxis_zeroline=False)

code_5 = """data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

fig = go.Figure(data=go.Scatter(x=data['Postal'],
                                y=data['Population'],
                                mode='markers+text',
                                marker_color=data['Population'],
                                hovertext=data['State']),
               layout=go.Layout(title="Population of USA States"))

fig.show()"""

data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

fig_5 = go.Figure(data=go.Scatter(x=data['Postal'],
                                y=data['Population'],
                                mode='markers+text',
                                marker_color=data['Population'],
                                hovertext=data['State']),
               layout=go.Layout(title="Population of USA States"))

code_6 = """fig = go.Figure(data=go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500),
        colorscale='Viridis',
        showscale=True,
    )
))
fig.show()"""

fig_6 = go.Figure(data=go.Scatter(
    y = np.random.randn(500),
    mode='markers',
    marker=dict(
        size=16,
        color=np.random.randn(500),
        colorscale='Viridis',
        showscale=True,
    )
))

code_7 = """df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex",
                 width=800, height=400)

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)

fig.show()"""

df = px.data.tips()
fig_7 = px.scatter(df, x="total_bill", y="tip", facet_col="sex",
                 width=800, height=400)

fig_7.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)

code_8 = """df = px.data.gapminder().query("year==2007")

fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 log_x=True, size_max=60)

fig.show()"""

df = px.data.gapminder().query("year==2007")

fig_8 = px.scatter(df, x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 log_x=True, size_max=60)

code_9 = """df = px.data.gapminder()

fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)

fig.update_layout(xaxis=dict(gridcolor="white", gridwidth=2),
                  yaxis=dict(gridcolor="white", gridwidth=2),
                  paper_bgcolor='rgb(235, 235, 235)',
                  plot_bgcolor='rgb(235, 235, 235)')
fig.show()"""

df = px.data.gapminder()

fig_9 = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)

fig_9.update_layout(xaxis=dict(gridcolor="white", gridwidth=2),
                  yaxis=dict(gridcolor="white", gridwidth=2),
                  paper_bgcolor='rgb(235, 235, 235)',
                  plot_bgcolor='rgb(235, 235, 235)')



code_11 = """df = px.data.gapminder()

df = df[df.continent.isin(["Americas"])]

fig = px.line(df, x="year", y="lifeExp", color='country')
fig.show()"""

df = px.data.gapminder()

df = df[df.continent.isin(["Americas"])]

fig_11 = px.line(df, x="year", y="lifeExp", color='country')

code_12 = """month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

y_data = [high_2000, low_2000, high_2007, low_2007, high_2014, low_2014]
names = ["High 2000", "Low 2000", "High 2007", "Low 2007", "High 2014", "Low 2014"]
colors = ["firebrick", "royalblue"]
dash_type = ["solid", "dash", "dot"]

fig = go.Figure(
    data=[
        go.Scatter(x=month, y=y, name=name, line=dict(color=color, width=4, dash=dash))
        for (y, name, color, dash) in zip(y_data, names, colors*3, dash_type*2)
    ],
    layout=go.Layout(
        title='Average High and Low Temperatures in New York',
        xaxis_title='Month',
        yaxis_title='Temperature (degrees F)'
    )
)

fig.show()"""

month = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
         'August', 'September', 'October', 'November', 'December']
high_2000 = [32.5, 37.6, 49.9, 53.0, 69.1, 75.4, 76.5, 76.6, 70.7, 60.6, 45.1, 29.3]
low_2000 = [13.8, 22.3, 32.5, 37.2, 49.9, 56.1, 57.7, 58.3, 51.2, 42.8, 31.6, 15.9]
high_2007 = [36.5, 26.6, 43.6, 52.3, 71.5, 81.4, 80.5, 82.2, 76.0, 67.3, 46.1, 35.0]
low_2007 = [23.6, 14.0, 27.0, 36.8, 47.6, 57.7, 58.9, 61.2, 53.3, 48.5, 31.0, 23.6]
high_2014 = [28.8, 28.5, 37.0, 56.8, 69.7, 79.7, 78.5, 77.8, 74.1, 62.6, 45.3, 39.9]
low_2014 = [12.7, 14.3, 18.6, 35.5, 49.9, 58.0, 60.0, 58.6, 51.7, 45.2, 32.2, 29.1]

y_data = [high_2000, low_2000, high_2007, low_2007, high_2014, low_2014]
names = ["High 2000", "Low 2000", "High 2007", "Low 2007", "High 2014", "Low 2014"]
colors = ["firebrick", "royalblue"]
dash_type = ["solid", "dash", "dot"]

fig_12 = go.Figure(
    data=[
        go.Scatter(x=month, y=y, name=name, line=dict(color=color, width=4, dash=dash))
        for (y, name, color, dash) in zip(y_data, names, colors*3, dash_type*2)
    ],
    layout=go.Layout(
        title='Average High and Low Temperatures in New York',
        xaxis_title='Month',
        yaxis_title='Temperature (degrees F)'
    )
)

code_13 = """fig = go.Figure(
    data=[
        go.Scatter(
            x=list(range(15)),
            y=[10, 20, "X", 15, 10, 5, 15, "A", 20, 10, 10, 15, 25, 20, 10],
            name = '<b>No</b> Gaps',
            connectgaps=True
        ),
        go.Scatter(
            x=list(range(15)),
            y=[5, 15, np.NaN, 10, 5, 0, 10, None, 15, 5, 5, 10, 20, 15, 5],
            name='Gaps')
    ]
)

fig.show()"""

fig_13 = go.Figure(
    data=[
        go.Scatter(
            x=list(range(15)),
            y=[10, 20, "X", 15, 10, 5, 15, "A", 20, 10, 10, 15, 25, 20, 10],
            name = '<b>No</b> Gaps',
            connectgaps=True
        ),
        go.Scatter(
            x=list(range(15)),
            y=[5, 15, np.NaN, 10, 5, 0, 10, None, 15, 5, 5, 10, 20, 15, 5],
            name='Gaps')
    ]
)


code_14 = """x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 1])

fig = go.Figure(data=[go.Scatter(x=x, y=y, name="linear", line_shape="linear"),
                      go.Scatter(x=x, y=y+5, name="spline", line_shape="spline"),
                      go.Scatter(x=x, y=y+10, name="hv", line_shape="hv")])

fig.update_traces(hoverinfo='name', mode='lines+markers')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

fig.show()"""

x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 1])

fig_14 = go.Figure(data=[go.Scatter(x=x, y=y, name="linear", line_shape="linear"),
                      go.Scatter(x=x, y=y+5, name="spline", line_shape="spline"),
                      go.Scatter(x=x, y=y+10, name="hv", line_shape="hv")])

fig_14.update_traces(hoverinfo='name', mode='lines+markers')
fig_14.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

code_15 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')

fig = go.Figure([
    go.Scatter(
        name='Measurement',
        x=df['Time'],
        y=df['10 Min Sampled Avg'],
        mode='lines',
        line_color='rgb(31, 119, 180)'
    ),
    go.Scatter(
        name='Upper Bound',
        x=df['Time'],
        y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
        mode='lines',
        marker_color="#444",
        line_width=0
    ),
    go.Scatter(
        name='Lower Bound',
        x=df['Time'],
        y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
        marker_color="#444",
        line_width=0,
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
])

fig.update_traces(showlegend=False)
fig.update_layout(hovermode="x")
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')

fig_15 = go.Figure([
    go.Scatter(
        name='Measurement',
        x=df['Time'],
        y=df['10 Min Sampled Avg'],
        mode='lines',
        line_color='rgb(31, 119, 180)'
    ),
    go.Scatter(
        name='Upper Bound',
        x=df['Time'],
        y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
        mode='lines',
        marker_color="#444",
        line_width=0
    ),
    go.Scatter(
        name='Lower Bound',
        x=df['Time'],
        y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
        marker_color="#444",
        line_width=0,
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
])

fig_15.update_traces(showlegend=False)
fig_15.update_layout(hovermode="x")

code_16 = """data_canada = px.data.gapminder()
fig = px.bar(data_canada[data_canada.country == "Canada"], x='year', y='pop', hover_data=["lifeExp"])
fig.show()"""

data_canada = px.data.gapminder()
fig_16 = px.bar(data_canada[data_canada.country == "Canada"], x='year', y='pop', hover_data=["lifeExp"])

code_17 = """long_df = px.data.medals_long()
display(long_df)

px.bar(long_df, x="nation", y="count", color="medal", barmode="stack").show()"""

long_df = px.data.medals_long()

fig_17 = px.bar(long_df, x="nation", y="count", color="medal", barmode="stack")

code_18 = """data = px.data.gapminder()

data_canada = data[data.country == 'Canada']
fig = px.bar(data_canada, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',
             labels={'pop':'population of Canada'}, height=400)
fig.show()"""

data = px.data.gapminder()

data_canada = data[data.country == 'Canada']
fig_18 = px.bar(data_canada, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',
             labels={'pop':'population of Canada'}, height=400)

code_19 = """df = px.data.tips()
fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group",
             facet_row="time", facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],
                              "time": ["Lunch", "Dinner"]})
fig.show()"""

df = px.data.tips()
fig_19 = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group",
             facet_row="time", facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"],
                              "time": ["Lunch", "Dinner"]})

code_20 = """df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df, y='pop', x='country', text='pop')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()"""

df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig_20 = px.bar(df, y='pop', x='country', text='pop')
fig_20.update_traces(texttemplate='%{text:.2s}', textposition='outside')

code_21 = """fig = go.Figure(
    data=[
        go.Bar(x=list(range(1995, 2013)),
               y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263, 350, 430, 474, 526, 488, 537, 500, 439],
               name='Rest of world',
               marker_color='rgb(55, 83, 109)'),
        go.Bar(x=list(range(1995, 2013)),
               y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270, 299, 340, 403, 549, 499],
               name='China',
               marker_color='rgb(26, 118, 255)')],
               
    layout=go.Layout(
        title='US Export of Plastic Scrap',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='USD (millions)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,      # gap between bars of adjacent location coordinates
        bargroupgap=0.1   # gap between bars of the same location coordinate
    ))

fig.show()"""

fig_21 = go.Figure(
    data=[
        go.Bar(x=list(range(1995, 2013)),
               y=[219, 146, 112, 127, 124, 180, 236, 207, 236, 263, 350, 430, 474, 526, 488, 537, 500, 439],
               name='Rest of world',
               marker_color='rgb(55, 83, 109)'),
        go.Bar(x=list(range(1995, 2013)),
               y=[16, 13, 10, 11, 28, 37, 43, 55, 56, 88, 105, 156, 270, 299, 340, 403, 549, 499],
               name='China',
               marker_color='rgb(26, 118, 255)')],
               
    layout=go.Layout(
        title='US Export of Plastic Scrap',
        xaxis_tickfont_size=14,
        yaxis=dict(
            title='USD (millions)',
            titlefont_size=16,
            tickfont_size=14,
        ),
        legend=dict(
            x=0,
            y=1.0,
            bgcolor='rgba(255, 255, 255, 0)',
            bordercolor='rgba(255, 255, 255, 0)'
        ),
        barmode='group',
        bargap=0.15,      # gap between bars of adjacent location coordinates
        bargroupgap=0.1   # gap between bars of the same location coordinate
    ))

code_22 = """df = px.data.tips()
fig = px.box(df, x="sex", y="total_bill")
fig.show()"""

df = px.data.tips()
fig_22 = px.box(df, x="sex", y="total_bill")

code_23 = """x0 = np.random.randn(50)
x1 = np.random.randn(50) + 2

fig = go.Figure(data=[go.Box(x=x0), go.Box(x=x1)])

fig.update_layout(height=250, width=600, margin=dict(l=20,r=20,b=20,t=20))
fig.show()"""

x0 = np.random.randn(50)
x1 = np.random.randn(50) + 2

fig_23 = go.Figure(data=[go.Box(x=x0), go.Box(x=x1)])

fig_23.update_layout(height=250, width=600, margin=dict(l=20,r=20,b=20,t=20))

code_24 = """df = px.data.tips()
fig = px.box(df, x="time", y="total_bill", color="smoker", points="all", notched=True)
fig.show()"""

df = px.data.tips()
fig_24 = px.box(df, x="time", y="total_bill", color="smoker", points="all", notched=True)

code_25 = """df = px.data.tips()
fig = px.violin(df, y="total_bill", box=True,
                points='all')
fig.show()"""

df = px.data.tips()
fig_25 = px.violin(df, y="total_bill", box=True,
                points='all')

code_26 = """df = px.data.tips()
fig = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all",
          hover_data=df.columns)
fig.show()"""

df = px.data.tips()
fig_26 = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all",
          hover_data=df.columns)

code_27 = """df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/violin_data.csv")

fig = go.Figure()

fig.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'Yes' ],
                        y=df['total_bill'][ df['smoker'] == 'Yes' ],
                        legendgroup='Yes', scalegroup='Yes', name='Yes',
                        side='negative',
                        line_color='blue')
             )
fig.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'No' ],
                        y=df['total_bill'][ df['smoker'] == 'No' ],
                        legendgroup='No', scalegroup='No', name='No',
                        side='positive',
                        line_color='orange')
             )
fig.update_traces(meanline_visible=True)
fig.update_layout(violingap=0, violinmode='overlay', height=300, width=600, margin_t=20)
fig.show()"""

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/violin_data.csv")

fig_27 = go.Figure()

fig_27.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'Yes' ],
                        y=df['total_bill'][ df['smoker'] == 'Yes' ],
                        legendgroup='Yes', scalegroup='Yes', name='Yes',
                        side='negative',
                        line_color='blue')
             )
fig_27.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'No' ],
                        y=df['total_bill'][ df['smoker'] == 'No' ],
                        legendgroup='No', scalegroup='No', name='No',
                        side='positive',
                        line_color='orange')
             )
fig_27.update_traces(meanline_visible=True)
fig_27.update_layout(violingap=0, violinmode='overlay', height=300, width=600, margin_t=20)

code_28 = """data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
            (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

fig = go.Figure()
for data_line, color in zip(data, colors):
    fig.add_trace(go.Violin(x=data_line, line_color=color))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
fig.show()"""

data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
            (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

fig_28 = go.Figure()
for data_line, color in zip(data, colors):
    fig_28.add_trace(go.Violin(x=data_line, line_color=color))

fig_28.update_traces(orientation='h', side='positive', width=3, points=False)
fig_28.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)

code_29 = """df = px.data.tips()

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=df["total_bill"], nbinsx=80), 1, 1)
fig.add_trace(go.Histogram(x=df["day"]), 1, 2)

fig.show()"""

df = px.data.tips()

fig_29 = make_subplots(rows=1, cols=2)

fig_29.add_trace(go.Histogram(x=df["total_bill"], nbinsx=80), 1, 1)
fig_29.add_trace(go.Histogram(x=df["day"]), 1, 2)

code_30 = """x0 = np.random.randn(500)
x1 = np.random.randn(500) + 1

fig = go.Figure(data=[go.Histogram(x=x0, marker_color='#EB89B5', name="Pink"), 
                      go.Histogram(x=x1, marker_color='#330C73', name="Purple")])

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.6)

fig.show()"""

x0 = np.random.randn(500)
x1 = np.random.randn(500) + 1

fig_30 = go.Figure(data=[go.Histogram(x=x0, marker_color='#EB89B5', name="Pink"), 
                      go.Histogram(x=x1, marker_color='#330C73', name="Purple")])

fig_30.update_layout(barmode='overlay')
fig_30.update_traces(opacity=0.6)

code_31 = """df = px.data.tips()
fig = px.histogram(df, x="total_bill", y="tip", color="sex",
                   marginal="box",
                   hover_data=df.columns)
fig.update_layout(height=350, width=750, margin=dict(l=20,r=20,t=20,b=20))
fig.show()"""

df = px.data.tips()
fig_31 = px.histogram(df, x="total_bill", y="tip", color="sex",
                   marginal="box",
                   hover_data=df.columns)
fig_31.update_layout(height=350, width=750, margin=dict(l=20,r=20,t=20,b=20))

code_32 = """x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

hist_data = [x1, x2, x3, x4]
group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()"""

x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

hist_data = [x1, x2, x3, x4]
group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

fig_32 = ff.create_distplot(hist_data, group_labels, bin_size=.2)

code_33 = """fig = go.Figure(
    data=go.Heatmap(
        z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        y=['Morning', 'Afternoon', 'Evening'],
        hoverongaps = False),
    layout=go.Layout(width=750,height=400,margin_t=20)
)
fig.show()"""

fig_33 = go.Figure(
    data=go.Heatmap(
        z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        y=['Morning', 'Afternoon', 'Evening'],
        hoverongaps = False),
    layout=go.Layout(width=750,height=400,margin_t=20)
)

z = [[.1, .3, .5, .7],
     [1, .8, .6, .4],
     [.6, .4, .2, .0],
     [.9, .7, .5, .3]]

fig_34 = ff.create_annotated_heatmap(z, colorscale='Viridis')
fig_34.update_layout(width=750, height=400, margin_t=20)

code_34 = """z = [[.1, .3, .5, .7],
     [1, .8, .6, .4],
     [.6, .4, .2, .0],
     [.9, .7, .5, .3]]

fig = ff.create_annotated_heatmap(z, colorscale='Viridis')
fig.update_layout(width=750, height=400, margin_t=20)
fig.show()"""

code_35 = """z = [[.1, .3, .5],
     [1.0, .8, .6],
     [.6, .4, .2]]

x = ['Team A', 'Team B', 'Team C']
y = ['Game Three', 'Game Two', 'Game One']

z_text = [['Win', 'Lose', 'Win'],
          ['Lose', 'Lose', 'Win'],
          ['Win', 'Win', 'Lose']]

fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

fig.show()"""

z = [[.1, .3, .5],
     [1.0, .8, .6],
     [.6, .4, .2]]

x = ['Team A', 'Team B', 'Team C']
y = ['Game Three', 'Game Two', 'Game One']

z_text = [['Win', 'Lose', 'Win'],
          ['Lose', 'Lose', 'Win'],
          ['Win', 'Win', 'Lose']]

fig_35 = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

code_36 = """df = px.data.tips()

fig = px.density_heatmap(df, x="total_bill", y="tip", histfunc="avg")

fig.show()"""

df = px.data.tips()

fig_36 = px.density_heatmap(df, x="total_bill", y="tip", histfunc="avg")

code_37 = """df = px.data.tips()

fig = px.density_heatmap(df, x="total_bill", y="tip", facet_row="sex", facet_col="smoker")

fig.show()"""

df = px.data.tips()

fig_37 = px.density_heatmap(df, x="total_bill", y="tip", facet_row="sex", facet_col="smoker")

code_38 = """df = px.data.gapminder().query("year == 2007")

fig = px.treemap(df, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                 color='lifeExp', color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))

fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()"""

df = px.data.gapminder().query("year == 2007")

fig_38 = px.treemap(df, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                 color='lifeExp', color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))

fig_38.update_layout(margin = dict(t=50, l=25, r=25, b=25))

code_39 = """df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width", marginal_x="histogram", marginal_y="rug")
fig.show()"""

df = px.data.iris()
fig_39 = px.scatter(df, x="sepal_length", y="sepal_width", marginal_x="histogram", marginal_y="rug")

code_40 = """df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                 marginal_x="box", marginal_y="violin")
fig.show()"""

df = px.data.iris()
fig_40 = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                 marginal_x="box", marginal_y="violin")

code_41 = """df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")
fig.show()"""

df = px.data.tips()
fig_41 = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")

code_42 = """df = px.data.iris()
fig = px.scatter_matrix(df,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species")
fig.show()"""

df = px.data.iris()
fig_42 = px.scatter_matrix(df,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species")

code_43 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')
index_vals = df['class'].astype('category').cat.codes

fig = go.Figure(data=go.Splom(
                dimensions=[{"label":col, "values":df[col]} 
                            for col in ['sepal length','sepal width','petal length','petal width']],
                diagonal_visible=False,
                text=df['class'],
                marker=dict(color=index_vals,
                            showscale=False,
                            line_color='white', line_width=0.5)))
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')
index_vals = df['class'].astype('category').cat.codes

fig_43 = go.Figure(data=go.Splom(
                dimensions=[{"label":col, "values":df[col]} 
                            for col in ['sepal length','sepal width','petal length','petal width']],
                diagonal_visible=False,
                text=df['class'],
                marker=dict(color=index_vals,
                            showscale=False,
                            line_color='white', line_width=0.5)))

code_44 = """fig = go.Figure(data=[go.Table(
    header=dict(values=['A Scores', 'B Scores']),
    cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]])
)])

fig.update_layout(width=500, height=350)

fig.show()"""

fig_44 = go.Figure(data=[go.Table(
    header=dict(values=['A Scores', 'B Scores']),
    cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]])
)])

fig_44.update_layout(width=500, height=350)

code_45 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color='lavender',
               align='left'))
])

fig.update_layout(margin=dict(l=50,r=50,t=50,b=50))
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig_45 = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color='lavender',
               align='left'))
])

fig_45.update_layout(margin=dict(l=50,r=50,t=50,b=50))

code_46 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

colors = n_colors('rgb(255, 220, 220)', 'rgb(220, 0, 0)', 15, colortype='rgb')
df["colour"] = pd.cut(df.Population, bins=15, right=False, labels=colors)

fig = go.Figure(data=[go.Table(
    header=dict(values=[f"<b>{col}</b>" for col in df.columns[:-1]],
                fill_color="white",
                align='left',
                font={"color":'black', "size":18}),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color=[df.colour],
               align='left'))
])

fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

colors = n_colors('rgb(255, 220, 220)', 'rgb(220, 0, 0)', 15, colortype='rgb')
df["colour"] = pd.cut(df.Population, bins=15, right=False, labels=colors)

fig_46 = go.Figure(data=[go.Table(
    header=dict(values=[f"<b>{col}</b>" for col in df.columns[:-1]],
                fill_color="white",
                align='left',
                font={"color":'black', "size":18}),
    cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
               fill_color=[df.colour],
               align='left'))
])






st.header("Scatter plots")
for (code, fig) in zip([code_1, code_2, code_3, code_4, code_5, code_6, code_7, code_8, code_9], [fig_1, fig_2, fig_3, fig_4, fig_5, fig_6, fig_7, fig_8, fig_9]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Line charts")
for (code, fig) in zip([code_11, code_12, code_13, code_14, code_15], [fig_11, fig_12, fig_13, fig_14, fig_15]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Bar charts")
for (code, fig) in zip([code_16, code_17, long_df, code_18, code_19, code_20, code_21], [fig_16, fig_17, None, fig_18, fig_19, fig_20, fig_21]):
    if fig is not None:
        st.code(code, language="python")
        st.plotly_chart(fig)
    else:
        st.table(code)

st.header("Box plots")
for (code, fig) in zip([code_22, code_23, code_24], [fig_22, fig_23, fig_24]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Violin plots")
for (code, fig) in zip([code_25, code_26, code_27, code_28], [fig_25, fig_26, fig_27, fig_28]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Histograms")
for (code, fig) in zip([code_29, code_30, code_31, code_32], [fig_29, fig_30, fig_31, fig_32]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Heatmaps")
for (code, fig) in zip([code_33, code_34, code_35, code_36, code_37], [fig_33, fig_34, fig_35, fig_36, fig_37]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Treemaps")
st.code(code_38, language="python")
st.plotly_chart(fig_38)

st.header("Marginal plots")
for (code, fig) in zip([code_38, code_39, code_40, code_41], [fig_38, fig_39, fig_40, fig_41]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Scatterplot matrix")
for (code, fig) in zip([code_42, code_43], [fig_42, fig_43]):
    st.code(code, language="python")
    st.plotly_chart(fig)

st.header("Tables")
for (code, fig) in zip([code_44, code_45, code_46], [fig_44, fig_45, fig_46]):
    st.code(code, language="python")
    st.plotly_chart(fig)
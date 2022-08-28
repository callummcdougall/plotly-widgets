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
    "Image size & margins",
    "Titles",
    "Legends",
    "Error bars & bounds",
    ("Colours", "Discrete colours", "Continuous colour scales"),
    "Templates",
    ("Adding shapes", "Lines", "Rectangles"),
    "Configuration options",
    "Annotations",
    "Animations",
    "Subplots",
    "Time series & date axes (and financial charts)",
    "Marginal plots"
]


with st.sidebar:
    for title in all_titles:
        if type(title) == str:
            st.markdown(f"[{title}]({parse_title(title)})", unsafe_allow_html=True)
        else:
            st.markdown(f"[{title[0]}]({parse_title(title[0])})", unsafe_allow_html=True)
            for word in title[1:]:
                pass
                # st.markdown(f"* [{word}]({parse_title(word)})", unsafe_allow_html=True)
    
code_1 = """fig = go.Figure(go.Scatter(
    x = [1,2,3,4,5],
    y = [2.02,1.63,6.83,4.84,4.73]
))

fig.update_layout(margin=dict(l=20,r=20,t=20,b=20), width=400, height=250)
fig.show()"""

fig_1 = go.Figure(go.Scatter(
    x = [1,2,3,4,5],
    y = [2.02,1.63,6.83,4.84,4.73]
))

fig_1.update_layout(margin=dict(l=20,r=20,t=20,b=20), width=400, height=250)




markdown_1 = """By default, Plotly graphs take up the whole notebook width, and have quite large margins. Using **`fig.update_layout`** you can change this. Three especially useful arguments are **`width`**, **`height`** and **`margin`**. Note that the `l`, `r`, `t` and `b` parameters in the `margin` dict stand for left, right, top and bottom respectively."""
markdown_2 = """This section describes how to add titles to your graph, axes and legend. This can be done while creating the figure, or can be added later.

You can add titles in plotly express using the **`title`** argument, and you can also add titles to the legend and axes by passing a dictionary to the **`labels`** argument. For example:"""
markdown_3 = """Note that the keys of `labels` are the keyword arguments in the line above. If you didn't use a dataframe but instead used arrays for `x`, `y` and `color`, you can find out what your label keys should be by plotting the graph without labels and seeing what text appears on the axes by default - this text should be what you use as keys of your `labels` dict.

Alternatively, you can change the titles using the **`update_layout`** function:"""
markdown_4 = """"""
markdown_ = """"""
markdown_ = """"""
markdown_ = """"""

code_2 = """df = px.data.iris()
fig = px.scatter(
    df, x="sepal_length", y="sepal_width", color="species",
    labels={
        "sepal_length": "Sepal Length (cm)",
        "sepal_width": "Sepal Width (cm)",
        "species": "Species of Iris"
    },
    title="Manually Specified Labels"
)
fig.show()"""

df = px.data.iris()
fig_2 = px.scatter(
    df, x="sepal_length", y="sepal_width", color="species",
    labels={
        "sepal_length": "Sepal Length (cm)",
        "sepal_width": "Sepal Width (cm)",
        "species": "Species of Iris"
    },
    title="Manually Specified Labels"
)

code_3 = """fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    name="Name of Trace 1"       # this sets its legend entry
))

fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
    name="Name of Trace 2"
))

fig.update_layout(
    title="Plot Title",
    xaxis_title="X Axis Title",
    yaxis_title="Y Axis Title",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

fig.show()"""

fig_3 = go.Figure()

fig_3.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    name="Name of Trace 1"       # this sets its legend entry
))

fig_3.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
    name="Name of Trace 2"
))

fig_3.update_layout(
    title="Plot Title",
    xaxis_title="X Axis Title",
    yaxis_title="Y Axis Title",
    legend_title="Legend Title",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="RebeccaPurple"
    )
)

markdown_5 = """Legends are shown to the right of the image by default. This shows how you can hide the legend, or customise it.

Note - you can click on an item in a legend to toggle whether it is visible in a graph (try it for the examples below!)."""

code_4 = """fig = go.Figure(
    data=[
        go.Scatter(x = [1,2,3,4,5], y = [2.02,1.63,6.83,4.84,4.73]), 
        go.Scatter(x = [1,2,3,4,5], y = [3.02,2.63,7.83,5.84,5.73]), 
    ]
)

fig.update_layout(showlegend=False)
fig.show()"""

fig_4 = go.Figure(
    data=[
        go.Scatter(x = [1,2,3,4,5], y = [2.02,1.63,6.83,4.84,4.73]), 
        go.Scatter(x = [1,2,3,4,5], y = [3.02,2.63,7.83,5.84,5.73]), 
    ]
)

fig_4.update_layout(showlegend=False)

code_5 = """# giving legends a name

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    name="Name of Trace 1"       # this sets its legend entry
))


fig.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
    name="Name of Trace 2"
))

fig.update_layout(legend_title="Legend Title")

fig.show()"""


fig_5 = go.Figure()

fig_5.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    name="Name of Trace 1"       # this sets its legend entry
))


fig_5.add_trace(go.Scatter(
    x=[0, 1, 2, 3, 4, 5, 6, 7, 8],
    y=[1, 0, 3, 2, 5, 4, 7, 6, 8],
    name="Name of Trace 2"
))

fig_5.update_layout(legend_title="Legend Title")

code_6 = """df = px.data.gapminder().query("year==2007")
fig = px.scatter(df, x="gdpPercap", y="lifeExp", color="continent",
    size="pop", size_max=45, log_x=True)

fig.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

fig.show()"""

df = px.data.gapminder().query("year==2007")
fig_6 = px.scatter(df, x="gdpPercap", y="lifeExp", color="continent",
    size="pop", size_max=45, log_x=True)

fig_6.update_layout(legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))


markdown_6 = "This shows how you can add error bars and error bounds to scatter plots."

code_7 = """fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[2, 1, 3, 4],
    error_y=dict(
        type='data',
        symmetric=False,
        array=[0.2, 0.2, 0.2, 0.2],
        arrayminus=[0.3, 0.6, 0.9, 1.2])
))

fig.show()"""

fig_7 = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[2, 1, 3, 4],
    error_y=dict(
        type='data',
        symmetric=False,
        array=[0.2, 0.2, 0.2, 0.2],
        arrayminus=[0.3, 0.6, 0.9, 1.2])
    ))

code_8 = """fig = go.Figure(data=go.Scatter(
        x=[0, 1, 2],
        y=[6, 10, 2],
        error_y=dict(
            type='percent',
            value=50,
            visible=True)
    ))
fig.show()"""

fig_8 = go.Figure(data=go.Scatter(
        x=[0, 1, 2],
        y=[6, 10, 2],
        error_y=dict(
            type='percent',
            value=50,
            visible=True)
    ))

code_9 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')

fig = go.Figure([
    go.Scatter(
        x=df['Time'],
        y=df['10 Min Sampled Avg'],
        mode='lines',
        line_color='rgb(31, 119, 180)'
    ),
    go.Scatter(
        x=df['Time'],
        y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
        mode='lines',
        marker_color="#444",
        line_width=0
    ),
    go.Scatter(
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
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/wind_speed_laurel_nebraska.csv')

fig_9 = go.Figure([
    go.Scatter(
        x=df['Time'],
        y=df['10 Min Sampled Avg'],
        mode='lines',
        line_color='rgb(31, 119, 180)'
    ),
    go.Scatter(
        x=df['Time'],
        y=df['10 Min Sampled Avg']+df['10 Min Std Dev'],
        mode='lines',
        marker_color="#444",
        line_width=0
    ),
    go.Scatter(
        x=df['Time'],
        y=df['10 Min Sampled Avg']-df['10 Min Std Dev'],
        marker_color="#444",
        line_width=0,
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty'
    )
])

fig_9.update_traces(showlegend=False)

markdown_7 = """This describes the different colours that plotly supports (both individual colours, and colour scales). 

Each subsection here includes a way to see the full list of colours, and how to use these colours in a plot."""

code_10 = """fig = px.colors.qualitative.swatches()
fig.show()"""

fig_10 = px.colors.qualitative.swatches()

code_11 = """fig = px.colors.sequential.swatches()
fig.update_layout(height=2000)
fig.show()"""

fig_11 = px.colors.sequential.swatches()
fig_11.update_layout(height=2000)

code_12 = """df = px.data.wind()
fig = px.bar_polar(
    df, r="frequency", theta="direction", color="strength",
    color_discrete_sequence= px.colors.sequential.Plotly3,
    title="Part of a continuous color scale used as a discrete sequence"
)

fig.show()"""

df = px.data.wind()
fig_12 = px.bar_polar(
    df, r="frequency", theta="direction", color="strength",
    color_discrete_sequence= px.colors.sequential.Plotly3,
    title="Part of a continuous color scale used as a discrete sequence"
)

code_13 = """fig = px.colors.sequential.swatches_continuous()
fig.update_layout(height=2000)
fig.show()"""

fig_13 = px.colors.sequential.swatches_continuous()
fig_13.update_layout(height=2000)

code_14 = """fig = px.colors.diverging.swatches_continuous()
fig.show()"""

fig_14 = px.colors.diverging.swatches_continuous()

code_15 = """fig = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.1)

for (row_idx, col_idx), continuous_colorscale in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ["Viridis", "Cividis", "Purples", "RdBu"]):
    fig.add_trace(
        go.Contour(
            z=np.outer(np.arange(5), np.arange(5)) + np.random.normal(0, 4, (5, 5)),
            colorscale=continuous_colorscale,
            colorbar=dict(len=0.52, y=0.78-0.55*row_idx, x=0.46+0.55*col_idx) 
            # previous line is just to get the two separate colorbars in the right positions
        ),
        row=row_idx+1, col=col_idx+1
    )

fig.update_layout(title_text="Sample of different continuous colour scales")
fig.show()"""

fig_15 = make_subplots(rows=2, cols=2, shared_yaxes=True, shared_xaxes=True, horizontal_spacing=0.1, vertical_spacing=0.1)

for (row_idx, col_idx), continuous_colorscale in zip([(0, 0), (0, 1), (1, 0), (1, 1)], ["Viridis", "Cividis", "Purples", "RdBu"]):
    fig_15.add_trace(
        go.Contour(
            z=np.outer(np.arange(5), np.arange(5)) + np.random.normal(0, 4, (5, 5)),
            colorscale=continuous_colorscale,
            colorbar=dict(len=0.52, y=0.78-0.55*row_idx, x=0.46+0.55*col_idx) 
            # previous line is just to get the two separate colorbars in the right positions
        ),
        row=row_idx+1, col=col_idx+1
    )

fig_15.update_layout(title_text="Sample of different continuous colour scales")

markdown_8 = """This is a fun and easy way to make plotly charts look cool!

You just pass the `templates` keyword argument into your plotly express graph (or your `go.Layout` call). It takes a string as an argument. The options are:

* `plotly` (default)
* `plotly_light`
* `plotly_dark`
* `simple_white`
* `ggplot2`
* `seaborn`
* `presentation`

Each of these will change lots of aspects of the graph (e.g. font, colors, sizing). It can be a quick way to get nice-looking graphs without messing around with formatting!"""

code_16 = """fig = go.Figure(go.Scatter(
    x = [1,2,3,4,5],
    y = [2.02,1.63,6.83,4.84,4.73]
))

fig.update_layout(margin=dict(l=20,r=20,t=60,b=20), width=400, height=250)

for template in ["plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white", "presentation"]:
    fig.update_layout(title_text=template, template=template)
    fig.show()"""

markdown_9 = """This section describes how you can add shapes to images. Examples include lines, bars, or just regular polygons. It also shows how you can customise graphs to enable you to draw your own shapes (e.g. to highlight some points on a scatter plot)."""

code_17 = """fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[2, 3.5, 6],
    y=[1, 1.5, 1],
    text=["Vertical Line",
          "Horizontal Dashed Line",
          "Diagonal dotted Line"],
    mode="text",
))

fig.update_xaxes(range=[0, 7])
fig.update_yaxes(range=[0, 2.5])

fig.add_shape(type="line",
    x0=1, y0=0, x1=1, y1=2,
    line=dict(color="RoyalBlue",width=3)
)
fig.add_shape(type="line",
    x0=2, y0=2, x1=5, y1=2,
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    )
)
fig.add_shape(type="line",
    x0=4, y0=0, x1=6, y1=2,
    line=dict(
        color="MediumPurple",
        width=4,
        dash="dot",
    )
)
fig.show()"""

fig_17 = go.Figure()

fig_17.add_trace(go.Scatter(
    x=[2, 3.5, 6],
    y=[1, 1.5, 1],
    text=["Vertical Line",
          "Horizontal Dashed Line",
          "Diagonal dotted Line"],
    mode="text",
))

fig_17.update_xaxes(range=[0, 7])
fig_17.update_yaxes(range=[0, 2.5])

fig_17.add_shape(type="line",
    x0=1, y0=0, x1=1, y1=2,
    line=dict(color="RoyalBlue",width=3)
)
fig_17.add_shape(type="line",
    x0=2, y0=2, x1=5, y1=2,
    line=dict(
        color="LightSeaGreen",
        width=4,
        dash="dashdot",
    )
)
fig_17.add_shape(type="line",
    x0=4, y0=0, x1=6, y1=2,
    line=dict(
        color="MediumPurple",
        width=4,
        dash="dot",
    )
)

code_18 = """fig = go.Figure()

fig.add_trace(go.Scatter(
    x=[1.5, 4.5],
    y=[0.75, 0.75],
    text=["Unfilled Rectangle", "Filled Rectangle"],
    mode="text",
))

fig.update_xaxes(range=[0, 7], showgrid=False)
fig.update_yaxes(range=[0, 3.5])

fig.add_shape(type="rect",
    x0=1, y0=1, x1=2, y1=3,
    line=dict(color="RoyalBlue"),
)
fig.add_shape(type="rect",
    x0=3, y0=1, x1=6, y1=2,
    line=dict(
        color="RoyalBlue",
        width=2,
    ),
    fillcolor="LightSkyBlue",
)
fig.update_shapes(dict(xref='x', yref='y'))
fig.show()"""

fig_18 = go.Figure()

fig_18.add_trace(go.Scatter(
    x=[1.5, 4.5],
    y=[0.75, 0.75],
    text=["Unfilled Rectangle", "Filled Rectangle"],
    mode="text",
))

fig_18.update_xaxes(range=[0, 7], showgrid=False)
fig_18.update_yaxes(range=[0, 3.5])

fig_18.add_shape(type="rect",
    x0=1, y0=1, x1=2, y1=3,
    line=dict(color="RoyalBlue"),
)
fig_18.add_shape(type="rect",
    x0=3, y0=1, x1=6, y1=2,
    line=dict(
        color="RoyalBlue",
        width=2,
    ),
    fillcolor="LightSkyBlue",
)
fig_18.update_shapes(dict(xref='x', yref='y'))

code_19 = """fig = go.Figure()

fig.add_trace(go.Scatter(
    x=["2015-02-01", "2015-02-02", "2015-02-03", "2015-02-04", "2015-02-05",
       "2015-02-06", "2015-02-07", "2015-02-08", "2015-02-09", "2015-02-10",
       "2015-02-11", "2015-02-12", "2015-02-13", "2015-02-14", "2015-02-15",
       "2015-02-16", "2015-02-17", "2015-02-18", "2015-02-19", "2015-02-20",
       "2015-02-21", "2015-02-22", "2015-02-23", "2015-02-24", "2015-02-25",
       "2015-02-26", "2015-02-27", "2015-02-28"],
    y=[-14, -17, -8, -4, -7, -10, -12, -14, -12, -7, -11, -7, -18, -14, -14,
       -16, -13, -7, -8, -14, -8, -3, -9, -9, -4, -13, -9, -6],
    mode="lines",
    name="temperature"
))

fig.add_vrect(
    x0="2015-02-04", x1="2015-02-06",
    fillcolor="LightSalmon", opacity=0.5,
    layer="below", line_width=0,
),

fig.add_vrect(
    x0="2015-02-20", x1="2015-02-22",
    fillcolor="LightSalmon", opacity=0.5,
    layer="below", line_width=0,
)

fig.show()"""
fig_19 = go.Figure()

fig_19.add_trace(go.Scatter(
    x=["2015-02-01", "2015-02-02", "2015-02-03", "2015-02-04", "2015-02-05",
       "2015-02-06", "2015-02-07", "2015-02-08", "2015-02-09", "2015-02-10",
       "2015-02-11", "2015-02-12", "2015-02-13", "2015-02-14", "2015-02-15",
       "2015-02-16", "2015-02-17", "2015-02-18", "2015-02-19", "2015-02-20",
       "2015-02-21", "2015-02-22", "2015-02-23", "2015-02-24", "2015-02-25",
       "2015-02-26", "2015-02-27", "2015-02-28"],
    y=[-14, -17, -8, -4, -7, -10, -12, -14, -12, -7, -11, -7, -18, -14, -14,
       -16, -13, -7, -8, -14, -8, -3, -9, -9, -4, -13, -9, -6],
    mode="lines",
    name="temperature"
))

fig_19.add_vrect(
    x0="2015-02-04", x1="2015-02-06",
    fillcolor="LightSalmon", opacity=0.5,
    layer="below", line_width=0,
);

fig_19.add_vrect(
    x0="2015-02-20", x1="2015-02-22",
    fillcolor="LightSalmon", opacity=0.5,
    layer="below", line_width=0,
)

code_20 = """df = px.data.stocks(indexed=True)
fig = px.line(df, facet_col="company", facet_col_wrap=2)

fig.add_hline(y=1, 
              line_dash="dot",
              annotation_text="Jan 1, 2018 baseline",
              annotation_position="bottom right")

fig.add_vrect(x0="2018-09-24", 
              x1="2018-12-18", 
              col=1,
              annotation_text="decline", 
              annotation_position="top left",
              fillcolor="green", 
              opacity=0.25, 
              line_width=0)

fig.show()"""

df = px.data.stocks(indexed=True)
fig_20 = px.line(df, facet_col="company", facet_col_wrap=2)

fig_20.add_hline(y=1, 
              line_dash="dot",
              annotation_text="Jan 1, 2018 baseline",
              annotation_position="bottom right")

fig_20.add_vrect(x0="2018-09-24", 
              x1="2018-12-18", 
              col=1,
              annotation_text="decline", 
              annotation_position="top left",
              fillcolor="green", 
              opacity=0.25, 
              line_width=0)

markdown_10 = """This shows you how to change the options on the graph, to do things like:

* allow users to draw their own shapes on the graph
* hide the display bar that shows up on the top-right of plots"""

code_21 = """fig = go.Figure(
    data=go.Scatter(
        x=[1, 2, 3],
        y=[1, 3, 1]
    ),
    layout_title_text="No modebar!"
)

fig.show(config=dict(displayModeBar=False))"""

fig_21 = go.Figure(
    data=go.Scatter(
        x=[1, 2, 3],
        y=[1, 3, 1]
    ),
    layout_title_text="No modebar!"
)

code_22 = """fig = go.Figure()

fig.add_annotation(
    x=0.5, xref="paper", # setting ref to "paper" means (0, 1) are ends of plot, rather than referring to plot axes
    y=0.5, yref="paper",
    text="Click and drag here <br> to draw a rectangle <br><br> or select another shape <br>in the modebar",
    font_size=20
)

fig.add_shape(
    editable=True,
    x0=-1, x1=0, y0=2, y1=3,
    xref='x', yref='y'# this means the values refer to the plot axes
)

fig.update_layout(dragmode='drawrect')

fig.show(config={'modeBarButtonsToAdd': [
    'drawline',
    'drawopenpath',
    'drawclosedpath',
    'drawcircle',
    'drawrect',
    'eraseshape'
]})"""

fig_22 = go.Figure()

fig_22.add_annotation(
    x=0.5, xref="paper", # setting ref to "paper" means (0, 1) are ends of plot, rather than referring to plot axes
    y=0.5, yref="paper",
    text="Click and drag here <br> to draw a rectangle <br><br> or select another shape <br>in the modebar",
    font_size=20
)

fig_22.add_shape(
    editable=True,
    x0=-1, x1=0, y0=2, y1=3,
    xref='x', yref='y'# this means the values refer to the plot axes
)

fig_22.update_layout(dragmode='drawrect')

markdown_11 = """You can also use `line_color` and `fillcolor` arguments to customise the shapes:"""

code_23 = """fig = go.Figure()

fig.add_annotation(
    x=0.5, xref="paper",
    y=0.5, yref="paper",
    text="Click and drag here <br> to draw a rectangle <br><br> or select another shape <br>in the modebar",
    font_size=20
)

fig.add_shape(
    line_color='yellow', fillcolor='turquoise',
    editable=True,
    x0=-1, x1=0, y0=2, y1=3,
    xref='x', yref='y'
)

fig.update_layout(dragmode='drawrect',
                  newshape=dict(line_color='yellow', fillcolor='turquoise', opacity=0.5))

fig.show(config={'modeBarButtonsToAdd': [
    'drawline',
    'drawopenpath',
    'drawclosedpath',
    'drawcircle',
    'drawrect',
    'eraseshape'
]})"""

fig_23 = go.Figure()

fig_23.add_annotation(
    x=0.5, xref="paper",
    y=0.5, yref="paper",
    text="Click and drag here <br> to draw a rectangle <br><br> or select another shape <br>in the modebar",
    font_size=20
)

fig_23.add_shape(
    line_color='yellow', fillcolor='turquoise',
    editable=True,
    x0=-1, x1=0, y0=2, y1=3,
    xref='x', yref='y'
)

fig_23.update_layout(dragmode='drawrect',
                  newshape=dict(line_color='yellow', fillcolor='turquoise', opacity=0.5))


markdown_12 = "Annotations are relatively straightforward in plotly - you basically add them just like you add points for a regular scatter graph, except that it's text rather than a point!)."

code_24 = """fig = go.Figure(
    data = go.Scatter(
        x=[0.5, 1, 1.5],
        y=[1, 1, 1],
        mode="text",
        text=["Text A", "Text B", "Text C"]
    ),
    layout = go.Layout(
        width=400,
        height=250,
        margin=dict(l=20,r=20,t=20,b=20),
        xaxis_range=[0, 2]
    )
)

fig.show()"""

fig_24 = go.Figure(
    data = go.Scatter(
        x=[0.5, 1, 1.5],
        y=[1, 1, 1],
        mode="text",
        text=["Text A", "Text B", "Text C"]
    ),
    layout = go.Layout(
        width=400,
        height=250,
        margin=dict(l=20,r=20,t=20,b=20),
        xaxis_range=[0, 2]
    )
)

markdown_13 = """Note you can also annotate a plotly express figure by using the `add_traces` method, and passing in `text`."""

code_25 = """df = px.data.gapminder().query("year==2007")

fig = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent", log_x=True, size_max=60)

fig.add_traces(go.Scatter(
    x=[2000, 20000],
    y=[80, 60],
    mode="text",
    text=["Here's some text!", "Here's some more text!"]
))

fig.show()"""

df = px.data.gapminder().query("year==2007")

fig_25 = px.scatter(df, x="gdpPercap", y="lifeExp", size="pop", color="continent", log_x=True, size_max=60)

fig_25.add_traces(go.Scatter(
    x=[2000, 20000],
    y=[80, 60],
    mode="text",
    text=["Here's some text!", "Here's some more text!"]
))

markdown_14 = """Plotly has pretty advanced animations features, and the full syntax is too complicated to be worth including here, so I've just added one example to give an idea.

`animation_frame` tells you which column of the dataframe should correspond to the menu on the bottom. So in the case below, each point along the menu displays a different graph from data of the form `df[df.year == year]`."""

code_26 = """df = px.data.gapminder()
px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90]).show()"""

df = px.data.gapminder()
fig_26 = px.scatter(df, x="gdpPercap", y="lifeExp", animation_frame="year",
           size="pop", color="continent", hover_name="country",
           log_x=True, size_max=55, range_x=[100,100000], range_y=[25,90])

markdown_15 = """There are ways to do animations with `go.Figure()`, but these are quite complicated and not worth diving into."""

markdown_16 = """This shows you how to create subplots, i.e. multiple graphs in the same output."""

markdown_17 = """Note that you can create subplots from plotly express charts using this small hack: 

* Create your plotly express figure as you normally would (e.g. `subfig = px.scatter(...)`)
* Extract the traces from it using `traces = subfig.data`
* Loop through each trace and add it to your subplot figure using `fig.add_trace`, as shown above"""

code_27 = """fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
fig.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)

fig.show()"""

fig_27 = make_subplots(rows=1, cols=2)

fig_27.add_trace(go.Scatter(y=[4, 2, 1], mode="lines"), row=1, col=1)
fig_27.add_trace(go.Bar(y=[2, 1, 3]), row=1, col=2)

code_28 = """df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/Mining-BTC-180.csv", usecols=range(1, 9))

for i, row in enumerate(df["Date"]):
    df.iat[i, 0] = re.compile(" 00:00:00").split(df["Date"][i])[0]

fig = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Mining-revenue-USD"],
        mode="lines",
        name="mining revenue"
    ),
    row=3, col=1
)

fig.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Hash-rate"],
        mode="lines",
        name="hash-rate-TH/s"
    ),
    row=2, col=1
)

fig.add_trace(
    go.Table(
        header=dict(
            values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)",
                    "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD",
                    "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
            font_size=10,
            align="left"
        ),
        cells=dict(
            values=[df[k].tolist() for k in df.columns],
            align = "left")
    ),
    row=1, col=1
)
fig.update_layout(
    height=700,
    width=800,
    margin=dict(l=20,t=50,b=20,r=20),
    showlegend=False,
    title_text="Bitcoin mining stats for 180 days",
)

fig.show()"""

df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/Mining-BTC-180.csv", usecols=range(1, 9))

for i, row in enumerate(df["Date"]):
    df.iat[i, 0] = re.compile(" 00:00:00").split(df["Date"][i])[0]

fig_28 = make_subplots(
    rows=3, cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    specs=[[{"type": "table"}],
           [{"type": "scatter"}],
           [{"type": "scatter"}]]
)

fig_28.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Mining-revenue-USD"],
        mode="lines",
        name="mining revenue"
    ),
    row=3, col=1
)

fig_28.add_trace(
    go.Scatter(
        x=df["Date"],
        y=df["Hash-rate"],
        mode="lines",
        name="hash-rate-TH/s"
    ),
    row=2, col=1
)

fig_28.add_trace(
    go.Table(
        header=dict(
            values=["Date", "Number<br>Transactions", "Output<br>Volume (BTC)",
                    "Market<br>Price", "Hash<br>Rate", "Cost per<br>trans-USD",
                    "Mining<br>Revenue-USD", "Trasaction<br>fees-BTC"],
            font_size=10,
            align="left"
        ),
        cells=dict(
            values=[df[k].tolist() for k in df.columns],
            align = "left")
    ),
    row=1, col=1
)
fig_28.update_layout(
    height=700,
    width=800,
    margin=dict(l=20,t=50,b=20,r=20),
    showlegend=False,
    title_text="Bitcoin mining stats for 180 days",
)

markdown_18 = """This is a showcase of some of the ways you can format the axes of a timeseries graph."""

code_29 = """df = px.data.stocks()
fig = px.line(df, x='date', y="GOOG") # plotly express handles date formatting automatically (even if dates are strings)
fig.show()"""

df = px.data.stocks()
fig_29 = px.line(df, x='date', y="GOOG") # plotly express handles date formatting automatically (even if dates are strings)

code_30 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
fig = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])]) # same is true for plotly graph objects
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')
fig_30 = go.Figure([go.Scatter(x=df['Date'], y=df['AAPL.High'])]) # same is true for plotly graph objects

code_31 = """df = px.data.stocks(indexed=True)-1
fig = px.area(df, facet_col="company", facet_col_wrap=3)
fig.show()"""

df = px.data.stocks(indexed=True)-1
fig_31 = px.area(df, facet_col="company", facet_col_wrap=3)

code_32 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(data=[go.Candlestick(x=df['Date'], 
                                     open=df['AAPL.Open'], 
                                     high=df['AAPL.High'], 
                                     low=df['AAPL.Low'], 
                                     close=df['AAPL.Close'])])
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig_32 = go.Figure(data=[go.Candlestick(x=df['Date'], 
                                     open=df['AAPL.Open'], 
                                     high=df['AAPL.High'], 
                                     low=df['AAPL.Low'], 
                                     close=df['AAPL.Close'])])

markdown_19 = """These next examples showcase particular behaviour of date axes in plotly. Both examples' code is pretty complicated and not very generalisable, so I wouldn't recommend remembering how to write it, just copy it from here if you need it!"""

code_33 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = px.line(df, x='Date', y='AAPL.High', title='Time Series with Rangeslider')

fig.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    )
)
fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig_33 = px.line(df, x='Date', y='AAPL.High', title='Time Series with Rangeslider')

fig_33.update_xaxes(
    rangeslider_visible=True,
    rangeselector=dict(
        buttons=[
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ]
    )
)

code_34 = """df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig = go.Figure(go.Scatter(
    x = df['Date'],
    y = df['mavg']
))

timeperiods = [None, 1000, 60000, 3600000, 86400000, 604800000, "M1", "M12", None]
timeformats = ["%H:%M:%S.%L ms", "%H:%M:%S s", "%H:%M m", "%H:%M h", "%e. %b d", "%e. %b w", "%b '%y M", "%Y Y"]

df = pd.DataFrame({
    "dtickrange": [timeperiods[i:i+2] for i in range(len(timeperiods)-1)],
    "value": timeformats
})

fig.update_xaxes(
    rangeslider_visible=True,
    tickformatstops = [dict(df.loc[i, :]) for i in df.index]
)

fig.show()"""

df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/finance-charts-apple.csv')

fig_34 = go.Figure(go.Scatter(
    x = df['Date'],
    y = df['mavg']
))

timeperiods = [None, 1000, 60000, 3600000, 86400000, 604800000, "M1", "M12", None]
timeformats = ["%H:%M:%S.%L ms", "%H:%M:%S s", "%H:%M m", "%H:%M h", "%e. %b d", "%e. %b w", "%b '%y M", "%Y Y"]

df = pd.DataFrame({
    "dtickrange": [timeperiods[i:i+2] for i in range(len(timeperiods)-1)],
    "value": timeformats
})

fig_34.update_xaxes(
    rangeslider_visible=True,
    tickformatstops = [dict(df.loc[i, :]) for i in df.index]
)

markdown_20 = """This feature allows you to put marginal plots (e.g. rugs, box plots or violin plots) on the x and y axes of your main plot. You can use it in lots of different types of graph."""

code_35 = """df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                 marginal_x="box", marginal_y="violin")
fig.show()"""

df = px.data.iris()
fig_35 = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                 marginal_x="box", marginal_y="violin")

code_36 = """df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")
fig.show()"""

df = px.data.tips()
fig_36 = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")




st.header("Image size & margins")
with st.expander(""):
    st.markdown(markdown_1)
    st.code(code_1, language="python")
    st.plotly_chart(fig_1)

st.header("Titles")
with st.expander(""):
    st.markdown(markdown_2)
    st.code(code_2, language="python")
    st.plotly_chart(fig_2)
    st.markdown(markdown_3)
    st.code(code_3, language="python")
    st.plotly_chart(fig_3)

st.header("Legends")
with st.expander(""):
    st.markdown(markdown_5)
    st.code(code_4, language="python")
    st.plotly_chart(fig_4)
    st.code(code_5, language="python")
    st.plotly_chart(fig_5)
    st.code(code_6, language="python")
    st.plotly_chart(fig_6)

st.header("Error bars & bounds")
with st.expander(""):
    st.markdown(markdown_6)
    st.code(code_7, language="python")
    st.plotly_chart(fig_7)
    st.code(code_8, language="python")
    st.plotly_chart(fig_8)
    st.code(code_9, language="python")
    st.plotly_chart(fig_9)

st.header("Colours")
with st.expander(""):
    st.markdown(markdown_7)
    st.subheader("Discrete")
    st.code(code_10, language="python")
    st.plotly_chart(fig_10)
    st.code(code_11, language="python")
    st.plotly_chart(fig_11)
    st.code(code_12, language="python")
    st.plotly_chart(fig_12)
    st.subheader("Continuous")
    st.code(code_13, language="python")
    st.plotly_chart(fig_13)
    st.code(code_14, language="python")
    st.plotly_chart(fig_14)
    st.code(code_15, language="python")
    st.plotly_chart(fig_15)
    st.markdown("Note, you can reverse any colorscale by appending '_r' to its name.")

st.header("Templates")
with st.expander(""):
    st.markdown(markdown_8)
    st.code(code_16)
    st.image("images/plotly-introduction-fig2.png")

st.header("Adding shapes")
with st.expander(""):
    st.markdown(markdown_9)
    st.subheader("Lines")
    st.code(code_17, language="python")
    st.plotly_chart(fig_17)
    st.subheader("Rectangles")
    st.code(code_18, language="python")
    st.plotly_chart(fig_18)
    st.code(code_19, language="python")
    st.plotly_chart(fig_19)
    st.code(code_20, language="python")
    st.plotly_chart(fig_20)

st.header("Configuration options")
with st.expander(""):
    st.markdown(markdown_10)
    st.code(code_21, language="python")
    st.plotly_chart(fig_21, config=dict(displayModeBar=False))
    st.code(code_22, language="python")
    st.plotly_chart(fig_22, config={'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ]})
    st.markdown(markdown_11)
    st.code(code_23, language="python")
    st.plotly_chart(fig_23, config={'modeBarButtonsToAdd': [
        'drawline',
        'drawopenpath',
        'drawclosedpath',
        'drawcircle',
        'drawrect',
        'eraseshape'
    ]})

st.header("Annotations")
with st.expander(""):
    st.markdown(markdown_12)
    st.code(code_24, language="python")
    st.plotly_chart(fig_24)
    st.markdown(markdown_13)
    st.code(code_25, language="python")
    st.plotly_chart(fig_25)

st.header("Animations")
with st.expander(""):
    st.markdown(markdown_14)
    st.code(code_26, language="python")
    st.plotly_chart(fig_26)
    st.markdown(markdown_15)

st.header("Subplots")
with st.expander(""):
    st.markdown(markdown_16)
    st.code(code_27, language="python")
    st.plotly_chart(fig_27)
    st.code(code_28, language="python")
    st.plotly_chart(fig_28)
    st.markdown(markdown_17)

st.header("Time series & date axes (and financial charts)")
with st.expander(""):
    st.markdown(markdown_18)
    st.code(code_29, language="python")
    st.plotly_chart(fig_29)
    st.code(code_30, language="python")
    st.plotly_chart(fig_30)
    st.code(code_31, language="python")
    st.plotly_chart(fig_31)
    st.code(code_32, language="python")
    st.plotly_chart(fig_32)
    st.markdown(markdown_19)
    st.code(code_33, language="python")
    st.plotly_chart(fig_33)
    st.code(code_34, language="python")
    st.plotly_chart(fig_34)

st.header("Marginal plots")
with st.expander(""):
    st.markdown(markdown_20)
    st.code(code_35, language="python")
    st.plotly_chart(fig_35)
    st.code(code_36, language="python")
    st.plotly_chart(fig_36)

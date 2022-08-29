import streamlit as st

import ipywidgets as wg
from IPython.display import display, HTML

import pandas as pd
import numpy as np

import sys, os
import time
import datetime
import re
import scipy

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
                text-decoration: none;
			}
            a:hover {
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

all_titles = ["Bessel Functions", "Sine Waves", "Flights", "Querying tables", "Tabs"]

with st.sidebar:
    for title in all_titles:
        st.markdown(f"[{title}]({parse_title(title)})", unsafe_allow_html=True)

@st.cache(hash_funcs={dict: lambda _: None})
def loading_dataframes():

    df_dict = dict()

    df_flights = pd.read_csv('https://raw.githubusercontent.com/yankev/testing/master/datasets/nycflights.csv')
    df_flights = df_flights.drop(df_flights.columns[[0]], axis=1)
    df_dict["flights"] = df_flights

    df_tables = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
    df_dict["tables"] = df_tables

    return df_dict

df_dict = loading_dataframes()

st.markdown("""In this section, I demonstrate how Plotly and Widgets can be used in tandem. 

Note that, although the code is correct, the widgets themselves are native to Streamlit rather than being IPyWidgets (which is why they look different than previous sections). This is because Streamlit doesn't support IPyWidgets yet. If you want to play around with the real Plotly-Widget objects, I'd recommend using my [**notebook**](https://github.com/callummcdougall/plotly-widgets/blob/main/WidgetsPlotlyGuide.ipynb). The widgets will also run much faster there!

Tangential sidenote though - [**Streamlit**](https://docs.streamlit.io/library/api-reference) is awesome and really easy to learn (I'd estimate about 2-3 hours to learn enough to make a site like this one, and another 4-5 hours to actually make it), so if you ever want to port over interactive Plotly charts into an actual website, I'd strongly recommend it!""")

code_bessel = """x = np.linspace(0, np.pi, 1000)

layout = go.Layout(
    title = "SIMPLE EXAMPLE",           # fig.layout.title = "SIMPLE EXAMPLE"
    yaxis = {"title": "volts"},         # fig.layout.yaxis.title = "volts"
    xaxis = {"title": "nanoseconds"},
    width = 500,
    height = 300,
    margin = dict(l=40,r=40,t=60,b=40)
)

@wg.interact
def update_plot(signals = wg.SelectMultiple(options=list(range(6)), value=(0, ), description = "Bessel Order"), 
                freq = wg.FloatSlider(min=1, max=20, value=1, desription="Freq")):
    
    fig = go.Figure(layout = layout)
    
    for s in signals:
        
        trace = go.Scatter(
            x = x,
            y = scipy.special.jv(s, freq * x), 
            mode = "lines",
            name = f"Bessel {s}", 
        )
        
        fig.add_traces(trace)

    fig.show()"""

st.header("Bessel Functions")

st.markdown("This is a simple example using the **`wg.interact()`** function (for more information, see the previous section). The syntax is the same here, except the function ends by showing a Plotly figure, rather than printing text output.")

with st.expander("Code"):
    st.code(code_bessel, language="python")



columns_bessel = st.columns(2)
with columns_bessel[0]:
    signals = st.multiselect(options=list(range(6)), default=0, label="Bessel Order")
    freq = st.slider(label="Freq", min_value=1, max_value=20, value=1)

x = np.linspace(0, np.pi, 1000)
layout = go.Layout(
    title = "SIMPLE EXAMPLE",
    yaxis = {"title": "volts"},
    xaxis = {"title": "nanoseconds"},
    width = 500,
    height = 300,
    margin = dict(l=40,r=40,t=60,b=40)
)
fig_bessel = go.Figure(layout = layout)
for s in signals:
    trace = go.Scatter(
        x = x,
        y = scipy.special.jv(s, freq * x), 
        mode = "lines",
        name = f"Bessel {s}", 
    )
    fig_bessel.add_traces(trace)

st.plotly_chart(fig_bessel)


st.header("Sine Waves")

st.markdown("""This shows how you can abbreviate widgets when you use **`wg.interact`**. For instance, **`(1.0, 4.0, 0.05)`** means a slider widget with minimum 1, maximum 4, and step size 0.05. Personally, I don't like using this notation, but it's useful to be aware of.""")

code_sine = """fig = go.FigureWidget(
    layout=go.Layout(
        title_text="y = sin(ax - b)",
        width=500,
        height=300,
        margin=dict(l=20,r=20,t=50,b=20)
    )
)
scatt = fig.add_scatter()

xs=np.linspace(0, 6, 100)

@wg.interact(a=(1.0, 4.0, 0.05), b=(0, 10.0, 0.05), color=['red', 'green', 'blue'])
def update(a=3.6, b=4.3, color='blue'):
    ys = np.sin(a*xs - b)
    with fig.batch_update():
        fig.data[0].x = xs
        fig.data[0].y = ys
        fig.data[0].line.color = color
    fig.show()"""

with st.expander("Code"):
    st.code(code_sine)

fig = go.FigureWidget(
    layout=go.Layout(
        title_text="y = sin(ax - b)",
        width=500,
        height=300,
        margin=dict(l=20,r=20,t=50,b=20)
    )
)
scatt = fig.add_scatter()
xs=np.linspace(0, 6, 100)

cols_sine = st.columns(2)
with cols_sine[0]:
    a = st.slider(label="a", min_value=1.0, max_value=4.0, step=0.05, value=3.6)
    b = st.slider(label="b", min_value=0.0, max_value=10.0, step=0.05, value=4.3)
    color = st.selectbox(options=["red", "green", "blue"], index=2, label="Color")

ys = np.sin(a*xs - b)
with fig.batch_update():
    fig.data[0].x = xs
    fig.data[0].y = ys
    fig.data[0].line.color = color
st.plotly_chart(fig)




st.header("Flights")

code_flights = """df = pd.read_csv('https://raw.githubusercontent.com/yankev/testing/master/datasets/nycflights.csv')
df = df.drop(df.columns[[0]], axis=1)

month_widget = wg.IntSlider(min=1, max=12, step=1, value=1, description='Month:', continuous_update=False)
use_date_widget = wg.Checkbox(value=True, description='Date: ')
airline_widget = wg.Dropdown(description='Airline:   ', options=df['carrier'].unique(), value='DL')
origin_widget = wg.Dropdown(description='Origin Airport:', options=df['origin'].unique(), value='LGA')

trace1 = go.Histogram(x=df['arr_delay'], opacity=0.75, name='Arrival Delays')
trace2 = go.Histogram(x=df['dep_delay'], opacity=0.75, name='Departure Delays')
data2 = [trace1, trace2]

layout2 = go.Layout(title='NYC FlightDatabase', 
                    barmode='overlay', 
                    xaxis={"title":"Delay in Minutes"}, 
                    yaxis={"title":"Number of Delays"})

fig_hist = go.FigureWidget(data=data2, layout=layout2)


def update_histogram(change):

    filter_list = (df["carrier"] == airline_widget.value) & (df["origin"] == origin_widget.value)
    if use_date_widget.value:
        filter_list &= (df['month'] == month_widget.value)
    temp_df = df[filter_list]

    with fig_hist.batch_update():
        fig_hist.data[0].x = temp_df['arr_delay']
        fig_hist.data[1].x = temp_df['dep_delay']

update_histogram("unimportant text") # useful for triggering first response

for widget in [airline_widget, origin_widget, month_widget, use_date_widget]:
    widget.observe(update_histogram, names="value")

widget_box_1 = wg.VBox([use_date_widget, month_widget])
widget_box_2 = wg.VBox([airline_widget, origin_widget])
widget_box_main = wg.HBox([widget_box_1, widget_box_2])

wg.VBox([widget_box_main, fig_hist])"""

with st.expander("Code"):
    st.code(code_flights, language="python")

markdown_flights = """This introduces the **`go.FigureWidget`**, object which has identical syntax to the Plotly **`go.Figure`**, but it is also treated like a widget in the **`ipywidgets`** library (i.e. you don't need to create an output widget to put the graph in; the graph itself is an output widget).

This also uses the **`fig.batch_update()`** method. This is a good way to update FigureWidgets in your functions. It sends all the updates at the same time, rather than one at a time (which can create flickering).

P.S. — This example is probably the most complicated in this whole site, so don't worry if it seems hard to interpret at first."""

st.markdown(markdown_flights)

df_flights = df_dict["flights"]

cols_flights_1 = st.columns(3)
cols_flights_2 = st.columns(3)
with cols_flights_1[0]: use_date = st.checkbox(value=True, label="Date: ")
with cols_flights_1[1]: month = st.slider(min_value=1, max_value=12, step=1, value=1, label="Month:")
with cols_flights_2[0]: textbox = st.selectbox(label="Airline:   ", index=3, options=df_flights["carrier"].unique().tolist())
with cols_flights_2[1]: origin = st.selectbox(label="Origin Airport:   ", index=1, options=tuple(df_flights['origin'].unique().tolist()))

trace1 = go.Histogram(x=df_flights['arr_delay'], opacity=0.75, name='Arrival Delays')
trace2 = go.Histogram(x=df_flights['dep_delay'], opacity=0.75, name='Departure Delays')
data = [trace1, trace2]
layout = go.Layout(title='NYC FlightDatabase', barmode='overlay', xaxis_title="Delay in Minutes", yaxis_title="Number of Delays")
fig = go.Figure(data=data, layout=layout)
    
mask = (df_flights["carrier"] == textbox) & (df_flights["origin"] == origin)
if use_date: mask &= (df_flights['month'] == month)
temp_df_flights = df_flights[mask]
with fig.batch_update():
    fig.data[0].x = temp_df_flights['arr_delay']
    fig.data[1].x = temp_df_flights['dep_delay']

st.plotly_chart(fig)

code_tables = '''df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig = go.FigureWidget(
    data=[
        go.Table(
            header={"values": list(df.columns),
                    "fill_color": 'paleturquoise',
                    "align": 'left'},
            cells={"values": [df.Rank, df.State, df.Postal, df.Population],
                   "fill_color": 'lavender',
                   "align": 'left'}
        )
    ],
    layout=go.Layout(margin=dict(l=40,r=40,t=40,b=40))
)

label_widget = wg.HTML()

def handle_submit(sender):
    query = sender.value
    try:
        df_reduced = df if query == "" else df.query(query)
        label_widget.value = ""
    except:
        label_widget.value = f"""{repr(query)} is invalid query. Examples: 
        <code><b>State=='Alabama'</b></code>, or <code><b>Postal<'MM' and Population<1000000</b></code>"""
        df_reduced = pd.DataFrame(columns=df.columns)
    fig.data[0].cells.values = [df_reduced.Rank, df_reduced.State, df_reduced.Postal, df_reduced.Population]
        
text_widget = wg.Text(
    value='',
    placeholder='Type something, then press Enter!',
    description='Query:',
    layout=wg.Layout(width="75%")
)
        
text_widget.on_submit(handle_submit)

display(wg.VBox([text_widget, label_widget, fig]))'''

st.header("Querying tables")

with st.expander("Code"):
    st.code(code_tables, language="python")

st.markdown("""This is an example of how you can query a table in realtime. For more information on the **`df.query()`** method, see [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).

A few example queries you can try on this data:

* `State=='Alabama'`
* `Postal<'MM' and Population<1000000`

Note - the reason this responds slowly is because of how Streamlit's API works; the response in Jupyter Notebooks would be near instant.""")

df_tables = df_dict["tables"]

fig_tables = go.FigureWidget(
    data=[
        go.Table(
            header={"values": list(df_tables.columns),
                    "fill_color": 'paleturquoise',
                    "align": 'left'},
            cells={"values": [df_tables.Rank, df_tables.State, df_tables.Postal, df_tables.Population],
                   "fill_color": 'lavender',
                   "align": 'left'}
        )
    ],
    layout=go.Layout(margin=dict(l=40,r=40,t=40,b=40))
)

cols_table = st.columns(2)
with cols_table[0]:
    text_widget = st.text_area(
        value='',
        placeholder="""Type something, then press Enter!""",
        label='Query:',
    )

if text_widget == "":
    st.plotly_chart(fig_tables)
else:
    try:
        df_tables_reduced = df_tables.query(text_widget)
        fig_tables.data[0].cells.values = [df_tables_reduced.Rank, df_tables_reduced.State, df_tables_reduced.Postal, df_tables_reduced.Population]
        st.plotly_chart(fig_tables)
    except:
        st.error("""Invalid query (see Pandas [documentation page](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) on queries).""")


code_tabs = '''df0 = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')

fig_table = go.FigureWidget(
    data=[
        go.Table(
            header={"values": list(df0.columns),
                    "fill_color": 'paleturquoise',
                    "align": 'left'},
            cells={"values": [df0[col] for col in df0.columns],
                   "fill_color": 'lavender',
                   "align": 'left'}
        )
    ],
    layout=go.Layout(margin=dict(l=40,r=40,t=40,b=40))
)

label_widget = wg.HTML()

def handle_submit(sender):
    query = sender.value
    try:
        df0_reduced = df0 if query == "" else df0.query(query)
        label_widget.value = ""
    except:
        label_widget.value = f"""{repr(query)} is invalid query. Examples: 
        <code><b>State=='Alabama'</b></code>, or <code><b>Postal<'MM' and Population<1000000</b></code>"""
        df0_reduced = pd.DataFrame(columns=df0.columns)
    fig_table.data[0].cells.values = [df0_reduced[col] for col in df0.columns]
        
text_widget = wg.Text(
    value='',
    placeholder='Type something, then press Enter!',
    description='Query:',
    layout=wg.Layout(width="75%")
)

text_widget.on_submit(handle_submit)
        

    
df = pd.read_csv('https://raw.githubusercontent.com/yankev/testing/master/datasets/nycflights.csv')
df = df.drop(df.columns[[0]], axis=1)

month_widget = wg.IntSlider(min=1, max=12, step=1, value=1, description='Month:', continuous_update=False)
use_date_widget = wg.Checkbox(value=True, description='Date: ')
airline_widget = wg.Dropdown(description='Airline:   ', options=df['carrier'].unique(), value='DL')
origin_widget = wg.Dropdown(description='Origin Airport:', options=df['origin'].unique(), value='LGA')

trace1 = go.Histogram(x=df['arr_delay'], opacity=0.75, name='Arrival Delays')
trace2 = go.Histogram(x=df['dep_delay'], opacity=0.75, name='Departure Delays')
data2 = [trace1, trace2]

layout2 = go.Layout(title='NYC FlightDatabase', 
                    barmode='overlay', 
                    xaxis={"title":"Delay in Minutes"}, 
                    yaxis={"title":"Number of Delays"})

fig_hist = go.FigureWidget(data=data2, layout=layout2)


def update_histogram(change):

    filter_list = (df["carrier"] == airline_widget.value) & (df["origin"] == origin_widget.value)
    if use_date_widget.value:
        filter_list &= (df['month'] == month_widget.value)
    temp_df = df[filter_list]

    with fig_hist.batch_update():
        fig_hist.data[0].x = temp_df['arr_delay']
        fig_hist.data[1].x = temp_df['dep_delay']
        
update_histogram("unimportant text") # useful for triggering first response



for widget in [airline_widget, origin_widget, month_widget, use_date_widget]:
    widget.observe(update_histogram, names="value")

box_layout = wg.Layout(
    border='solid 1px gray',
    margin='0px 10px 10px 0px',
    padding='5px 5px 5px 5px')
    
children = [
    wg.VBox([text_widget, label_widget, fig_table]), 
    wg.VBox([
        wg.HBox([
            wg.VBox([use_date_widget, month_widget], layout=box_layout), 
            wg.VBox([airline_widget, origin_widget], layout=box_layout)
        ]), 
        fig_hist
    ])
]

tab = wg.Tab(children = children)
tab.set_title(0, 'box #1')
tab.set_title(1, 'box #2')

display(tab)'''

st.title("Tabs")

with st.expander("Code"):
    st.code(code_tabs, language="python")

st.markdown("""The final example of this notebook, this shows how you can combine 2 of the above instances into a single GUI, with Tabs. Most of the code below is copied from the two previous examples.

I'm using screenshots of the actual widgets here rather than Streamlit widgets, because you can see both the code examples above.""")

cols_tabs = st.columns(2)
with cols_tabs[0]: st.image("../images/table-1.png")
with cols_tabs[1]: st.image("../images/table-2.png")

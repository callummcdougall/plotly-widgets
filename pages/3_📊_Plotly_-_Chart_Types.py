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

with st.sidebar:
    for title in all_titles:
        st.markdown(f"[{title}]({parse_title(title)})", unsafe_allow_html=True)


@st.cache(hash_funcs={dict: lambda _: None})
def fetching_plotly_figures():
    d = dict()
    df = px.data.iris()
    d[1] = px.scatter(df, x="sepal_width", y="sepal_length", color="species", size='petal_length', hover_data=['petal_width'])
    N = 100
    random_x = np.linspace(0, 1, N)
    random_y = [np.random.randn(N) + 5, np.random.randn(N), np.random.randn(N) - 5]
    trace_info = ["markers", "lines+markers", "lines"]
    d[2] = go.Figure(data=[go.Scatter(x=random_x, y=y, mode=t, name=t) for (y, t) in zip(random_y, trace_info)])
    d[3] = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], mode='markers', marker_size=[40, 60, 80, 100], marker_color=[0, 1, 2, 3]))
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
    fig_4.update_traces(mode='markers', marker_line_width=2, marker_size=10)
    fig_4.update_layout(title='Styled Scatter',
                    yaxis_zeroline=True, 
                    xaxis_zeroline=False)
    d[4] = fig_4
    data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")
    d[5] = go.Figure(data=go.Scatter(x=data['Postal'],
                                    y=data['Population'],
                                    mode='markers+text',
                                    marker_color=data['Population'],
                                    hovertext=data['State']),
                layout=go.Layout(title="Population of USA States"))
    d[6] = go.Figure(data=go.Scatter(
        y = np.random.randn(500),
        mode='markers',
        marker=dict(
            size=16,
            color=np.random.randn(500),
            colorscale='Viridis',
            showscale=True,
        )
    ))
    df = px.data.tips()
    fig_7 = px.scatter(df, x="total_bill", y="tip", facet_col="sex",
                    width=800, height=400)
    fig_7.update_layout(
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="LightSteelBlue",
    )
    d[7] = fig_7
    df = px.data.gapminder().query("year==2007")
    d[8] = px.scatter(df, x="gdpPercap", y="lifeExp",
                    size="pop", color="continent",
                    log_x=True, size_max=60)
    df = px.data.gapminder()
    fig_9 = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                    size="pop", color="continent",
                    hover_name="country", log_x=True, size_max=60)
    fig_9.update_layout(xaxis=dict(gridcolor="white", gridwidth=2),
                    yaxis=dict(gridcolor="white", gridwidth=2),
                    paper_bgcolor='rgb(235, 235, 235)',
                    plot_bgcolor='rgb(235, 235, 235)')
    d[9] = fig_9
    country = ['Switzerland (2011)', 'Chile (2013)', 'Japan (2014)', 'United States (2012)', 'Poland (2010)', 'Estonia (2015)', 'Luxembourg (2013)', 'Portugal (2011)']
    voting_pop = [40, 45.7, 52, 53.6, 54.5, 54.7, 55.1, 56.6]
    reg_voters = [49.1, 42, 52.7, 84.3, 55.3, 64.2, 91.1, 58.9]
    fig_10 = go.Figure()
    fig_10.add_trace(go.Scatter(
        x=voting_pop,
        y=country,
        name='Percent of estimated voting age population',
        marker=dict(
            color='rgba(156, 165, 196, 0.95)',
            line_color='rgba(156, 165, 196, 1.0)',
        )
    ))
    fig_10.add_trace(go.Scatter(
        x=reg_voters, y=country,
        name='Percent of estimated registered voters',
        marker=dict(
            color='rgba(204, 204, 204, 0.95)',
            line_color='rgba(217, 217, 217, 1.0)'
        )
    ))
    fig_10.update_traces(mode='markers', marker=dict(line_width=1, symbol='circle', size=16))
    fig_10.update_layout(
        title="Votes cast for 10 lowest voting age population in OECD countries",
        xaxis=dict(
            showgrid=False,
            showline=True,
            linecolor='rgb(102, 102, 102)',
            tickfont_color='rgb(102, 102, 102)',
            showticklabels=True,
            dtick=10,
            ticks='outside',
            tickcolor='rgb(102, 102, 102)',
        ),
        margin=dict(l=140, r=40, b=50, t=80),
        legend=dict(
            font_size=11,
            yanchor='middle',
            xanchor='right',
        ),
        width=800,
        height=600,
        paper_bgcolor='white',
        plot_bgcolor='white',
        hovermode='closest',
    )
    d[10] = fig_10
    df = px.data.gapminder()
    df = df[df.continent.isin(["Americas"])]
    d[11] = px.line(df, x="year", y="lifeExp", color='country')
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
    d[12] = go.Figure(
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
    d[13] = go.Figure(
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
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([1, 3, 2, 3, 1])
    fig_14 = go.Figure(data=[go.Scatter(x=x, y=y, name="linear", line_shape="linear"),
                        go.Scatter(x=x, y=y+5, name="spline", line_shape="spline"),
                        go.Scatter(x=x, y=y+10, name="hv", line_shape="hv")])

    fig_14.update_traces(hoverinfo='name', mode='lines+markers')
    fig_14.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))
    d[14] = fig_14
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
    d[15] = fig_15
    data_canada = px.data.gapminder()
    d[16] = px.bar(data_canada[data_canada.country == "Canada"], x='year', y='pop', hover_data=["lifeExp"])
    long_df = px.data.medals_long()
    d[17] = px.bar(long_df, x="nation", y="count", color="medal", barmode="stack")


    data = px.data.gapminder()
    data_canada = data[data.country == 'Canada']
    d[18] = px.bar(data_canada, x='year', y='pop',
                hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',
                labels={'pop':'population of Canada'}, height=400, color_continuous_scale="Bluered")
    df = px.data.tips()
    d[19] = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group",
                facet_row="time", facet_col="day",
                category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]}, 
                color_discrete_sequence=["darkorange", "brown"])
    df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
    fig_20 = px.bar(df, y='pop', x='country', text='pop', template='ggplot2')
    fig_20.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    d[20] = fig_20
    d[21] = go.Figure(
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
    df = px.data.tips()
    d[22] = px.box(df, x="sex", y="total_bill")
    x0 = np.random.randn(50)
    x1 = np.random.randn(50) + 2
    fig_23 = go.Figure(data=[go.Box(x=x0), go.Box(x=x1)])
    fig_23.update_layout(height=250, width=600, margin=dict(l=20,r=20,b=20,t=20))
    d[23] = fig_23
    df = px.data.tips()
    d[24] = px.box(df, x="time", y="total_bill", color="smoker", points="all", notched=True)
    df = px.data.tips()
    d[25] = px.violin(df, y="total_bill", box=True, points='all')
    df = px.data.tips()
    d[26] = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all", hover_data=df.columns)
    df = pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/violin_data.csv")
    fig_27 = go.Figure()
    fig_27.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'Yes' ],
                            y=df['total_bill'][ df['smoker'] == 'Yes' ],
                            legendgroup='Yes', scalegroup='Yes', name='Yes',
                            side='negative',
                            line_color='blue'))
    fig_27.add_trace(go.Violin(x=df['day'][ df['smoker'] == 'No' ],
                            y=df['total_bill'][ df['smoker'] == 'No' ],
                            legendgroup='No', scalegroup='No', name='No',
                            side='positive',
                            line_color='orange'))
    fig_27.update_traces(meanline_visible=True)
    fig_27.update_layout(violingap=0, violinmode='overlay', height=300, width=600, margin_t=20)
    d[27] = fig_27
    data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) + (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])
    colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')
    fig_28 = go.Figure()
    for data_line, color in zip(data, colors):
        fig_28.add_trace(go.Violin(x=data_line, line_color=color))
    fig_28.update_traces(orientation='h', side='positive', width=3, points=False)
    fig_28.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
    d[28] = fig_28
    df = px.data.tips()
    fig_29 = make_subplots(rows=1, cols=2)
    fig_29.add_trace(go.Histogram(x=df["total_bill"], nbinsx=80), 1, 1)
    fig_29.add_trace(go.Histogram(x=df["day"]), 1, 2)
    d[29] = fig_29
    x0 = np.random.randn(500)
    x1 = np.random.randn(500) + 1
    fig_30 = go.Figure(data=[go.Histogram(x=x0, marker_color='#EB89B5', name="Pink"), go.Histogram(x=x1, marker_color='#330C73', name="Purple")])
    fig_30.update_layout(barmode='overlay')
    fig_30.update_traces(opacity=0.6)
    d[30] = fig_30
    df = px.data.tips()
    fig_31 = px.histogram(df, x="total_bill", y="tip", color="sex",
                    marginal="box",
                    hover_data=df.columns)
    fig_31.update_layout(height=350, width=750, margin=dict(l=20,r=20,t=20,b=20))
    d[31] = fig_31
    x1 = np.random.randn(200) - 2
    x2 = np.random.randn(200)
    x3 = np.random.randn(200) + 2
    x4 = np.random.randn(200) + 4
    hist_data = [x1, x2, x3, x4]
    group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']
    d[32] = ff.create_distplot(hist_data, group_labels, bin_size=.2)
    d[33] = go.Figure(
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
    fig_34 = px.imshow(z, color_continuous_scale='Viridis')
    d[34] = fig_34
    z = [[.1, .3, .5],
        [1.0, .8, .6],
        [.6, .4, .2]]
    x = ['Team A', 'Team B', 'Team C']
    y = ['Game Three', 'Game Two', 'Game One']
    z_text = [['Win', 'Lose', 'Win'],
            ['Lose', 'Lose', 'Win'],
            ['Win', 'Win', 'Lose']]
    fig_35 = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
    fig_35.update_traces(text=z_text, texttemplate="%{text}<br><br>z = %{z}")
    fig_35.update_xaxes(side="top")
    d[35] = fig_35
    data = np.random.normal(size=(25, 10)) + 0.3 * np.random.normal(size=(10,))[None, :]
    d["35b"] = px.imshow(np.corrcoef(data), color_continuous_scale="RdBu_r", origin="lower")
    df = px.data.tips()
    d[36] = px.density_heatmap(df, x="total_bill", y="tip", histfunc="avg")
    d[37] = px.density_heatmap(df, x="total_bill", y="tip", facet_row="sex", facet_col="smoker")
    df = px.data.gapminder().query("year == 2007")
    fig_38 = px.treemap(df, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                    color='lifeExp', color_continuous_scale='RdBu',
                    color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))
    fig_38.update_layout(margin = dict(t=50, l=25, r=25, b=25))
    d[38] = fig_38
    df = px.data.iris()
    d[39] = px.scatter(df, x="sepal_length", y="sepal_width", marginal_x="histogram", marginal_y="rug")
    d[40] = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                    marginal_x="box", marginal_y="violin")
    df = px.data.tips()
    d[41] = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")
    df = px.data.iris()
    d[42] = px.scatter_matrix(df,
        dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
        color="species")
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/iris-data.csv')
    index_vals = df['class'].astype('category').cat.codes
    d[43] = go.Figure(data=go.Splom(
                    dimensions=[{"label":col, "values":df[col]} 
                                for col in ['sepal length','sepal width','petal length','petal width']],
                    diagonal_visible=False,
                    text=df['class'],
                    marker=dict(color=index_vals,
                                showscale=False,
                                line_color='white', line_width=0.5)))
    df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv')
    colors = n_colors('rgb(255, 220, 220)', 'rgb(220, 0, 0)', 15, colortype='rgb')
    df["colour"] = pd.cut(df.Population, bins=15, right=False, labels=colors)
    fig_44 = go.Figure(data=[go.Table(
        header=dict(values=['A Scores', 'B Scores']),
        cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]])
    )])
    fig_44.update_layout(width=500, height=350)
    d[44] = fig_44
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
    d[45] = fig_45
    colors = n_colors('rgb(255, 220, 220)', 'rgb(220, 0, 0)', 15, colortype='rgb')
    df["colour"] = pd.cut(df.Population, bins=15, right=False, labels=colors)
    d[46] = go.Figure(data=[go.Table(
        header=dict(values=[f"<b>{col}</b>" for col in df.columns[:-1]],
                    fill_color="white",
                    align='left',
                    font={"color":'black', "size":18}),
        cells=dict(values=[df.Rank, df.State, df.Postal, df.Population],
                fill_color=[df.colour],
                align='left'))
    ])
    return d, long_df

code_1 = """df = px.data.iris()

fig = px.scatter(df, x="sepal_width", y="sepal_length", color="species",
                 size='petal_length', hover_data=['petal_width'])
fig.show()"""

code_2 = """N = 100
random_x = np.linspace(0, 1, N)
random_y = [np.random.randn(N) + 5, np.random.randn(N), np.random.randn(N) - 5]

trace_info = ["markers", "lines+markers", "lines"]
fig = go.Figure(data=[go.Scatter(x=random_x, y=y, mode=t, name=t) for (y, t) in zip(random_y, trace_info)])

fig.show()"""

code_3 = """fig = go.Figure(data=go.Scatter(
    x=[1, 2, 3, 4],
    y=[10, 11, 12, 13],
    mode='markers',
    marker_size=[40, 60, 80, 100],
    marker_color=[0, 1, 2, 3]))

fig.show()"""

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

code_5 = """data= pd.read_csv("https://raw.githubusercontent.com/plotly/datasets/master/2014_usa_states.csv")

fig = go.Figure(data=go.Scatter(x=data['Postal'],
                                y=data['Population'],
                                mode='markers+text',
                                marker_color=data['Population'],
                                hovertext=data['State']),
               layout=go.Layout(title="Population of USA States"))

fig.show()"""

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

code_7 = """df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", facet_col="sex",
                 width=800, height=400)

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="LightSteelBlue",
)

fig.show()"""

code_8 = """df = px.data.gapminder().query("year==2007")

fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 log_x=True, size_max=60)

fig.show()"""

code_9 = """df = px.data.gapminder()

fig = px.scatter(df.query("year==2007"), x="gdpPercap", y="lifeExp",
                 size="pop", color="continent",
                 hover_name="country", log_x=True, size_max=60)

fig.update_layout(xaxis=dict(gridcolor="white", gridwidth=2),
                  yaxis=dict(gridcolor="white", gridwidth=2),
                  paper_bgcolor='rgb(235, 235, 235)',
                  plot_bgcolor='rgb(235, 235, 235)')
fig.show()"""

code_10 = """country = ['Switzerland (2011)', 'Chile (2013)', 'Japan (2014)', 'United States (2012)',
           'Poland (2010)', 'Estonia (2015)', 'Luxembourg (2013)', 'Portugal (2011)']
voting_pop = [40, 45.7, 52, 53.6, 54.5, 54.7, 55.1, 56.6]
reg_voters = [49.1, 42, 52.7, 84.3, 55.3, 64.2, 91.1, 58.9]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=voting_pop,
    y=country,
    name='Percent of estimated voting age population',
    marker=dict(
        color='rgba(156, 165, 196, 0.95)',
        line_color='rgba(156, 165, 196, 1.0)',
    )
))
fig.add_trace(go.Scatter(
    x=reg_voters, y=country,
    name='Percent of estimated registered voters',
    marker=dict(
        color='rgba(204, 204, 204, 0.95)',
        line_color='rgba(217, 217, 217, 1.0)'
    )
))

fig.update_traces(mode='markers', 
                  marker=dict(line_width=1, symbol='circle', size=16))

fig.update_layout(
    title="Votes cast for 10 lowest voting age population in OECD countries",
    xaxis=dict(
        showgrid=False,
        showline=True,
        linecolor='rgb(102, 102, 102)',
        tickfont_color='rgb(102, 102, 102)',
        showticklabels=True,
        dtick=10,
        ticks='outside',
        tickcolor='rgb(102, 102, 102)',
    ),
    margin=dict(l=140, r=40, b=50, t=80),
    legend=dict(
        font_size=11,
        yanchor='middle',
        xanchor='right',
    ),
    width=800,
    height=450,
    paper_bgcolor='white',
    plot_bgcolor='white',
    hovermode='closest',
)
fig.show()"""

code_11 = """df = px.data.gapminder()

df = df[df.continent.isin(["Americas"])]

fig = px.line(df, x="year", y="lifeExp", color='country')
fig.show()"""

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

code_14 = """x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 3, 2, 3, 1])

fig = go.Figure(data=[go.Scatter(x=x, y=y, name="linear", line_shape="linear"),
                      go.Scatter(x=x, y=y+5, name="spline", line_shape="spline"),
                      go.Scatter(x=x, y=y+10, name="hv", line_shape="hv")])

fig.update_traces(hoverinfo='name', mode='lines+markers')
fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=16))

fig.show()"""

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

code_16 = """data_canada = px.data.gapminder()
fig = px.bar(data_canada[data_canada.country == "Canada"], x='year', y='pop', hover_data=["lifeExp"])
fig.show()"""

code_17 = """long_df = px.data.medals_long()
display(long_df)

px.bar(long_df, x="nation", y="count", color="medal", barmode="stack").show()"""

code_18 = """data = px.data.gapminder()

data_canada = data[data.country == 'Canada']
fig = px.bar(data_canada, x='year', y='pop',
             hover_data=['lifeExp', 'gdpPercap'], color='lifeExp',
             labels={'pop':'population of Canada'}, height=400, color_continuous_scale="Bluered")
fig.show()"""

code_19 = """df = px.data.tips()
fig = px.bar(df, x="sex", y="total_bill", color="smoker", barmode="group",
             facet_row="time", facet_col="day",
             category_orders={"day": ["Thur", "Fri", "Sat", "Sun"], "time": ["Lunch", "Dinner"]}, 
             color_discrete_sequence=["darkorange", "brown"])
fig.show()"""

code_20 = """df = px.data.gapminder().query("continent == 'Europe' and year == 2007 and pop > 2.e6")
fig = px.bar(df, y='pop', x='country', text='pop', template='ggplot2')
fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')
fig.show()"""

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

code_22 = """df = px.data.tips()
fig = px.box(df, x="sex", y="total_bill")
fig.show()"""

code_23 = """x0 = np.random.randn(50)
x1 = np.random.randn(50) + 2

fig = go.Figure(data=[go.Box(x=x0), go.Box(x=x1)])

fig.update_layout(height=250, width=600, margin=dict(l=20,r=20,b=20,t=20))
fig.show()"""

code_24 = """df = px.data.tips()
fig = px.box(df, x="time", y="total_bill", color="smoker", points="all", notched=True)
fig.show()"""

code_25 = """df = px.data.tips()
fig = px.violin(df, y="total_bill", box=True,
                points='all')
fig.show()"""

code_26 = """df = px.data.tips()
fig = px.violin(df, y="tip", x="smoker", color="sex", box=True, points="all",
          hover_data=df.columns)
fig.show()"""

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

code_28 = """data = (np.linspace(1, 2, 12)[:, np.newaxis] * np.random.randn(12, 200) +
            (np.arange(12) + 2 * np.random.random(12))[:, np.newaxis])

colors = n_colors('rgb(5, 200, 200)', 'rgb(200, 10, 10)', 12, colortype='rgb')

fig = go.Figure()
for data_line, color in zip(data, colors):
    fig.add_trace(go.Violin(x=data_line, line_color=color))

fig.update_traces(orientation='h', side='positive', width=3, points=False)
fig.update_layout(xaxis_showgrid=False, xaxis_zeroline=False)
fig.show()"""

code_29 = """df = px.data.tips()

fig = make_subplots(rows=1, cols=2)

fig.add_trace(go.Histogram(x=df["total_bill"], nbinsx=80), 1, 1)
fig.add_trace(go.Histogram(x=df["day"]), 1, 2)

fig.show()"""

code_30 = """x0 = np.random.randn(500)
x1 = np.random.randn(500) + 1

fig = go.Figure(data=[go.Histogram(x=x0, marker_color='#EB89B5', name="Pink"), 
                      go.Histogram(x=x1, marker_color='#330C73', name="Purple")])

fig.update_layout(barmode='overlay')
fig.update_traces(opacity=0.6)

fig.show()"""



code_31 = """df = px.data.tips()
fig = px.histogram(df, x="total_bill", y="tip", color="sex",
                   marginal="box",
                   hover_data=df.columns)
fig.update_layout(height=350, width=750, margin=dict(l=20,r=20,t=20,b=20))
fig.show()"""

code_32 = """x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

hist_data = [x1, x2, x3, x4]
group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)

fig.show()"""

code_33 = """fig = go.Figure(
    data=go.Heatmap(
        z=[[1, None, 30, 50, 1], [20, 1, 60, 80, 30], [30, 60, 1, -10, 20]],
        x=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
        y=['Morning', 'Afternoon', 'Evening'],
        hoverongaps = False),
    layout=go.Layout(width=750,height=400,margin_t=20)
)
fig.show()"""

code_34 = """z = [[.1, .3, .5, .7],
     [1, .8, .6, .4],
     [.6, .4, .2, .0],
     [.9, .7, .5, .3]]

fig = px.imshow(z, color_continuous_scale='Viridis')
fig.show()"""

code_35 = """z = [[.1, .3, .5],
     [1.0, .8, .6],
     [.6, .4, .2]]

x = ['Team A', 'Team B', 'Team C']
y = ['Game Three', 'Game Two', 'Game One']

z_text = [['Win', 'Lose', 'Win'],
          ['Lose', 'Lose', 'Win'],
          ['Win', 'Win', 'Lose']]

fig = px.imshow(z, x=x, y=y, color_continuous_scale='Viridis', aspect="auto")
fig.update_traces(text=z_text, texttemplate="%{text}<br><br>z = %{z}")
fig.update_xaxes(side="top")

fig.show()"""

code_35b = """data = np.random.normal(size=(25, 10)) + 0.3 * np.random.normal(size=(10,))[None, :]

fig = px.imshow(np.corrcoef(data), color_continuous_scale="RdBu_r", origin="lower")

fig.show()"""

code_36 = """df = px.data.tips()

fig = px.density_heatmap(df, x="total_bill", y="tip", histfunc="avg")

fig.show()"""

code_37 = """df = px.data.tips()

fig = px.density_heatmap(df, x="total_bill", y="tip", facet_row="sex", facet_col="smoker")

fig.show()"""

code_38 = """df = px.data.gapminder().query("year == 2007")

fig = px.treemap(df, path=[px.Constant("world"), 'continent', 'country'], values='pop',
                 color='lifeExp', color_continuous_scale='RdBu',
                 color_continuous_midpoint=np.average(df['lifeExp'], weights=df['pop']))

fig.update_layout(margin = dict(t=50, l=25, r=25, b=25))
fig.show()"""

code_39 = """df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width", marginal_x="histogram", marginal_y="rug")
fig.show()"""

code_40 = """df = px.data.iris()
fig = px.scatter(df, x="sepal_length", y="sepal_width", color="species", 
                 marginal_x="box", marginal_y="violin")
fig.show()"""

code_41 = """df = px.data.tips()
fig = px.scatter(df, x="total_bill", y="tip", color="sex", facet_col="day", marginal_x="box")
fig.show()"""

code_42 = """df = px.data.iris()
fig = px.scatter_matrix(df,
    dimensions=["sepal_width", "sepal_length", "petal_width", "petal_length"],
    color="species")
fig.show()"""

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

code_44 = """fig = go.Figure(data=[go.Table(
    header=dict(values=['A Scores', 'B Scores']),
    cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]])
)])

fig.update_layout(width=500, height=350)

fig.show()"""

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

d, long_df = fetching_plotly_figures()

st.header("Scatter plots")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_1, code_2, code_3, code_4, code_5, code_6, code_7, code_8, code_9, code_10], [d[i] for i in range(1, 11)]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_scatter_1 = st.columns(2)
cols_scatter_2 = st.columns(2)
for i in range(2):
    with cols_scatter_1[i]: st.image(f"images/charts-scatter-{i}.png")
for i in range(2):
    with cols_scatter_2[i]: st.image(f"images/charts-scatter-{i+2}.png")

st.header("Line charts")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_11, code_12, code_13, code_14, code_15], [d[i] for i in range(11, 16)]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_line = st.columns(2)
for i in range(2):
    with cols_line[i]: st.image(f"images/charts-line-{i}.png")

st.header("Bar charts")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_16, code_17, long_df, code_18, code_19, code_20, code_21], [d[16], d[17], None, d[18], d[19], d[20], d[21]]):
        if fig is not None:
            st.code(code, language="python")
            st.plotly_chart(fig)
        else:
            st.table(code)
cols_bar_1 = st.columns(2)
cols_bar_2 = st.columns(2)
for i in range(2):
    with cols_bar_1[i]: st.image(f"images/charts-bar-{i}.png")
for i in range(2):
    with cols_bar_2[i]: st.image(f"images/charts-bar-{i+2}.png")

st.header("Box plots")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_22, code_23, code_24], [d[22], d[23], d[24]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_box = st.columns(2)
for i in range(2):
    with cols_box[i]: st.image(f"images/charts-box-{i}.png")

st.header("Violin plots")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_25, code_26, code_27, code_28], [d[25], d[26], d[27], d[28]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_violin = st.columns(2)
for i in range(2):
    with cols_violin[i]: st.image(f"images/charts-violin-{i}.png")

st.header("Histograms")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_29, code_30, code_31, code_32], [d[29], d[30], d[31], d[32]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_histogram = st.columns(2)
for i in range(2):
    with cols_histogram[i]: st.image(f"images/charts-histogram-{i}.png")

st.header("Heatmaps")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_33, code_34, code_35, code_35b, code_36, code_37], [d[33], d[34], d[35], d["35b"], d[36], d[37]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_heatmap_1 = st.columns(2)
cols_heatmap_2 = st.columns(2)
for i in range(2):
    with cols_heatmap_1[i]: st.image(f"images/charts-heatmap-{i}.png")
for i in range(2):
    with cols_heatmap_2[i]: st.image(f"images/charts-heatmap-{i+2}.png")

st.header("Treemaps")
with st.expander("Code & interactive graphs"):
    st.code(code_38, language="python")
    st.plotly_chart(d[38])
cols_treemap = st.columns(2)
with cols_treemap[0]: st.image("images/charts-treemap.png")

st.header("Marginal plots")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_39, code_40, code_41], [d[39], d[40], d[41]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_marginal = st.columns(2)
for i in range(2):
    with cols_marginal[i]: st.image(f"images/charts-marginal-{i}.png")

st.header("Scatterplot matrix")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_42, code_43], [d[42], d[43]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_scatterplot = st.columns(2)
for i in range(2):
    with cols_scatterplot[i]: st.image(f"images/charts-scatterplot-{i}.png")

st.header("Tables")
with st.expander("Code & interactive graphs"):
    for (code, fig) in zip([code_44, code_45, code_46], [d[44], d[45], d[46]]):
        st.code(code, language="python")
        st.plotly_chart(fig)
cols_table = st.columns(2)
for i in range(2):
    with cols_table[i]: st.image(f"images/charts-table-{i}.png")

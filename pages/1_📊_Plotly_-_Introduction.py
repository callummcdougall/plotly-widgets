import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd

# streamlit_style = """
# 			<style>
# 			@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@100&display=swap');

# 			html, body, [class*="css"]  {
# 			font-family: 'Roboto', sans-serif;
# 			}
# 			</style>
# 			"""
# st.markdown(streamlit_style, unsafe_allow_html=True)

st.set_page_config(
    layout="wide",
    page_title="Plotly & Widgets",
    page_icon="📊",
    menu_items={
        "Get help": "https://www.perfectlynormal.co.uk/",
        "About": "##### This was created to demo the Python libraries of Plotly and IPyWidgets, and show how they can be combined to create interactive output in Python notebooks."
    }
)

st.title("Why Plotly?")

st.markdown("""Learning a new library's syntax is hard and time-consuming, especially so for plotting libraries (since there are so many different things you might want to plot). Many people reading this are probably already familiar with `matplotlib`, and might be wondering why they would want to switch. Here, I sketch out a few reasons why I made the jump, and consider Plotly to be superior to matplotlib. I'll elaborate on each of these later on.""")

markdown_1 = """## Interactive plots

This is the big one. Matplotlib does make interactive plots, but they aren't nearly as big a feature for that programming langauge. In contrast, all of Plotly's graphs are interactive by default. You can zoom in on particular chunks of data, hover over it to see values, etc. This is a massive help when you're trying to make sense of data, whether that means examining the activations of neurons in a transformer, or examining in detail a certain trading pattern from a week in 2019.

Below is just one example of a powerful interactive graph which you can create with a very small amount of code. You can try hovering over datapoints to see extra information, or clicking and dragging to zoom in on a section of the graph (and double clicking to reset the axes)."""

code_text = """df = px.data.gapminder().query("year==2007")

    fig = px.scatter(df, x="gdpPercap", y="lifeExp",
                    size="pop", color="continent",
                    hover_name="country", log_x=True, size_max=60)

    fig.show()"""

with st.expander("Interactive plots"):
    
    st.markdown(markdown_1)
    st.code(code_text, language="python")

    df = px.data.gapminder().query("year==2007")
    fig_1 = px.scatter(df, x="gdpPercap", y="lifeExp",
                    size="pop", color="continent",
                    hover_name="country", log_x=True, size_max=60)
    st.plotly_chart(fig_1)


markdown_2 = """
## Intuitive functions and parameters

I personally found matplotlib quite hard to learn, because the functions and params and the ways they were used never seemed intuitive for me. Plotly always seemed more straightforward. To summarise how the core parts of it work:

> **You first define a figure. This can either be from plotly express (e.g. `px.scatter`) or from plotly graph objects (e.g. `go.Figure(data=go.Scatter)`). `px` generally requires less code, while `go` requires more code but gives you more low-level control. Both `px` and `go` will accept data in the form of lists or arrays, but if you use `px` then you have the added option of passing in data as a dataframe, and having the other arguments refer to column names of that dataframe.**

> **You can then update the figure by calling the method `update_traces` (to change the data) or `update_layout` (to change the appearance). Most parameters are accessed via nested dictionaries, e.g. you would pass `title = dict(text="My Title")` into the update layout function to change the title. This can also be abbreviated to `title_text="My Title` (this is called magic underscore notation in Plotly).**

The [documentation pages](https://plotly.com/python/) for Plotly are also really good, and you can find basically any kind of graph you want there."""

with st.expander("Intuitive functions and parameters"):
    st.markdown(markdown_2)

markdown_21 = """## Integration with IPython widgets

Widgets are interactive ways to control certain values in Python (e.g. sliders, or dropdown menus). These can be easily combined with Plotly using the `go.FigureWidget` wrapper to create graphs which change in response to the widgets' values. This is another great way to explore your data in more detail."""

with st.expander("Integration with IPython widgets"):
    st.markdown(markdown_21)
    st.image("plotly-introduction-fig1.png")

st.markdown("""I expect matplotlib to still be the best tool for some tasks, and it might be that learning a new graphing library doesn't appeal to you. But before you reject the idea, I'd encourage you to look through some of the other sections of this course, and give it a try!""")



markdown_22 = """The core object in Plotly is the Figure object, which is pretty similar to the Figure object in matplotlib. It has quite a few attributes, but the most important ones are **`data`** and **`layout`**.

* **`data`** contains the actual data of the figure, in the form of a list of **`traces`**. A trace is a single set of related graphical marks in a figure. For instance, the plot below has three traces, corresponding to the three groups of differently-formatted points.

* **`layout`** contains the top-level attributes of the figure, such as dimensions, margins, titles and axes.

### Creating and Displaying Figures

The low-level way to create figures in plotly is by using **`plotly.graph_objects`**. 

You can display figures in your notebook by using **`fig.show()`**."""

code_2 = """import plotly.graph_objects as go

fig = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(height=400, width=600)
)
fig.show()"""

code_3 = """>>> fig.data
(Bar({
     'x': [1, 2, 3], 'y': [1, 3, 2]
 }),)

>>> fig.layout
Layout({
    'height': 400, 'template': '...', 'width': 600
})"""

markdown_3 = """Here we can see that `data` is stored as a tuple of traces (in this case just one), and `layout` is stored as a single object. We can extract a specific value either as an attribute, or by treating it as a dictionary. Many properties of graphs can be accessed and initialised as nested dictionaries."""

st.header("The `Figure` object")
st.markdown(markdown_22)
st.code(code_2, language="python")
fig_2 = go.Figure(
    data=[go.Bar(x=[1, 2, 3], y=[1, 3, 2])],
    layout=go.Layout(height=400, width=600)
)
st.plotly_chart(fig_2)
st.code(code_3, language="python")
st.markdown(markdown_3)



code_4 = """>>> fig.data[0].x
(1, 2, 3)
>>> fig.layout["height"]
400"""

st.code(code_4, language="python")

markdown_4 = """### Updating Figures

Graphs can be updated by using the methods **`update_traces`** (for the data) and **`update_layout`** (for the layout). An example is shown below."""

st.markdown(markdown_4)

code_5 = """fig.update_traces(marker_color="red")
fig.update_layout(title_text="A Figure Specified By A Graph Object", width=600, height=400)

fig.show()"""

st.code(code_5, language="python")

fig_2.update_traces(marker_color="red")
fig_2.update_layout(title_text="A Figure Specified By A Graph Object", width=600, height=400)
st.plotly_chart(fig_2)

markdown_5 = """Note — you might expect **`marker_color`** to be under layout, not traces. However, layout deals with the top-level attributes of the figure (e.g. title and legend), whereas traces deals with the values *and appearance* of the *data*, so this includes things like marker colour."""

st.markdown(markdown_5)


markdown_6 = """**Magic underscore notation** is how we string together nested properties, to make code look nicer. For instance, we called `update_layout(title_text=...)` in the code above, i.e. using an underscore to connect `title` and `text`. We can also use nested dictionaries, which can be more useful when we want to set multiple sub-properties at once. 

Example:"""

code_6 = """fig.update_layout(
    title=dict(
        text="We can use dictionaries rather than underscores!",
        font=dict(
            family="Times New Roman",
            color="red"
        )
    )
)
fig.show()"""

fig_2.update_layout(
    title=dict(
        text="We can use dictionaries rather than underscores!",
        font=dict(
            family="Times New Roman",
            color="red"
        )
    )
)

code_7 = """fig.update_layout(
    title_text="We can use dictionaries rather than underscores!",
    title_font_family="Times New Roman",
    title_font_color="red"
)"""


st.header("Magic underscore notation")
st.markdown(markdown_6)
st.code(code_6, language="python")
st.plotly_chart(fig_2)
st.markdown("This is equivalent to the following code:")
st.code(code_7, language="python")


markdown_7 = """We have already covered plotly graph objects, and how they are used to construct figures. The other main way to construct figures in plotly is by using **`plotly.express`**. Below is a comparison of the two methods.

### **Plotly graph objects** 
* Usually imported as **`import plotly.graph_objects as go`**
* More low-level, and gives you more control over the graph
* Makes some features easier to implement, e.g. subplots or faceted plots
    
### **Plotly express**
* Usually imported as **`import plotly.express as px`**
* Creates entire figures at once, with very little code
* Automates a lot of plotly features, like hover labels and styling
* Very useful for quite niche graphs (e.g. treemaps)
* Can read in data as a dataframe, rather than just arrays

### Which one to use?

In reality, plotly express is built on top of graph objects, it just abstracts away some of the details. However, there are still times when graph objects are easier to use. In the code below, we'll see lots of examples of both.

One big advantage of plotly express is that data for graphs can be supplied by a Pandas dataframe. The basic idea is that you have `df` as your first argument, and then other arguments like `x`, `y`, `color` are set to column names rather than arrays. Below is an example of this in action:"""

code_8 = """df = pd.DataFrame({
  "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
  "Contestant": ["Alex", "Alex", "Alex", "Jordan", "Jordan", "Jordan"],
  "Number Eaten": [2, 1, 3, 1, 3, 2],
})

import plotly.express as px

fig = px.bar(df, x="Fruit", y="Number Eaten", color="Contestant", barmode="group")
fig.show()"""

code_9 = """import plotly.graph_objects as go

fig = go.Figure()
for contestant, group in df.groupby("Contestant"):
    fig.add_trace(go.Bar(
        x=group["Fruit"], y=group["Number Eaten"], name=contestant,
        hovertemplate="Contestant=%s<br>Fruit=%%{x}<br>Number Eaten=%%{y}<extra></extra>"% contestant
    ))
fig.update_layout(legend_title_text="Contestant")
fig.update_xaxes(title_text="Fruit")
fig.update_yaxes(title_text="Number Eaten")
fig.show()"""

df = pd.DataFrame({
  "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
  "Contestant": ["Alex", "Alex", "Alex", "Jordan", "Jordan", "Jordan"],
  "Number Eaten": [2, 1, 3, 1, 3, 2],
})
fig_3 = px.bar(df, x="Fruit", y="Number Eaten", color="Contestant", barmode="group")
fig_4 = go.Figure()
for contestant, group in df.groupby("Contestant"):
    fig_4.add_trace(go.Bar(
        x=group["Fruit"], y=group["Number Eaten"], name=contestant,
        hovertemplate="Contestant=%s<br>Fruit=%%{x}<br>Number Eaten=%%{y}<extra></extra>"% contestant
    ))
fig_4.update_layout(legend_title_text="Contestant")
fig_4.update_xaxes(title_text="Fruit")
fig_4.update_yaxes(title_text="Number Eaten")


st.header("Plotly Express vs. Graph Objects")
st.markdown(markdown_7)
st.code(code_8, language="python")
st.plotly_chart(fig_3)
st.code(code_9, language="python")
st.plotly_chart(fig_3)




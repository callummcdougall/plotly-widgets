import streamlit as st

from pathlib import Path

showWarningOnDirectExecution = False
showErrorDetails = False

ROOT_FILE = "./"

st.set_page_config(
    layout="wide",
    page_title="Plotly & Widgets",
    page_icon="📊",
    menu_items={
        "Get help": "https://www.perfectlynormal.co.uk/",
        "About": "##### This was created to demo the Python libraries of Plotly and IPyWidgets, and show how they can be combined to create interactive output in Python notebooks."
    }
)



markdown_1 = """

# Welcome! 👋

This is a Streamlit app with example [Plotly](https://plotly.com/python/) and [IPyWidgets](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20Basics.html) code. Most of it was taken from the respective websites' documentation pages; some of it was written by me.

**`widgets`** are python objects that can be visually represented in the notebook, e.g. sliders, textboxes or dropdowns. They can be used to build interactive GUIs.

**`plotly`** is a library for producing interactive graphs. It also matches all the features of a conventional library like matplotlib, and (in my opinion) can be a lot more intuitive to use. There are 2 main ways to create graphs in plotly: using **`plotly graph objects`** (imported as **`go`**, this is more low-level, and gives you more control over the graph) and **`plotly express`** (imported as **`px`**, this is useful for easily producing nice-looking graphs with very little code).

Plotly and widgets can be combined to create interactive output that you control the appearance of (e.g. using dropdowns to select different groups of data to show on a graph).

---

### The purpose of this guide

This guide has two main purposes. 

The first is to **provide a (relatively) condensed summary** of what I see as the most useful parts of both libaries, without having to scroll through endless tutorials and documentation pages.

The second is to **provide a useful resource to find code** which produces the plot / widget you want. 

Depending on what you're hoping to use this for, you might prefer using the Jupyter Notebook, where you can actually run the cells."""

markdown_2 = """---

### How to read this

You can use the menu on the left for navigation, and scroll through the pages. There will be several interactive plots which should work even on this site (sidenote - Streamlit is great!). 

However the IPython widgets won't work, so if you want to try these out rather than just reading about them then you'll need to use the notebook.

---

### Contents

* **`plotly`**
    * **Introduction** — describes what Plotly is in more detail, and goes over the core Plotly syntax
    * **Basic Features** — shows how to add features like legend, titles, error bars and hover text to graphs
    * **Chart Cypes** — showcases a variety of different plotly charts (e.g. line, scatter, heatmap)
* **`widgets`**
    * **Introduction** — gives an overview of what widgets are, goes over the core syntax for widgets
    * **Widget Types** — showcases most available widgets
    * **Layout** — shows how to display widgets together, and give them a nice layout
    * **Output, Interactions & Events** — shows how to use widgets in other functions' input and output
* **`plotly & widgets`** — showcases a few examples of widgets being used as a GUI to interact with plotly graphs

---

If you find this guide useful, I'd love to know (and also if you create any cool graphs that you think could be added to this notebook, please send them!).

Happy plotting!
"""

st.markdown(markdown_1)

with open("WidgetsPlotlyGuideTrimmed.ipynb", "rb") as file:
    st.download_button(data=file, label="Download Jupyter Notebook", file_name="Plotly-Widgets-tutorial.ipynb")

st.markdown(markdown_2)

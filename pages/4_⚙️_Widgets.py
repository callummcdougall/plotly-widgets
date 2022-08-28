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

st.title("Widgets")

markdown = """Unfortunately widgets don't work well with Streamlit's API. For this section, I'd recommend you use the [Jupyter Notebook](https://github.com/callummcdougall/plotly-widgets/blob/main/WidgetsPlotlyGuideTrimmed.ipynb) instead, so you can actually go through each of the widgets yourself.

Alternatively, you can skip to the next section to see some examples of what it looks like when Plotly and Widgets work together (I have made these ones interactive).

Below, I've included a few screenshots from the notebook, showing the kinds of things you can do with widgets."""

st.markdown(markdown)

for i in range(1, 9):
    st.markdown("---")
    st.image(f"images/w{i}.png")
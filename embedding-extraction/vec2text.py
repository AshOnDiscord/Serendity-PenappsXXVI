import requests
import pandas as pd
import numpy as np
import ast
import umap
from sklearn.cluster import KMeans

url = "http://localhost:5000/all_data"
response = requests.get(url)


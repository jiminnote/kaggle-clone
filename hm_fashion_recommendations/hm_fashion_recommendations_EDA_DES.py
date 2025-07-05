import os

import numpy as np
import datatable as dt # Pandas처럼 데이터를 다루는 라이브러리인데, 속도가 빠르고 대용량 처리에 최적화

import pandas as pd
from scipy import stats

# plotly, matplotlib, seaborn은 그래프 시각화
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)
pio.templates.default = "none"
import cv2 # cv2는 Python에서 OpenCV(Open Source Computer Vision Library) 를 사용하기 위해 불러오는 명령어


import warnings
warnings.filterwarnings('ignore')

path = '/kaggle/input/h-and-m-personalized-fashion-recommendations'

#  datatable로 불러와서 .to_pandas()로 판다스로 변환
sample_sumbission = dt.fread(path+'/sample_submission.csv').to_pandas()
art = dt.fread(path+'/articles.csv').to_pandas()
trans_train = dt.fread(path+'/transactions_train.csv').to_pandas()
cust = dt.fread(path+ '/customers.csv').to_pandas()

## 1. Articles 

#주피터 환경에서만 가능
display(art.shape)
display(art.head(5))


art.info() # 데이터프레임의 행 수, 열 수, 컬럼별 결측치, 데이터 타입 등을 확인할 수 있는 함수 / EDA(탐색적 데이터 분석)의 시작점으로 자주 사용


# 각 컬럼이 얼마나 다양한 값(범주/수치 등) 을 갖고 있는지 파악하기 위한 for문
from termcolor import colored
for col in art.columns:
    x = art[col].nunique()  # numique() : unique한 값 몇 개인지 세는 함수 / 각 컬럼을 순회하면서 컬럼에 들어있는 데이터를  distinct 하여 count 한다고 보면된다.
    print("{}: {} unique values".format(col, colored(x, 'white'))) 
    # colored(x, 'white') : 출력할 때 값 x를 흰색 글씨로 표시 (터미널 색상 강조용)


### 1.1 Product Type Name

fig = px.histogram(art, x="product_type_name",
                   width=900, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Product Type Name ", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') # 막대 순서를 높은 값부터 정렬


# 상위 3개만 진하게(crimson), 나머지 연한 회색
colors = ['lightgray'] * 100  
colors[0] = 'crimson'  
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors,)  # 막대 색상 적용
fig.show()

### 1.2 Product Group Name

fig = px.histogram(art, x="product_group_name",
                   width=900, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Product Group Name ", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )
fig.show()

### 1.3 Product Index Name/ Index Group Name

fig = px.histogram(art, x="index_name",
                   width=600, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Index Name ", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') 

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )

fig.show()


fig = px.histogram(art, x="index_group_name",
                   width=600, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Index Group Name ", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending') # ordering the x-axis values

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )

fig.show()

### 1.4 Graphical Appearance Name

fig = px.histogram(art, x="graphical_appearance_name",
                   width=700, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Graphical Appearance Name", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 
fig.update_traces(marker_color=colors, 
                )

fig.show()


### 1.5 Garment Group Name

fig = px.histogram(art, x="garment_group_name",
                   width=700, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Garment Group Name", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )

fig.show()

### 1.6 Perceived Color Value Name

fig = px.histogram(art, x="perceived_colour_value_name",
                   width=700, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Perceived Color Value Name", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )

fig.show()

### 1.7 Section Name

fig = px.histogram(art, x="section_name",
                   width=700, 
                   height=400,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Section Name", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 100  
colors[0] = 'crimson' 
colors[1] = 'crimson' 
colors[2] = 'crimson' 


fig.update_traces(marker_color=colors, 
                )

fig.show()

## 2. Customers

display(cust.shape)
display(cust.info())

for col in cust.columns:
    x = cust[col].nunique()    
    print("{}: ======> {} unique values".format(col, colored(x, 'blue')))

cust.head()

# 결측값 처리
cust['FN'] = cust['FN'].fillna(0) # FN 컬럼의 결측값 0 으로 채움
cust['Active'] = cust['Active'].fillna(0) # Active 컬럼의 결측값 0 으로 채움
cust['club_member_status'] = cust['club_member_status'].fillna('na')  # club_member_status 컬럼의 결측값 na 으로 채움

### 2.1 Age of Customers

fig = px.histogram(cust, x="age",
                   width=700, 
                   height=400,
                   histnorm='percent',
                   template="simple_white",
                   color='Active',
                   color_discrete_sequence =['gray', 'crimson']
                   )

fig.update_layout(title="Age of customers", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_yaxes(categoryorder='total ascending') 

fig.show()

### 2.2 Club Member Status

fig = px.histogram(cust, x="club_member_status",
                   width=600, 
                   height=350,
                   histnorm='percent',
                   template="simple_white"
                   )

fig.update_layout(title="Club Member Status", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

colors = ['lightgray'] * 10  
colors[0] = 'crimson'

fig.update_traces(marker_color=colors, 
                )

fig.show()

### 2.3 Fashion News Frequency

fig = px.histogram(cust, x="fashion_news_frequency",
                   width=700, 
                   height=350,
                   histnorm='percent',
                   template="simple_white",
                   color='Active',
                   barmode='group',
                   color_discrete_sequence =['gray', 'crimson']
                   )

fig.update_layout(title="Fashion News Frequency", 
                  font_family="San Serif",
                  titlefont={'size': 20},
                  legend=dict(
                  orientation="v", y=1, yanchor="top", x=1.0, xanchor="right" )                 
                 ).update_xaxes(categoryorder='total descending')

fig.show()

## 3.Sample product pictures


def getImagePaths(path):
    image_names = []
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            fullpath = os.path.join(dirname, filename)
            image_names.append(fullpath)
    return image_names

def display_multiple_img(images_paths, rows, cols,title):
    
    figure, ax = plt.subplots(nrows=rows,ncols=cols,figsize=(16,8))
    plt.suptitle(title, fontsize=20)
    for ind,image_path in enumerate(images_paths):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        try:
            ax.ravel()[ind].imshow(image)
            ax.ravel()[ind].set_axis_off()
        except:
            continue
    plt.tight_layout()
    plt.show()

images_path = getImagePaths('../input/h-and-m-personalized-fashion-recommendations/images/')
display_multiple_img(images_path[0:25], 5, 5,"Sample product images")
display_multiple_img(images_path[200:220], 5, 4,"Sample product images")

## 4.Transactions

trans_train.head()

for col in trans_train.columns:
    x = trans_train[col].nunique()    
    print("{}: ======> {} unique".format(col, colored(x, 'red')))

trans_train.info()
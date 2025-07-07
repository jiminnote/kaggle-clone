import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.express as px

import plotly.graph_objs as go
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from pprint import pprint
import warnings
warnings.filterwarnings("ignore")
pd.options.display.float_format = '{:.2f}'.format

df_train = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/train.csv")
df_train["datetime"] = pd.to_datetime(df_train["datetime"])
# 소비 데이터만 필터링
df_train_consumption = df_train[df_train["is_consumption"]==1]
# 월/주/일별 소비 데이터 평균 계산
monthly_consumption = df_train_consumption.groupby(pd.Grouper(key="datetime", freq='M')).mean()
weekly_consumption = df_train_consumption.groupby(pd.Grouper(key="datetime", freq='W')).mean()
daily_consumption = df_train_consumption.groupby(pd.Grouper(key="datetime", freq='D')).mean()
mean_consumption = df_train_consumption.target.mean()
# 생산 데이터만 필터링
df_train_production = df_train[df_train["is_consumption"]==0]
# 월/주/일별 생산 데이터 평균 계산
monthly_production = df_train_production.groupby(pd.Grouper(key="datetime", freq='M')).mean()
weekly_production = df_train_production.groupby(pd.Grouper(key="datetime", freq='W')).mean()
daily_production = df_train_production.groupby(pd.Grouper(key="datetime", freq='D')).mean()
mean_production = df_train_production.target.mean()


# 단일 시계열 데이터를 area chart로 시각화하고 평균선을 함께 표시
def plot_one(df, mean, color, title, annotation, yaxis_title, y="target", line_shape="linear"):
    fig = px.area(df, x=df.index, 
                  y=y, title=title,
                  line_shape=line_shape)
    fig.add_hline(y=mean, line_dash="dot", 
                  annotation_text=annotation, 
                  annotation_position="bottom right")
    fig.update_traces(line_color=color)
    fig.update_layout(xaxis_title="Date",
                      yaxis_title=yaxis_title)
    return fig

df_train.info()

df_train.isna().sum()

df_train = df_train.dropna(how="any")

df_train.describe().T

plot_one(daily_consumption, mean_consumption, 
         "#FA163F", "Daily Consumption", 
         "Average Consumption", "Consumption")

plot_one(daily_production, mean_production, 
         "#427D9D", "Daily Production", 
         "Average Production", "Production")

net_consumption = daily_consumption["target"]- daily_production["target"]
plot_one(net_consumption, net_consumption.mean(), 
         "#EC8F5E", "Net Consumption (Comsumption-Production)", 
         "Average Net Consumption",
         "Net Consumption")

# 범주형 변수 간의 관계를 시각화 (Parallel Categories, 파이차트, 분포도 등)
parallel_diagram = df_train[['county', 'product_type', 'is_business', 'is_consumption']]
fig = px.parallel_categories(parallel_diagram, color_continuous_scale=px.colors.sequential.Inferno)
fig.update_layout(title='Parallel category diagram on Train Data set')
fig.show()


# 두 범주형 변수의 조합 분포를 원형 그래프로 시각화
def plot_pie(df, col1, col2):
    # 두 범주형 변수의 조합별 개수 파이차트로 시각화
    df_ = df.groupby([col1, col2])[col1].count().reset_index(name='counts')
    df_[col1+","+col2] = df_[col1].astype(str) + ',' + df_[col2].astype(str) 
    fig = px.pie(df_, values='counts', names=col1+","+col2, title=col1+' | '+col2)
    fig.update_layout(autosize=True,width=700, height=650, 
                      margin=dict(l=50,r=50, b=60, t=50, pad=4),
                      paper_bgcolor="LightSteelBlue", showlegend=True)
    fig.show()

plot_pie(df_train, "product_type", "is_business")

plot_pie(df_train, "product_type", "is_consumption")

# 주어진 두 범주형 변수의 조합 수를 카운트하고, 분포(히스토그램) 시각화
def plot_dist(df, col1, col2, color):    
    df_ = df.groupby([col1, col2])[col1].count().reset_index(name='counts')
    plt.figure(figsize=(5,5))
    plt.legend()
    sns.distplot(df_['counts'],label='counts', color=color)
    plt.show()

plot_dist(df_train, 'product_type', 'is_business', 'red')

plot_dist(df_train, 'product_type', 'is_consumption', 'red')

sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.histplot(data=df_train, x='county', hue='is_business', multiple='stack', bins=30, palette='viridis', alpha=0.7)
plt.xlabel('County')
plt.ylabel('Count')
plt.title('Prosumer Distribution by County')
plt.legend(title='Is Business', loc='upper right')
plt.show()

desc_columns = ['county','is_business','product_type','is_consumption']

fig, axs = plt.subplots(1, len(desc_columns), figsize=(5*len(desc_columns), 3))

for i, column in enumerate(desc_columns):
    _ = sns.countplot(df_train, x=column, ax=axs[i])

_ = fig.tight_layout()

# 소비/생산 여부별 시간에 따른 평균 타겟값 계산
train_avgd = (
    df_train
    .groupby(['datetime','is_consumption'])
    ['target'].mean()
    .unstack()
    .rename({0: 'produced', 1:'consumed'}, axis=1)
)

fig, ax = plt.subplots(1, 1, figsize=(12, 4))
_ = train_avgd.plot(ax=ax, alpha=0.5)
_ = ax.set_ylabel('Energy consumed / produced')

fig,ax = plt.subplots(1,1,figsize=(6,4))
_ = train_avgd.resample('M').mean().plot(ax=ax, marker='.')
_ = ax.set_ylabel('Average monthly')

fig,ax = plt.subplots(1,1,figsize=(6,4))
train_avgd.groupby(train_avgd.index.hour).mean().plot(ax=ax, marker='.')
_ = ax.set_xlabel('Hour')

# 가스 가격 시계열 데이터 로딩 및 전처리 후 시각화
df_gas = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/gas_prices.csv")
df_gas.drop(["origin_date"], inplace=True, axis=1)
df_gas["forecast_date"] = pd.to_datetime(df_gas["forecast_date"])
monthly_gas = df_gas.groupby(pd.Grouper(key="forecast_date", freq='M')).mean()
weekly_gas = df_gas.groupby(pd.Grouper(key="forecast_date", freq='W')).mean()
daily_gas = df_gas.groupby(pd.Grouper(key="forecast_date", freq='D')).mean()
mean_gas_low = df_gas.lowest_price_per_mwh.mean()
mean_gas_high = df_gas.highest_price_per_mwh.mean()

# 두 개의 시계열 변수를 한 그래프에 area chart로 시각화하고 평균선도 함께 표시
def plot_two(df, 
             mean1, mean2, 
             color1, color2, 
             title, 
             annotation1, annotation2, 
             yaxis_title, 
             y1,y2,
             line_shape="linear"):
    fig = px.area(df, x=df.index, y=[y1, y2],
                 title=title,
                 color_discrete_map={y1: color1,
                                     y2: color2},
                 line_shape=line_shape)
    fig.add_hline(y=mean1, line_dash="dot",
                 annotation_text=annotation1, annotation_position="top right")
    fig.add_hline(y=mean2, line_dash="dot",
                 annotation_text=annotation2, annotation_position="bottom right")
    fig.update_layout(xaxis_title="Date", yaxis_title=yaxis_title,
                     legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

df_gas.info()

df_gas.describe().T

plot_two(daily_gas,
         mean_gas_low, mean_gas_high,
         "#EFB74F", "#247881",
         "Daily Price/MWh",
         "Average Low Price/MWh", "Average High Price/MWh", 
         "Euros/MWh",
         "lowest_price_per_mwh", "highest_price_per_mwh")

plot_two(weekly_gas,
         mean_gas_low, mean_gas_high,
         "#6499E9", "#6930C3",
         "Weekly Price/MWh",
         "Average Low Price/MWh", "Average High Price/MWh", 
         "Euros/MWh",
         "lowest_price_per_mwh", "highest_price_per_mwh")

# 기상 데이터 로딩 및 시계열로 평균값 산출 후 다양한 항목 시각화
df_historical = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/historical_weather.csv")
df_historical["datetime"] = pd.to_datetime(df_historical["datetime"])
monthly_historical = df_historical.groupby(pd.Grouper(key="datetime", freq='M')).mean()
weekly_historical = df_historical.groupby(pd.Grouper(key="datetime", freq='W')).mean()
daily_historical = df_historical.groupby(pd.Grouper(key="datetime", freq='D')).mean()
mean_historical_temp = df_historical.temperature.mean()
mean_historical_solar = df_historical.direct_solar_radiation.mean()
mean_historical_rain = df_historical.rain.mean()
mean_historical_snow = df_historical.snowfall.mean()
mean_historical_windspeed = df_historical.windspeed_10m.mean()
mean_historical_surface_pressure = df_historical.surface_pressure.mean()

df_historical.info()

df_historical.describe().T

plot_one(daily_historical,
         mean_historical_temp, 
         "#C59279", "Daily Historical Temperature", 
         "Average Historical Temperature",
         "Temperature",y="temperature")

plot_two(daily_historical,
         mean_historical_solar, mean_historical_solar,
         "#C70A80","#6499E9",
         "Daily Radiation Level",
         "Average Solar Radiation", "Average Solar Radiation",
         "Radiation Level",
         "direct_solar_radiation","shortwave_radiation")

plot_two(weekly_historical,
         mean_historical_solar, mean_historical_solar,
         "#C70A80","#6499E9",
         "Weekly Radiation Level",
         "Average Solar Radiation", "Average Solar Radiation",
         "Radiation Level",
         "direct_solar_radiation","shortwave_radiation",
         line_shape="spline")

plot_two(weekly_historical,
         mean_historical_rain, mean_historical_snow,
         "#325288","#565D47",
         "Weekly Rain/Snowfall",
         "Average Rainfall", "Average Snowfall",
         "Rain/Snow Level",
         "rain","snowfall",
         line_shape="spline")

plot_one(daily_historical,
         mean_historical_windspeed, 
         "#69C98D", "Daily Windspeed", 
         "Average Windspeed",
         "Windspeed (in m/s)",y="windspeed_10m",
          line_shape="spline")

plot_one(weekly_historical,
         mean_historical_windspeed, 
         "#1640D6", "Weekly Windspeed", 
         "Average Windspeed",
         "Windspeed (in m/s)",y="windspeed_10m",
          line_shape="spline")

fig = plot_one(daily_historical,
         mean_historical_surface_pressure, 
         "#D61640", "Daily Surface Pressure", 
         "Average Surface Pressure",
         "Pressure (in Hectopascals)",y="surface_pressure",
          line_shape="spline")
fig.update_layout(yaxis_range=[900,1100])

fig = plot_one(weekly_historical,
         mean_historical_surface_pressure, 
         "#26A620", "Weekly Surface Pressure", 
         "Average Surface Pressure",
         "Pressure (in Hectopascals)",y="surface_pressure",
          line_shape="spline")
fig.update_layout(yaxis_range=[900,1100])

df_historical["date"] = np.array(df_historical["datetime"], dtype="datetime64[D]")
df_historical

print(len(df_historical) - len(df_historical.drop_duplicates()))
location = df_historical[df_historical.duplicated()][["latitude", "longitude"]]
location.drop_duplicates()

# 고유 위경도 위치를 지도에 마킹하여 시각화
import folium
fmap = folium.Map((58.5, 25), zoom_start=7)
fmap.add_child(folium.LatLngPopup())
for i, (lat, lon) in df_historical[['latitude', 'longitude']].drop_duplicates().iterrows():
    popup = folium.Popup(f'({lat}, {lon}))', max_width=200)
    marker = folium.CircleMarker((lat, lon), radius=5, popup=popup, fill_color='#EC4074')
    marker.add_to(fmap)
fmap

# 이슬점보다 기온이 낮거나 같은 시간대를 계산 (안개/이슬 형성 가능성)
sum_list = []
sum_list2 = []
for i, (latitude, longitude) in df_historical[['latitude', 'longitude']].drop_duplicates().iterrows():
    mask1 = df_historical['latitude'] == latitude
    mask2 = df_historical['longitude'] == longitude
    location = df_historical[mask1 & mask2].reset_index(drop=True)
    df =  location.groupby('datetime')[['temperature', 'dewpoint']].mean()
    sum_list.append(sum(df['temperature'] <= df['dewpoint']))
    sum_list2.append(sum(df['temperature'] <= df['dewpoint']+1))
    
width = 14
print(f'total hour: {len(df)}')
print('temperature <= dewpoint')
for i in range(0, len(sum_list), width):
    print(sum_list[i:i+width])
print()
for i in range(0, len(sum_list), width):
    print(sum_list2[i:i+width])

# 주어진 기상 변수(col)를 시간 기준으로 시각화
def plot_historical_column(location, col):
    plt.figure(figsize=(10, 4))
    plt.plot(location.groupby('datetime')[[col]].mean(), label='hourly')
    plt.plot(location.groupby('date')[[col]].mean(), label='daily')
    plt.title(col)
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()

weather_gen = df_historical[['latitude', 'longitude']].drop_duplicates().iterrows()

i, (latitude, longitude) = next(weather_gen)
print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
mask1 = df_historical['latitude'] == latitude
mask2 = df_historical['longitude'] == longitude
location = df_historical[mask1 & mask2].reset_index(drop=True)

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plt.figure(figsize=(10, 4))
plt.plot(location.groupby('datetime')[['temperature']].mean(), label='temperature')
plt.plot(location.groupby('datetime')[['dewpoint']].mean(), label='dewpoint')
plt.title('hourly temperature vs dewpoint')
plt.xticks(rotation=25)
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(location.groupby('date')[['temperature']].mean(), label='temperature')
plt.plot(location.groupby('date')[['dewpoint']].mean(), label='dewpoint')
plt.title('daily temperature vs dewpoint')
plt.xticks(rotation=25)
plt.legend()
plt.show()

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'rain')
plot_historical_column(location, 'snowfall')

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'cloudcover_high')
plot_historical_column(location, 'cloudcover_mid')

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'cloudcover_low')
plot_historical_column(location, 'cloudcover_total')

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'windspeed_10m')
plot_historical_column(location, 'winddirection_10m')

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'shortwave_radiation')
plot_historical_column(location, 'direct_solar_radiation')

print(f'[{i}] latitude: {latitude}, longitude: {longitude}')
plot_historical_column(location, 'diffuse_radiation')
plot_historical_column(location, 'surface_pressure')

df_client = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/client.csv")
df_client

df_client.info()

df_client.describe().T

df_train_prediction_unit_id = df_train[["product_type", "county", "is_business"]].drop_duplicates()
df_client_prediction_unit_id = df_client[["product_type", "county", "is_business"]].drop_duplicates()
display(df_train_prediction_unit_id, df_client_prediction_unit_id)

# 학습 데이터에서 예측 단위 조합별 고유 prediction_unit_id 추출
df_train_prediction_unit_id = df_train[["product_type", "county", "is_business", "prediction_unit_id"]].drop_duplicates()
display(df_train_prediction_unit_id)

client_gen = df_client[["product_type", "county", "is_business"]].drop_duplicates().iterrows()

n_rows, n_cols = 3, 3
fig, axes = plt.subplots(n_rows, n_cols)
gen = zip([next(client_gen) for _ in range(n_rows * n_cols)], axes.ravel())
indexes = []
for (i, (product_type, county, is_business)), ax in gen:
    indexes.append(i)
    mask1 = df_client["product_type"] == product_type
    mask2 = df_client["county"] == county
    mask3 = df_client["is_business"] == is_business
    temp = df_client[mask1 & mask2 & mask3]
    
    ax.plot(temp["eic_count"].reset_index(drop=True))
    ax.set_ylabel("eic_count", color="blue")
    ax.set_xlabel("date")
    ax2 = ax.twinx()
    ax2.plot(temp["installed_capacity"].reset_index(drop=True), color="orange")
    ax2.set_ylabel("installed_capacity", color="orange")
print(f"prediction_unit_id: {indexes}")
plt.tight_layout()
plt.show()

parallel_diagram = df_client[['county', 'product_type', 'is_business']]
fig = px.parallel_categories(parallel_diagram, color_continuous_scale=px.colors.sequential.Inferno)
fig.update_layout(title='Parallel category diagram on client Data set')
fig.show()

plot_pie(df_client, "product_type", "is_business")

plot_dist(df_client, 'product_type', 'is_business', 'red')

df_electricity = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/electricity_prices.csv")
df_electricity["forecast_date"] = np.array(df_electricity["forecast_date"], dtype="datetime64")
df_electricity["date"] = np.array(df_electricity["forecast_date"], dtype="datetime64[D]")
df_electricity

df_electricity.info()

df_electricity.describe().T

hourly_electricity = df_electricity[["forecast_date","euros_per_mwh"]].set_index("forecast_date")
daily_electricity = df_electricity[["date","euros_per_mwh"]].groupby("date")["euros_per_mwh"].mean()
plt.figure(figsize=(11, 6))
plt.plot(hourly_electricity, label="hourly price")
plt.plot(daily_electricity, label="daily price")
plt.title("electricity price")
plt.xticks(rotation=25)
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(11, 6))
plt.plot(hourly_electricity[:48], label='hourly price')
plt.plot(daily_electricity[:3], label='daily price')
plt.title('electricity price')
plt.ylabel('euros_per_mwh')
plt.xticks(rotation=25)
plt.grid()
plt.legend()
plt.show()

df_electricity["time"] = df_electricity["forecast_date"].dt.strftime("%H:%M:%S")
fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [8, 1]}, sharey=True)
_ = sns.lineplot(df_electricity, x='forecast_date', y='euros_per_mwh', ax=axs[0])
_ = sns.boxplot(df_electricity, y='euros_per_mwh', ax=axs[1])
#_ = axs[1].get_yaxis().set_visible(False)
fig.tight_layout()

daily_elec_prices = (
    df_electricity[['forecast_date', 'euros_per_mwh']]
    .set_index('forecast_date')
    .resample('D')
    .mean()
)

fig, axs = plt.subplots(1, 2, figsize=(9, 4), gridspec_kw={'width_ratios': [8, 1]}, sharey=True)
_ = sns.lineplot(daily_elec_prices, x='forecast_date', y='euros_per_mwh', ax=axs[0])
_ = sns.boxplot(daily_elec_prices, y='euros_per_mwh', ax=axs[1])
#_ = axs[1].get_yaxis().set_visible(False)
fig.tight_layout()

df_forecast = pd.read_csv("/kaggle/input/predict-energy-behavior-of-prosumers/forecast_weather.csv")
df_forecast["forecast_datetime"] = np.array(df_forecast["forecast_datetime"], dtype="datetime64")
df_forecast["date"] = np.array(df_forecast["forecast_datetime"], dtype="datetime64[D]")
df_forecast

df_forecast.info()

df_forecast.describe().T

import folium
fmap = folium.Map((58.8, 25), zoom_start=7)
fmap.add_child(folium.LatLngPopup())
for i, (lat, lon) in df_forecast[["latitude", "longitude"]].drop_duplicates().iterrows():
    popup = folium.Popup(f"({lat}, {lon})", max_width=200)
    marker = folium.CircleMarker((lat, lon), radius=5, popup=popup, fill_color='#EC4074')
    marker.add_to(fmap)
fmap

sum_list = []
sum_list2 = []
for i, (latitude, longitude) in df_forecast[["latitude", "longitude"]].drop_duplicates().iterrows():
    mask1 = df_forecast["latitude"] == latitude
    mask2 = df_forecast["longitude"] == longitude
    location = df_forecast[mask1 & mask2].reset_index(drop=True)
    df = location.groupby('forecast_datetime')[["temperature", "dewpoint"]].mean()
    sum_list.append(sum(df["temperature"] <= df["dewpoint"]))
    sum_list2.append(sum(df["temperature"] <= df["dewpoint"] + 1))
    
width = 14
print(f"total: {len(df)} hours = 24 hours/day x 638 forecast days")
print()
print("[Number of times the expression 'temperature <= dewpoint' is satisfied]")
for i in range(0, len(sum_list), width):
    print(sum_list[i:i+width])
print("<The position of the number is equal to the position of the observation point>")
print()
print('temperature <= dewpoint + 1')
for i in range(0, len(sum_list), width):
    print(sum_list2[i:i+width])

# 주어진 기상 변수(col)를 시간 기준으로 시각화 (예보 데이터)
def plot_forecast_column(location, col):
    plt.figure(figsize=(10, 4))
    plt.plot(location.groupby('forecast_datetime')[[col]].mean(), label="hourly")
    plt.plot(location.groupby('date')[[col]].mean(), label="daily")
    plt.title(col)
    plt.xticks(rotation=25)
    plt.legend()
    plt.show()

#
# 고유 좌표쌍을 순회하기 위한 generator 정의 (예보 데이터 기준)
df_forecast_gen = df_forecast[["latitude", "longitude"]].drop_duplicates().iterrows()

i, (latitude, longitude) = next(df_forecast_gen)
print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
mask1 = df_forecast["latitude"] == latitude
mask2 = df_forecast["longitude"] == longitude
location = df_forecast[mask1 & mask2].reset_index(drop=True)

print(f'[{i} latitude: {latitude}, longitude: {longitude}]')
plt.figure(figsize=(10, 4))
plt.plot(location.groupby("forecast_datetime")[["temperature"]].mean(), label="temperature")
plt.plot(location.groupby("forecast_datetime")[["dewpoint"]].mean(), label="dewpoint")
plt.title("hourly temperature vs dewpoint")
plt.xticks(rotation=25)
plt.legend()
plt.show()

print(f'[{i} latitude: {latitude}, longitude: {longitude}]')
plt.figure(figsize=(10, 4))
plt.plot(location.groupby("date")[["temperature"]].mean(), label="temperature")
plt.plot(location.groupby("date")[["dewpoint"]].mean(), label="dewpoint")
plt.title("daily temperature vs dewpoint")
plt.xticks(rotation=25)
plt.legend()
plt.show()

print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
plot_forecast_column(location[:3000], "cloudcover_high")
plot_forecast_column(location[:3000], "cloudcover_mid")

print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
plot_forecast_column(location[:3000], "cloudcover_low")
plot_forecast_column(location[:3000], "cloudcover_total")

print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
plot_forecast_column(location[:3000], "10_metre_u_wind_component")
plot_forecast_column(location[:3000], "10_metre_v_wind_component")

print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
plot_forecast_column(location, "direct_solar_radiation")
plot_forecast_column(location, "surface_solar_radiation_downwards")

print(f"[{i}] latitude: {latitude}, longitude: {longitude}")
plot_forecast_column(location, "snowfall")
plot_forecast_column(location, "total_precipitation")

import json
# county ID를 사람이 읽을 수 있는 이름으로 매핑하기 위한 json 파일 로드
with open('/kaggle/input/predict-energy-behavior-of-prosumers/county_id_to_name_map.json', 'r') as f:
    json_data = json.load(f)
county_id_to_name_map = eval(json.dumps(json_data))
for key, value in county_id_to_name_map.items():
    print(key, value)

df_revealed = df_train[["county", "is_business", "product_type", "target", "is_consumption", "datetime", "prediction_unit_id"]].copy()

from colorama import Fore, Style, init;

def print_color(text:str, color = Fore.BLUE, style = Style.BRIGHT):
    '''Prints color outputs using colorama of a text string'''
    print(style + color + text + Style.RESET_ALL); 
    
# 데이터프레임의 크기 및 첫 번째 행 출력 유틸 함수
def display_df(df, name):
    '''Display df shape and first row '''
    print_color(text = f'{name} data has {df.shape[0]} rows and {df.shape[1]} columns. \n ===> First row:')
    display(df.head(1))

display_df(df_train, 'train')

display_df(df_client, 'client')

display_df(df_electricity, 'electricity')

display_df(df_forecast, 'forecast_weather')

display_df(df_gas, 'gas_prices')

display_df(df_historical, 'historical_weather')

display_df(df_revealed, 'revealed_targets')
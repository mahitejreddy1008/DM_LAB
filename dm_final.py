import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import streamlit as st
from statsmodels.tsa.seasonal import seasonal_decompose
from matplotlib.pylab import rcParams
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
st.set_option('deprecation.showPyplotGlobalUse', False)

def preprocess_data(df):
    # Drop unnecessary columns
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df = df.drop(['Area Code', 'Months Code', 'Element Code', 'Unit'], axis=1)

    # Filter rows with 'Temperature change' in the 'Element' column and months from January to December
    df_tempchange = df[df['Element'] == 'Temperature change']
    df_tempchange = df_tempchange[df_tempchange['Months'].isin(['January', 'February', 'March', 'April', 'May', 'June', 'July',
                                                               'August', 'September', 'October', 'November', 'December'])]

    # Rename columns and reshape the DataFrame using melt
    df_tempchange.columns = df_tempchange.columns.str.replace('Y', '')
    df_final = pd.melt(df_tempchange, id_vars=['Area', 'Months', 'Element'],
                       value_vars=[str(year) for year in range(1961, 2020)],
                       var_name='Year', value_name='Temperature')

    # Convert Year and Months columns to numeric and create Date and Decade columns
    df_final['Year'] = pd.to_numeric(df_final['Year'])
    month_to_num = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}
    df_final['Months_num'] = df_final['Months'].map(month_to_num)
    df_final['Year'] += (df_final['Months_num'] - 1) // 12
    df_final['Date'] = pd.to_datetime(df_final['Year'].astype(str) + '-' + df_final['Months_num'].astype(str), format='%Y-%m')
    df_final['Decade'] = (df_final['Date'].dt.year // 10) * 10

    return df_final


# In[ ]:


def mean_temp_country():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    st.title("Country wise mean temperature difference")
    country = st.selectbox('Select Country:', df_final['Area'].unique())
    mean_temperatures_by_country = df_final.groupby(['Area', 'Year'])['Temperature'].mean().reset_index()
    country_df = mean_temperatures_by_country[mean_temperatures_by_country['Area'] == country]
    if country_df.empty:
        print("Data not available for the selected country.")
        return
    
    # Plot mean temperature for the selected country using Plotly
    fig = px.line(country_df, x='Year', y='Temperature', 
                  title=f'Mean Temperature for {country}', labels={'Temperature': 'Mean Temperature Difference (°C)'})
    st.plotly_chart(fig)


# In[ ]:


def mean_temp_decade():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    st.title("Country wise mean temperature difference for each decade")
    country1 = st.selectbox('Select Country:', df_final['Area'].unique())
    mean_temperatures_by_decade = df_final.groupby(['Area', 'Decade'])['Temperature'].mean().reset_index()
    country_df = mean_temperatures_by_decade[mean_temperatures_by_decade['Area'] == country1]
    if country_df.empty:
        print("Data not available for the selected country.")
        return
    
    fig = px.line(country_df, x='Decade', y='Temperature', 
                  title=f'Mean Temperature for {country1}', labels={'Temperature': 'Mean Temperature difference (°C)'})
    st.plotly_chart(fig)


# In[ ]:


def generate_additional_plots():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)

    st.title("Visualization country wise")
    country2 = st.selectbox('Select Country:', df_final['Area'].unique())
    df_country = df_final[df_final['Area'] == country2]

    fig1 = px.bar(df_country.groupby('Year')['Temperature'].mean().reset_index(), x='Year', y='Temperature',
                  title=f'Average Temperature Change by Year in {country2}', labels={'Temperature': 'Average Temperature Change'})
    st.plotly_chart(fig1)

    fig2 = px.box(df_country, x='Year', y='Temperature',
                  title=f'Temperature Change Distribution by Year in {country2}', labels={'Temperature': 'Temperature Change'})
    st.plotly_chart(fig2)

    fig3 = px.histogram(df_country, x='Temperature',
                        title=f'Temperature Change Histogram in {country2}', labels={'Temperature': 'Temperature Change'})
    st.plotly_chart(fig3)

    heatmap_data = df_country.pivot_table(index='Year', columns='Months', values='Temperature', aggfunc='mean')
    months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    fig4 = px.imshow(heatmap_data,
                    labels=dict(x="Month", y="Year", color="Temperature"),
                    x=months_order,
                    y=list(range(df_country['Year'].min(), df_country['Year'].max() + 1)),
                    title=f'Temperature Change by Year and Month in {country2}')
    st.plotly_chart(fig4)


# In[ ]:


def top_10():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    st.title("Top 10 Countries which have Highest Temperature change")
    df1 = df_final.groupby(['Area', 'Year'])['Temperature'].mean().reset_index()
    df1['Year'] = pd.to_datetime(df1['Year'])
    df1 = df1.sort_values('Year',ascending=True)
    df1['Year'] = df1['Year'].dt.strftime('%m/%d/%Y')
    top_10 = df1.groupby('Area').sum().sort_values('Temperature', ascending=False)[:10].reset_index()['Area']
    st.write(top_10)


# In[ ]:


def prepare_model_data(df_final):
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    df_model = df_final.copy()
    df_model = df_model.sort_values(by=['Area', 'Year'])
    df_model['Temperature_Mean_Imputed'] = df_model.groupby('Area')['Temperature'].transform(lambda x: x.fillna(x.mean()))
    model_df = df_model.loc[df_model.Area == 'India']
    model_df.drop(['Area', 'Temperature', 'Months_num', 'Decade', 'Element'], axis=1, inplace=True)
    model_df.rename(columns={'Temperature_Mean_Imputed': 'Temperature'}, inplace=True)
    model_df.set_index('Date', inplace=True)
    
    return model_df


# In[ ]:


def time_series_analysis():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    model_df = prepare_model_data(df_final)
    st.title('Time Series Analysis')
    st.subheader('Null Values')
    null_counts = df_final.isnull().sum()
    fig, ax = plt.subplots()
    null_counts.plot(kind='bar')
    plt.xlabel('Columns')
    plt.ylabel('Count')
    plt.title('Number of Null Values in Each Column')
    st.pyplot(fig)
    
    rolling_mean = df_final['Temperature'].rolling(window = 12).mean()
    rolling_std = df_final['Temperature'].rolling(window = 12).std()
    # plt.plot(model_df['Temperature'], color = 'blue', label = 'Original')
    # plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
    # plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
    # plt.legend(loc = 'best')
    # plt.title('Rolling Mean & Rolling Standard Deviation')
    st.image("1.png")

    rcParams['figure.figsize'] = 15, 6
    seasonal_decomposing = seasonal_decompose(model_df['Temperature'], model='additive')
    seasonal_decomposing.plot()
    st.pyplot()
    
    def adf_test_and_display(series):
        result = adfuller(series)
        st.write('ADF Statistics:', result[0])
        st.write('p-value:', result[1])
        if result[1] <= 0.05:
            st.write("Stationary")
        else:
            st.write("Not stationary")

    adf_test_and_display(model_df['Temperature'])
    
    st.subheader("AutoCorrelation")
    fig, ax = plt.subplots(figsize=(12,6))
    ax=plot_acf(model_df['Temperature'], ax)
    st.write(ax)
    st.subheader("Partial AutoCorrelation")
    fig, ax = plt.subplots(figsize=(12,6))
    ax = plot_pacf(model_df['Temperature'], lags = 10)
    st.write(ax)



# In[ ]:


def train_sarima(data):
        model = SARIMAX(data, order=(1, 1, 1), seasonal_order=(1, 2, 1, 52))
        return model.fit()


# In[ ]:


def train_sarima_models(data):
    sarima_models = {}
    for month in data['Months'].unique():
        data_month = data[data['Months'] == month]['Temperature']
        sarima_models[month] = train_sarima(data_month)
    return sarima_models


# In[ ]:


def train_lstm(data):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(len(data_scaled)-12):
        X.append(data_scaled[i:i+12])
        y.append(data_scaled[i+12])
    X, y = np.array(X), np.array(y)
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(12, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=100, verbose=0)
    
    return model, scaler


# In[ ]:


def train_lstm_models(data):
    lstm_models = {}
    for month in data['Months'].unique():
        data_month = data[data['Months'] == month]['Temperature']
        lstm_models[month] = train_lstm(data_month)
    return lstm_models


# In[ ]:


def forecast_sarima(model, steps):
    return model.forecast(steps)


# In[ ]:


def forecast_lstm(model, scaler, data, steps):
    data_scaled = scaler.transform(data.values.reshape(-1, 1))
    input_data = data_scaled[-12:].reshape(1, 12, 1)
    forecast_scaled = []
    for _ in range(steps):
        forecast_scaled.append(model.predict(input_data)[0, 0])
        input_data = np.append(input_data[:, 1:, :], forecast_scaled[-1].reshape(1, 1, 1), axis=1)
    forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
    return forecast.flatten()


# In[ ]:


def forecasts(df_final, forecast_steps = 60):
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    model_df = prepare_model_data(df_final)

    sarima_models = train_sarima_models(df_final)
    lstm_models = train_lstm_models(df_final)

    forecast_sarima_values = {}
    forecast_lstm_values = {}

    for month in df_final['Months'].unique():
        sarima_model = sarima_models[month]
        lstm_model, scaler = lstm_models[month]
        data_month = df_final[df_final['Months'] == month]['Temperature']

        forecast_sarima_values[month] = forecast_sarima(sarima_model, forecast_steps)
        forecast_lstm_values[month] = forecast_lstm(lstm_model, scaler, data_month, forecast_steps)

    return forecast_sarima_values, forecast_lstm_values


# In[ ]:


def plot_forecast(actual, sarima_forecast, lstm_forecast, month):
    plt.figure(figsize=(10, 6))
    plt.plot(actual.index, actual.values, label='Actual', color='black')
    plt.plot(actual.index[-1] + pd.DateOffset(months=1) * np.arange(1, forecast_steps+1), sarima_forecast, label='SARIMA Forecast', linestyle='--', color='blue')
    plt.plot(actual.index[-1] + pd.DateOffset(months=1) * np.arange(1, forecast_steps+1), lstm_forecast, label='LSTM Forecast', linestyle='--', color='red')
    plt.title(f'Temperature Change Forecast for {month} in India')
    plt.xlabel('Year')
    plt.ylabel('Temperature Change')
    plt.legend()
    plt.grid(True)
    
    # Convert the Matplotlib plot to a Streamlit plot and display it
    st.pyplot()


# In[ ]:


def print_all_forecasts():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    model_df = prepare_model_data(df_final)
    st.title('Time Series Forecast Comparision')
    forecast_sarima_values, forecast_lstm_values = forecasts(df_final, forecast_steps = 60)
    for month in model_df['Months'].unique():
        actual_month = model_df[model_df['Months'] == month]['Temperature']
        plot_forecast(actual_month, forecast_sarima_values[month], forecast_lstm_values[month], month)


def all_forecast():
    st.title('Time Series Forecast Comparision')
    st.image("2.png")
    st.image("3.png")
    st.image("4.png")
    st.image("5.png")
    st.image("6.png")
    st.image("7.png")
    st.image("8.png")
    st.image("9.png")
    st.image("10.png")
    st.image("11.png")
    st.image("12.png")
    st.image("13.png")

# In[ ]:


def data():
    df = pd.read_csv('environment.csv', encoding = 'cp1252')
    df_final = preprocess_data(df)
    st.title('Data Information')
    st.subheader("Original Dataset")
    st.dataframe(df)
    st.write("data shape : ", df.shape)
    st.subheader("After preprocessing the Data")
    st.dataframe(df_final)
    st.write("data shape : ", df_final.shape)


# In[ ]:


def main():
    
        
    page = {
        "Country Mean Temperature" : mean_temp_country,
        "Mean Temperature Decade" : mean_temp_decade,
        "Visualization" : generate_additional_plots,
        "Top10" : top_10,
        "Time Series Analysis" : time_series_analysis,
        "Forecast" : all_forecast,
        "Data" : data,
    }
    st.sidebar.title("Weather Forecasting")
    pages = st.sidebar.selectbox("select the page :", page.keys())
    page[pages]()


# In[ ]:


if __name__ == "__main__":
    main()


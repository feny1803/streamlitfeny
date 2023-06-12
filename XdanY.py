import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

def linear_regression(x, y):
    x = sm.add_constant(x)  # Menambahkan konstanta
    model = sm.OLS(y, x).fit()
    return model.params[1], model.params[0], model.resid

def plot_boxplot(df):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df[['x', 'y']])
    plt.xlabel('Variabel')
    plt.ylabel('Nilai')
    plt.title('Boxplot X dan Y')
    st.pyplot(plt)

def plot_residual_analysis(y, y_pred):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Normal probability plot
    sm.ProbPlot(y - y_pred.flatten()).qqplot(line='s', ax=axes[0, 0])
    axes[0, 0].set_title('Normal Probability Plot')

    # Plot residual versus fitted
    sns.scatterplot(x=y_pred.flatten(), y=y.flatten() - y_pred.flatten(), ax=axes[0, 1])
    axes[0, 1].axhline(0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Fitted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Fitted')

    # Plot residual versus order
    sns.scatterplot(x=np.arange(len(y)), y=y.flatten() - y_pred.flatten(), ax=axes[1, 0])
    axes[1, 0].axhline(0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Observation Order')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals vs Order')

    # Histogram of residuals
    sns.histplot(y.flatten() - y_pred.flatten(), ax=axes[1, 1], kde=True)
    axes[1, 1].set_xlabel('Residuals')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Histogram of Residuals')

    plt.tight_layout()
    st.pyplot(plt)

def show_input_form():
    st.subheader('Input Angka X dan Y')
    n = st.number_input('Jumlah Data', min_value=1, step=1, value=10)
    
    data = {'x': [], 'y': []}
    
    for i in range(n):
        x = st.number_input(f'x{i+1}', key=f'x{i+1}')
        y = st.number_input(f'y{i+1}', key=f'y{i+1}')
        data['x'].append(x)
        data['y'].append(y)
    
    if st.button('Hitung'):
        df = pd.DataFrame(data)
        x = df['x'].values.reshape(-1, 1)
        y = df['y'].values.reshape(-1, 1)
        m, c, residuals = linear_regression(x, y)
        
        st.write(f'Koefisien Regresi (m): {m}')
        st.write(f'Intercept (c): {c}')
        
        return x, y, m, c, residuals

def show_main_menu():
    st.sidebar.title('Menu Utama')
    menu = st.sidebar.radio('Pilih Menu:', ('Analisis Regresi', 'Karakteristik Data', 'Pemeriksaan Residual'))

    if menu == 'Analisis Regresi':
        st.subheader('Analisis Regresi Linier')
        x, y, m, c, residuals = show_input_form()
        if x is not None and y is not None:
            st.subheader('Grafik Regresi Linier')
            plt.figure(figsize=(8, 6))
            plt.scatter(x, y)
            plt.plot(x, m * x + c, color='red')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title('Regresi Linier')
            st.pyplot(plt)
    elif menu == 'Karakteristik Data':
        st.subheader('Karakteristik Data')
        x, y, _, _, _ = show_input_form()
        if x is not None and y is not None:
            df = pd.DataFrame({'x': x.flatten(), 'y': y.flatten()})
            plot_boxplot(df)
    elif menu == 'Pemeriksaan Residual':
        st.subheader('Pemeriksaan Residual')
        x, y, m, c, residuals = show_input_form()
        if x is not None and y is not None:
            plot_residual_analysis(y, m * x + c)

if __name__ == '__main__':
    st.title('Aplikasi Analisis Regresi Linier')
    show_main_menu()


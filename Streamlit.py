import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv("datasets/df.csv", index_col=0)

st.title("Разведочный анализ данных")

st.header("Датасет", anchor=None)
st.dataframe(df)

st.header("Графики распределения признаков", anchor=None)

feature_1 = st.selectbox(
     'Выберите нужный признак, который хотите оценить в разбиении по таргету',
     ('AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS',
       'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
      'OWN_AUTO', 'FAMILY_INCOME', 'PERSONAL_INCOME',
       'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'))

if df[feature_1].dtypes == object:
    fig_1, ax_1 = plt.subplots()
    sns.histplot(data=df, x=feature_1)
    plt.title(f"Гистограмма {feature_1}")
    plt.ylabel('Количество объектов в выборке')
    plt.xticks(rotation=90)
    st.pyplot(fig_1)
else:
    fig_1, ax_1 = plt.subplots()
    sns.histplot(data=df, x=feature_1)
    plt.title(f"Гистограмма {feature_1}")
    plt.ylabel('Количество объектов в выборке')
    st.pyplot(fig_1)




st.header("Матрица корреляции", anchor=None)
cols_drop = list(df.select_dtypes(['object']).columns)
corr = df.drop(cols_drop + ['AGREEMENT_RK', 'ID_CLIENT'], axis=1).corr()
fig_heatmap, ax_heatmap = plt.subplots(figsize=(16, 8))
sns.heatmap(corr, annot = True, cmap='mako')
st.pyplot(fig_heatmap)




st.header("Гистограммы распределения признаков с разбиением по таргету", anchor=None)

feature_2 = st.selectbox(
     'Выберите нужный признак',
     ('AGE', 'GENDER', 'EDUCATION', 'MARITAL_STATUS', 'CHILD_TOTAL', 'DEPENDANTS',
       'SOCSTATUS_WORK_FL', 'SOCSTATUS_PENS_FL',
      'OWN_AUTO', 'FAMILY_INCOME', 'PERSONAL_INCOME',
       'LOAN_NUM_TOTAL', 'LOAN_NUM_CLOSED'))

if df[feature_2].dtypes == object:
    fig_2, ax_2 = plt.subplots()
    sns.histplot(data=df, x=feature_2, hue='TARGET', stat='probability', multiple="dodge", common_norm=False)
    plt.title(f"Гистограмма {feature_2}")
    plt.ylabel('Количество объектов в выборке')
    plt.xticks(rotation=90)
    st.pyplot(fig_2)
else:
    fig_2, ax_2 = plt.subplots()
    sns.histplot(data=df, x=feature_2, hue='TARGET', stat='probability', multiple="dodge", kde=True, common_norm=False)
    plt.title(f"Гистограмма {feature_2}")
    plt.ylabel('Количество объектов в выборке')
    st.pyplot(fig_2)

st.header("Числовые характеристики датасета", anchor=None)
st.table(df.drop(['AGREEMENT_RK', 'ID_CLIENT'], axis=1).describe())



import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import koreanize_matplotlib 
import matplotlib.pyplot as plt
koreanize_matplotlib.koreanize()  # 또는 그냥 import만 해도 됨

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="Google Play Store 분석", layout="wide")
st.title("📱 Google Play Store 앱 데이터 분석")
st.markdown("### 앱 설치 수, 평점, 용량의 관계를 시각적으로 분석해봅니다.")

# --- 데이터 로드 및 전처리 ---
@st.cache_data
def load_data():
    df = pd.read_csv("googleplaystore.csv")
    df = df[['App', 'Category', 'Rating', 'Reviews', 'Size', 'Installs', 'Type', 'Price']]
    
    df['Installs'] = df['Installs'].str.replace('+', '', regex=False).str.replace(',', '', regex=False)
    df['Installs'] = pd.to_numeric(df['Installs'], errors='coerce')
    
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
    df = df[(df['Rating'] >= 0) & (df['Rating'] <= 5)]
    
    df['Reviews'] = pd.to_numeric(df['Reviews'], errors='coerce')
    df['Price'] = df['Price'].astype(str).str.replace('$', '', regex=False)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

    def size_to_mb(size):
        try:
            if 'M' in size:
                return float(size.replace('M', ''))
            elif 'k' in size or 'K' in size:
                return float(size.replace('k', '').replace('K', '')) / 1024
            elif size == 'Varies with device':
                return np.nan
        except:
            return np.nan
        return np.nan

    df['Size_MB'] = df['Size'].apply(size_to_mb)
    df = df.dropna(subset=['Rating', 'Installs'])
    df['Log_Installs'] = df['Installs'].apply(lambda x: np.log1p(x))
    return df

df = load_data()

# --- 사이드바 필터 ---
st.sidebar.header("🔍 필터 옵션")
categories = st.sidebar.multiselect(
    "앱 카테고리를 선택하세요",
    options=sorted(df['Category'].dropna().unique()),
    default=None
)
if categories:
    df = df[df['Category'].isin(categories)]

st.markdown("---")

# --- 1. 평점 구간별 평균 설치 수 ---
st.subheader("1️⃣ 평점 구간별 설치 수 분석")
st.markdown("앱 평점을 구간별로 나누어 설치 수의 평균을 비교합니다.")

bins = np.arange(0, 5.5, 0.5)
labels = [f'{b:.1f}-{b+0.5:.1f}' for b in bins[:-1]]
df['Rating_Bin'] = pd.cut(df['Rating'], bins=bins, labels=labels, right=False)

grouped = df.groupby('Rating_Bin')['Log_Installs'].mean().reset_index()
grouped['Bin_Center'] = bins[:-1] + 0.25

fig1, ax1 = plt.subplots(figsize=(12, 6))
sns.scatterplot(data=df, x='Rating', y='Log_Installs', alpha=0.2, color='gray', ax=ax1, label='앱 분포')
ax1.plot(grouped['Bin_Center'], grouped['Log_Installs'], marker='o', color='crimson', label='구간별 평균')
ax1.set_xlabel("앱 평점")
ax1.set_ylabel("설치 수 (로그 변환)")
ax1.set_title("평점에 따른 설치 수 (구간 평균선 포함)")
ax1.legend()
st.pyplot(fig1)

# --- 2. 카테고리별 평균 설치 수 ---
st.subheader("2️⃣ 카테고리별 설치 수 분석")
st.markdown("앱 카테고리별 평균 설치 수 (로그 변환 기준)를 비교합니다.")

df_cat = df.dropna(subset=['Category'])
cat_grouped = df_cat.groupby('Category')['Log_Installs'].mean().reset_index().sort_values(by='Log_Installs', ascending=False)

fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.barplot(data=cat_grouped.head(15), x='Log_Installs', y='Category', palette='viridis', ax=ax2)
ax2.set_title("카테고리별 평균 설치 수 (상위 15개)")
st.pyplot(fig2)

# --- 3. 카테고리별 앱 용량 분석 ---
st.subheader("3️⃣ 카테고리별 앱 용량 분석")
st.markdown("앱 카테고리별 평균 용량(MB)을 비교합니다.")

df_size = df.dropna(subset=['Size_MB', 'Category'])
cat_size = df_size.groupby('Category')['Size_MB'].mean().reset_index().sort_values(by='Size_MB', ascending=False)

fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.barplot(data=cat_size.head(15), x='Size_MB', y='Category', palette='coolwarm', ax=ax3)
ax3.set_title("카테고리별 평균 앱 용량 (상위 15개)")
st.pyplot(fig3)

# --- 4. 앱 용량 vs 설치 수 관계 ---
st.subheader("4️⃣ 앱 용량과 설치 수의 관계")
st.markdown("앱 용량이 설치 수에 어떤 영향을 주는지 확인합니다.")

df_corr = df.dropna(subset=['Size_MB', 'Log_Installs'])
corr_val = df_corr[['Size_MB', 'Log_Installs']].corr().iloc[0, 1]

fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df_corr, x='Size_MB', y='Log_Installs', alpha=0.3, ax=ax4, label='앱 분포')
sns.regplot(data=df_corr, x='Size_MB', y='Log_Installs', scatter=False, color='red', ax=ax4, label='회귀선')
ax4.set_title(f"앱 용량 vs 설치 수 (상관계수: {corr_val:.2f})")
ax4.set_xlabel("앱 용량 (MB)")
ax4.set_ylabel("설치 수 (로그 변환)")
ax4.legend()
st.pyplot(fig4)

st.markdown("---")
st.caption("데이터 출처: Google Play Store Dataset on Kaggle")

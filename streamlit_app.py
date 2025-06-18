import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import koreanize_matplotlib 
import matplotlib.pyplot as plt
koreanize_matplotlib.koreanize()  # 한글 폰트 적용

# --- 페이지 기본 설정 ---
st.set_page_config(page_title="Google Play Store 분석", layout="wide")
st.title("📱 Google Play Store 앱 데이터 분석")
# st.markdown("### 앱 설치 수, 평점, 용량의 관계를 시각적으로 분석해봅니다.")

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
st.subheader("4️⃣ 설치 수 구간별 평균 앱 용량")
st.markdown("앱의 설치 수에 따라 평균 용량(MB)이 어떻게 달라지는지를 확인해봅니다.")

# 구간 설정
bins = [0, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, np.inf]
labels = ['<1천', '1천-1만', '1만-10만', '10만-100만', '100만-1천만', '1천만-1억', '1억 이상']
df_clean = df.dropna(subset=['Size_MB', 'Installs'])  # 필요한 열 필터링
df_clean['Install_Bin'] = pd.cut(df_clean['Installs'], bins=bins, labels=labels)

# 구간별 평균 앱 용량 계산
install_grouped = df_clean.groupby('Install_Bin')['Size_MB'].mean().reset_index()

# 꺾은선 그래프
fig, ax = plt.subplots(figsize=(10, 6))
sns.lineplot(data=install_grouped, x='Install_Bin', y='Size_MB', marker='o', sort=False, ax=ax)
ax.set_title('설치 수 구간별 평균 앱 용량 (꺾은선 그래프)')
ax.set_xlabel('설치 수 구간')
ax.set_ylabel('평균 앱 용량 (MB)')
ax.grid(True)
st.pyplot(fig)

# --- 5. 상위 15개 카테고리 평균 용량과 평균 설치 수 산점도 ---
st.subheader("5️⃣ 상위 15개 카테고리 평균 앱 용량과 설치 수")
st.markdown("카테고리별 평균 앱 용량(MB)과 평균 설치 수를 비교한 산점도입니다.")

# 카테고리별 평균 용량과 평균 설치수 집계 (Installs 원본 스케일)
df_top = df.dropna(subset=['Category', 'Size_MB', 'Installs'])
category_grouped = df_top.groupby('Category').agg({
    'Size_MB': 'mean',
    'Installs': 'mean'
}).reset_index()

top15_categories = category_grouped.sort_values(by='Installs', ascending=False).head(15)

fig5, ax5 = plt.subplots(figsize=(12, 6))
sns.scatterplot(
    data=top15_categories,
    x='Size_MB',
    y='Installs',
    hue='Category',
    s=150,
    palette='tab10',
    ax=ax5
)
ax5.set_title("상위 15개 카테고리 평균 앱 용량과 설치 수")
ax5.set_xlabel("평균 앱 용량 (MB)")
ax5.set_ylabel("평균 설치 수")
ax5.grid(True)
ax5.legend(title='카테고리', bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig5)




# --- 인사이트 요약 및 방향성 ---
st.markdown("""
---
# 🎯 분석 요약 및 1인 개발자·기획팀을 위한 전략 제안

## 분석 요약

| 항목 | 관찰된 경향 |
|------|-------------|
| **평점 vs 설치수** | 평점이 **4.0 이상이면** ‘괜찮은 앱’으로 인식되며, 이후 상승은 설치수에 **큰 영향 없음** |
| **용량 vs 설치수** | **용량이 크더라도 사용자 수요가 높으면 많이 설치됨** |
| **카테고리별 특성** | 변수(평점, 용량)의 영향력이 **카테고리에 따라 달라짐** |


### ✅ 결론
- **단순 변수 해석보다**  
  **사용자 타깃 · 카테고리 특성 · 시장 포지셔닝**을 고려한 전략이 앱 성공에 더 중요함

---

## 📍 1인 개발자 및 기획팀을 위한 실천적 방향

### 1. 목표 평점 전략화
- 목표 평점은 **4.0 이상 확보**에 집중하되, 그 이상은 **유지 전략**에 리소스를 분배
- 리뷰 관리를 통해 **초기 유저 피드백을 빠르게 반영**하여 임계 평점을 조기에 넘기는 것이 중요

---

### 2. 용량 최적화보다 가치 전달에 집중
- 무조건 용량을 줄이기보다, **타깃 유저에게 의미 있는 기능/콘텐츠 제공**에 집중
- 특히 **게임·생산성 앱** 등 고용량이 당연한 분야에선 **퀄리티로 기대 충족**이 우선

---

### 3. 카테고리 맞춤 전략 수립
- 앱의 **카테고리별 사용자 기대와 경쟁 상황**을 사전에 분석
- **게임**처럼 경쟁이 치열한 분야는 **차별화된 콘셉트와 UX**에 집중  
- **생산성/교육** 등 틈새시장은 **심플하고 명확한 기능 제공**이 효과적

---

### 4. 시장 포지셔닝과 유저층 명확화
- 앱 개발 전 반드시 다음 질문에 답해야 함:
  1. **누가 이 앱을 사용할 것인가?**  
  2. **어떤 문제를 해결하거나 욕구를 충족하는가?**

- 이를 통해 **기획–기능–마케팅**이 전략적으로 **일관된 흐름**을 가질 수 있음
""")



st.markdown("---")
st.caption("데이터 출처 Google Play Store Dataset on Kaggle")

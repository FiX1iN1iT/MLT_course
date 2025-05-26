import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Загрузка данных
@st.cache_data
def load_data():
    data = pd.read_csv('WineQT.csv')
    data = data.drop(columns=['Id'])
    data['quality_class'] = pd.cut(data['quality'], bins=[0, 4, 6, 10], labels=[0, 1, 2])
    X = data.drop(['quality', 'quality_class'], axis=1)
    y = data['quality_class']
    return X, y

X, y = load_data()

# Масштабирование данных
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Интерфейс Streamlit
st.title("Wine Quality Dataset")
st.write("""
### Модель: Random Forest Classifier
""")

# Слайдер для выбора гиперпараметра
n_estimators = st.slider(
    "Количество деревьев (n_estimators)", 
    min_value=10, 
    max_value=200, 
    value=100, 
    step=10
)

# Обучение модели с выбранным гиперпараметром
model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Оценка качества
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr')

# Вывод метрик
st.write("### Результаты модели")
st.write(f"**Accuracy:** {accuracy:.3f}")
st.write(f"**F1-score:** {f1:.3f}")
st.write(f"**ROC-AUC:** {roc_auc:.3f}")

# Вывод важности признаков
st.write("### Важность признаков")
feature_importance = pd.DataFrame({
    'Признак': X.columns,
    'Важность': model.feature_importances_
}).sort_values('Важность', ascending=False)
st.bar_chart(feature_importance.set_index('Признак'))

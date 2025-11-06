import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(
    page_title="Анализ данных о найме сотрудников",
    page_icon="👥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
st.title("👥 Анализ данных о найме сотрудников")
st.markdown("---")

# ==========================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================

def find_date_column(df):
    """Возвращает первый столбец, который похож на дату, или None."""
    date_columns = []
    for col in df.columns:
        if any(k in col.lower() for k in ['date', 'дата', 'time', 'время']):
            date_columns.append(col)
    if not date_columns:
        return None
    return date_columns[0]

# Функция для работы с фильтром по годам (обновлённая и более безопасная)
def apply_year_filter(df, selected_year):
    """Применяет фильтр по году к DataFrame"""
    if selected_year == "Все время":
        return df

    # normalize to int if possible
    try:
        selected_year = int(selected_year)
    except:
        pass

    date_col = find_date_column(df)
    if not date_col:
        st.warning("⚠️ Не найдены столбцы с датами для фильтрации по годам")
        return df

    try:
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        filtered_df = df[df[date_col].dt.year == selected_year].copy()
        st.info(f"📅 Применен фильтр по году: {selected_year}. Найдено {len(filtered_df)} записей из {len(df)}")
        return filtered_df
    except Exception as e:
        st.error(f"❌ Ошибка при применении фильтра по году: {e}")
        return df

# Функция для получения доступных годов
def get_available_years(df):
    """Получает список доступных годов из данных"""
    years = ["Все время"]
    date_col = find_date_column(df)
    if date_col:
        try:
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            available_years_list = sorted(df[date_col].dt.year.dropna().unique().astype(int))
            years.extend(available_years_list)
        except Exception as e:
            st.warning(f"⚠️ Не удалось извлечь годы из данных: {e}")
    return years

# Функция для автоматической загрузки встроенных данных
@st.cache_data
def load_builtin_data():
    """Загружает встроенный CSV файл с данными о найме"""
    try:
        csv_file = "merge-csv.com__68b9ee302f5dd.csv"
        encodings = ['utf-8', 'latin1', 'cp1251']
        df = None
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    csv_file,
                    encoding=encoding,
                    skiprows=3,
                    header=0,
                    engine='python'
                )
                st.success(f"✅ Встроенные данные загружены с кодировкой: {encoding}")
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.warning(f"Попытка с кодировкой {encoding} не удалась: {e}")
                continue
        if df is not None:
            return df
        else:
            st.error("❌ Не удалось загрузить встроенные данные")
            return None
    except Exception as e:
        st.error(f"❌ Ошибка при загрузке встроенных данных: {e}")
        return None

# Функция для загрузки и обработки данных
@st.cache_data
def load_data(uploaded_file):
    """Загружает CSV файл и возвращает DataFrame"""
    try:
        if uploaded_file is not None:
            encodings = ['utf-8', 'latin1', 'cp1251']
            for encoding in encodings:
                try:
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    return df
                except UnicodeDecodeError:
                    continue
            st.error("Не удалось прочитать файл. Попробуйте другой формат кодировки.")
            return None
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None

# ==========================
# АНАЛИТИЧЕСКИЕ БЛОКИ
# ==========================

def analyze_data(df):
    """Проводит базовый анализ данных"""
    st.subheader("📊 Общая информация о данных")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Общее количество записей", len(df))
    with col2:
        st.metric("Количество столбцов", len(df.columns))
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Пропущенные значения", missing_values)
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Размер данных (МБ)", f"{memory_usage:.2f}")

    st.subheader("📋 Структура данных")
    col_info_data = []
    for col in df.columns:
        try:
            missing_count = df[col].isnull().sum()
            unique_count = df[col].nunique()
            dtype_str = str(df[col].dtype)
            sample_values = df[col].dropna().head(3).astype(str).tolist()
            sample_str = ", ".join(sample_values) if sample_values else "N/A"
            col_info_data.append({
                'Столбец': col,
                'Тип данных': dtype_str,
                'Пропущено': missing_count,
                'Уникальных': unique_count,
                'Примеры': sample_str
            })
        except Exception as e:
            col_info_data.append({
                'Столбец': col,
                'Тип данных': 'Ошибка',
                'Пропущено': 0,
                'Уникальных': 0,
                'Примеры': f'Ошибка: {str(e)}'
            })
    col_info = pd.DataFrame(col_info_data)
    st.dataframe(col_info, width='stretch')
    return col_info

def detailed_hiring_analysis(df):
    """Детальный анализ найма с фокусом на источники и успешность"""
    st.subheader("🎯 Детальный анализ найма")
    hiring_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['hire', 'найм', 'принят', 'статус', 'результат', 'outcome', 'status']):
            hiring_columns.append(col)

    if hiring_columns:
        st.write(f"✅ Найдены столбцы найма: {hiring_columns}")
        main_hiring_col = hiring_columns[0]
        st.write(f"**Анализируем столбец:** {main_hiring_col}")

        hiring_dist = df[main_hiring_col].value_counts()
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Распределение статусов найма:**")
            st.dataframe(hiring_dist)
        with col2:
            fig = px.pie(values=hiring_dist.values, names=hiring_dist.index, title="Распределение статусов найма")
            st.plotly_chart(fig, width="stretch")

        st.subheader("🏆 Анализ успешных кандидатов")
        success_keywords = ['active', 'approved', 'найм', 'принят', 'успех']
        success_statuses = []
        for status in hiring_dist.index:
            status_lower = str(status).lower()
            if any(keyword in status_lower for keyword in success_keywords):
                success_statuses.append(status)

        if success_statuses:
            st.write(f"**Успешные статусы:** {success_statuses}")
            successful_df = df[df[main_hiring_col].isin(success_statuses)]
            st.write(f"**Количество успешных кандидатов:** {len(successful_df)}")

            if 'Worklist' in df.columns:
                st.write("**Должности успешных кандидатов:**")
                worklist_success = successful_df['Worklist'].value_counts()
                fig = px.bar(x=worklist_success.values, y=worklist_success.index, title="Должности успешных кандидатов", orientation='h')
                st.plotly_chart(fig, width="stretch")

            if 'State' in df.columns:
                st.write("**География успешных кандидатов:**")
                state_success = successful_df['State'].value_counts().head(10)
                fig = px.bar(x=state_success.values, y=state_success.index, title="Топ-10 штатов успешных кандидатов", orientation='h')
                st.plotly_chart(fig, width="stretch")

        st.subheader("📍 Анализ источников найма")
        source_columns = []
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in ['source', 'источник', 'recruiter', 'рекрутер']):
                source_columns.append(col)

        if source_columns:
            st.write(f"**Столбцы источников:** {source_columns}")
            for source_col in source_columns:
                st.write(f"**Анализ столбца:** {source_col}")
                source_dist = df[source_col].value_counts().head(10)
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Топ-10 источников:**")
                    st.dataframe(source_dist)
                with col2:
                    fig = px.pie(values=source_dist.values, names=source_dist.index, title=f"Распределение по {source_col}")
                    st.plotly_chart(fig, width="stretch")

                if success_statuses:
                    st.write("**Эффективность источников (отношение успешных):**")
                    source_effectiveness = {}
                    for source in source_dist.index:
                        if pd.notna(source) and source != "":
                            total_from_source = len(df[df[source_col] == source])
                            successful_from_source = len(df[(df[source_col] == source) & (df[main_hiring_col].isin(success_statuses))])
                            effectiveness = (successful_from_source / total_from_source) * 100 if total_from_source > 0 else 0
                            source_effectiveness[source] = effectiveness
                    sorted_effectiveness = dict(sorted(source_effectiveness.items(), key=lambda x: x[1], reverse=True))
                    fig = px.bar(x=list(sorted_effectiveness.values()), y=list(sorted_effectiveness.keys()), title="Эффективность источников найма (%)", orientation='h')
                    st.plotly_chart(fig, width="stretch")

        st.subheader("⏰ Временной анализ найма")
        time_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'дата', 'время', 'time', 'год', 'year'])]
        if time_columns:
            st.write(f"**Временные столбцы:** {time_columns}")
            for time_col in time_columns:
                try:
                    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                    df_time = df.dropna(subset=[time_col])
                    if len(df_time) > 0:
                        st.write(f"**Анализ столбца:** {time_col}")
                        df_time['Месяц'] = df_time[time_col].dt.to_period('M')
                        monthly_data = df_time.groupby(['Месяц', main_hiring_col]).size().unstack(fill_value=0)
                        recent_months = monthly_data.tail(24)
                        fig = px.line(recent_months, title=f"Тренд найма по месяцам ({time_col})", labels={'value': 'Количество', 'index': 'Месяц'})
                        st.plotly_chart(fig, width="stretch")
                        df_time['Год'] = df_time[time_col].dt.year
                        yearly_data = df_time.groupby(['Год', main_hiring_col]).size().unstack(fill_value=0)
                        fig = px.bar(yearly_data, title=f"Распределение по годам ({time_col})", barmode='group')
                        st.plotly_chart(fig, width="stretch")
                except Exception as e:
                    st.write(f"Не удалось проанализировать {time_col}: {e}")
    else:
        st.warning("Не найдены столбцы найма. Показываем все столбцы:")
        selected_col = st.selectbox("Выберите столбец для анализа:", df.columns)
        if selected_col:
            col_dist = df[selected_col].value_counts()
            col1, col2 = st.columns(2)
            with col1:
                st.write("Распределение значений:")
                st.dataframe(col_dist)
            with col2:
                fig = px.pie(values=col_dist.values, names=col_dist.index, title=f"Распределение в столбце {selected_col}")
                st.plotly_chart(fig, width="stretch")

def analyze_tenure(df):
    """Анализирует продолжительность работы сотрудников"""
    st.subheader("⏱️ Анализ продолжительности работы")
    tenure_columns = []
    for col in df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['tenure', 'стаж', 'длительность', 'duration', 'месяц', 'месяцев', 'лет']):
            tenure_columns.append(col)

    if tenure_columns:
        st.write(f"Найдены столбцы с продолжительностью работы: {tenure_columns}")
        for col in tenure_columns:
            st.write(f"**Анализ столбца: {col}**")
            tenure_stats = df[col].describe()
            st.write("Статистики продолжительности работы:")
            st.dataframe(tenure_stats)
            fig = px.histogram(df, x=col, title=f"Распределение продолжительности работы ({col})", labels={'x': col, 'y': 'Количество'})
            st.plotly_chart(fig, width="stretch")
            hiring_columns = [c for c in df.columns if any(keyword in c.lower() for keyword in ['hire', 'найм', 'принят', 'статус'])]
            if hiring_columns:
                hiring_col = hiring_columns[0]
                fig = px.box(df, x=hiring_col, y=col, title=f"Продолжительность работы по результатам найма")
                st.plotly_chart(fig, width="stretch")
    else:
        st.warning("Не найдены столбцы с продолжительностью работы")

def build_ml_model(df):
    """Строит модель машинного обучения для предсказания найма"""
    st.subheader("🤖 Машинное обучение: Предсказание найма")
    st.write("Выберите столбец для предсказания (целевая переменная):")
    target_col = st.selectbox("Целевая переменная:", df.columns)

    if target_col:
        st.write(f"Подготовка данных для предсказания: {target_col}")
        df_clean = df.dropna(subset=[target_col])
        unique_targets = df_clean[target_col].nunique()
        st.write(f"**Уникальных значений в целевой переменной:** {unique_targets}")

        if unique_targets < 2:
            st.error("❌ Недостаточно уникальных значений для построения модели (нужно минимум 2)")
            return

        target_counts = df_clean[target_col].value_counts()
        min_class_size = target_counts.min()

        if unique_targets > 100 or min_class_size < 2:
            st.warning(f"⚠️ **Проблема с данными:**")
            st.write(f"- Уникальных значений: {unique_targets}")
            st.write(f"- Минимальный размер класса: {min_class_size}")

            st.subheader("🔧 Варианты решения:")
            col1, col2 = st.columns(2)

            with col1:
                st.write("**1. Группировка редких классов**")
                min_samples = st.slider(
                    "Минимальное количество записей для класса:",
                    min_value=2, max_value=50, value=5,
                    help="Классы с меньшим количеством записей будут объединены в 'Другие'"
                )

                if st.button("Применить группировку"):
                    df_grouped = df_clean.copy()
                    target_counts = df_clean[target_col].value_counts()
                    frequent_classes = target_counts[target_counts >= min_samples].index
                    df_grouped[target_col] = df_grouped[target_col].apply(lambda x: x if x in frequent_classes else 'Другие')
                    st.success(f"✅ Группировка применена! Теперь {df_grouped[target_col].nunique()} классов")
                    new_counts = df_grouped[target_col].value_counts()
                    st.write("**Новое распределение классов:**")
                    st.dataframe(new_counts, width='stretch')
                    df_clean = df_grouped

            with col2:
                st.write("**2. Альтернативный анализ**")
                st.write("Вместо ML модели можно провести:")
                st.write("• Анализ корреляций")
                st.write("• Статистический анализ")
                st.write("• Визуализация зависимостей")
                if st.button("Перейти к корреляционному анализу"):
                    st.subheader("🔗 Корреляционный анализ")
                    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 1:
                        correlation_matrix = df_clean[numeric_cols].corr()
                        fig = px.imshow(correlation_matrix, title="Корреляционная матрица числовых переменных", color_continuous_scale='RdBu', aspect='auto')
                        st.plotly_chart(fig, width='stretch')
                        strong_correlations = []
                        for i in range(len(numeric_cols)):
                            for j in range(i+1, len(numeric_cols)):
                                corr_value = correlation_matrix.iloc[i, j]
                                if abs(corr_value) > 0.5:
                                    strong_correlations.append({
                                        'Переменная 1': numeric_cols[i],
                                        'Переменная 2': numeric_cols[j],
                                        'Корреляция': round(corr_value, 3)
                                    })
                        if strong_correlations:
                            st.write("**Сильные корреляции (>0.5):**")
                            st.dataframe(pd.DataFrame(strong_correlations), width='stretch')
                        else:
                            st.info("Сильных корреляций не найдено")
                    else:
                        st.warning("Недостаточно числовых столбцов для корреляционного анализа")
                    return

            if 'df_grouped' not in locals():
                st.info("👆 Выберите один из вариантов выше для продолжения")
                return

        st.write(f"**Минимальный размер класса:** {min_class_size}")
        st.success("✅ Данные подходят для машинного обучения!")

        st.write("**Распределение классов:**")
        target_counts = df_clean[target_col].value_counts()
        st.dataframe(target_counts, width='stretch')

        df_encoded = df_clean.copy()

        # Преобразование столбцов дат
        date_columns = []
        for col in df_encoded.columns:
            if df_encoded[col].dtype == 'datetime64[ns]' or 'date' in col.lower() or 'time' in col.lower():
                date_columns.append(col)

        for col in date_columns:
            if col != target_col:
                try:
                    df_encoded[f'{col}_year'] = df_encoded[col].dt.year
                    df_encoded[f'{col}_month'] = df_encoded[col].dt.month
                    df_encoded[f'{col}_day'] = df_encoded[col].dt.day
                    df_encoded[f'{col}_dayofweek'] = df_encoded[col].dt.dayofweek
                    df_encoded = df_encoded.drop(columns=[col])
                    st.info(f"📅 Столбец {col} преобразован в числовые признаки")
                except Exception as e:
                    st.warning(f"Не удалось обработать столбец с датой {col}: {e}")
                    df_encoded = df_encoded.drop(columns=[col])

        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        numerical_cols = df_encoded.select_dtypes(include=[np.number]).columns

        label_encoders = {}
        for col in categorical_cols:
            if col != target_col:
                try:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    label_encoders[col] = le
                except Exception as e:
                    st.warning(f"Не удалось закодировать столбец {col}: {e}")
                    df_encoded = df_encoded.drop(columns=[col])

        target_encoder = LabelEncoder()
        df_encoded[target_col] = target_encoder.fit_transform(df_encoded[target_col].astype(str))

        feature_cols = [col for col in df_encoded.select_dtypes(include=[np.number]).columns if col != target_col]

        if len(feature_cols) > 0:
            X = df_encoded[feature_cols]
            y = df_encoded[target_col]
            st.write(f"**Количество признаков:** {len(feature_cols)}")
            st.write(f"**Размер данных:** {len(X)} записей")

            st.write("**Используемые признаки:**")
            feature_info = pd.DataFrame({
                'Признак': feature_cols,
                'Тип': [str(df_encoded[col].dtype) for col in feature_cols],
                'Уникальных значений': [df_encoded[col].nunique() for col in feature_cols]
            })
            st.dataframe(feature_info, width='stretch')

            missing_values = X.isnull().sum().sum()
            if missing_values > 0:
                st.warning(f"⚠️ Найдено {missing_values} пропущенных значений в признаках")
                X = X.fillna(X.mean())
                st.info("✅ Пропущенные значения заполнены средними значениями")

            can_stratify = all(target_counts >= 2)
            if can_stratify:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            else:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                st.warning("⚠️ Используется разделение без stratify из-за недостаточного количества данных в некоторых классах")

            st.write(f"📊 Размер обучающей выборки: {len(X_train)}")
            st.write(f"📊 Размер тестовой выборки: {len(X_test)}")

            try:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                st.write("**Результаты модели:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("Отчет о классификации:")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, width='stretch')
                with col2:
                    st.write("Матрица ошибок:")
                    cm = confusion_matrix(y_test, y_pred)
                    fig = px.imshow(cm, labels=dict(x="Предсказанные", y="Фактические"),
                                    x=target_encoder.classes_, y=target_encoder.classes_,
                                    title="Матрица ошибок")
                    st.plotly_chart(fig, width='stretch')

                feature_importance = pd.DataFrame({'Признак': feature_cols, 'Важность': model.feature_importances_}).sort_values('Важность', ascending=False)
                st.write("**Важность признаков:**")
                fig = px.bar(feature_importance.head(10), x='Важность', y='Признак', title="Топ-10 важных признаков", orientation='h')
                st.plotly_chart(fig, width='stretch')

                st.write("**Предсказание для новых данных:**")
                st.write("Введите значения признаков для предсказания:")
                input_data = {}
                cols_per_row = 3
                for i, col in enumerate(feature_cols[:10]):
                    if i % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    with cols[i % cols_per_row]:
                        if col in categorical_cols:
                            unique_vals = df_clean[col].unique()
                            input_data[col] = st.selectbox(f"{col}:", unique_vals)
                        else:
                            input_data[col] = st.number_input(f"{col}:", value=float(df_clean[col].mean()))
                if st.button("Сделать предсказание"):
                    input_df = pd.DataFrame([input_data])
                    for col in categorical_cols:
                        if col in input_data and col in label_encoders:
                            try:
                                input_df[col] = label_encoders[col].transform([input_data[col]])[0]
                            except:
                                st.error(f"Ошибка кодирования для {col}")
                                continue
                    try:
                        prediction = model.predict(input_df)[0]
                        prediction_proba = model.predict_proba(input_df)[0]
                        predicted_class = target_encoder.inverse_transform([prediction])[0]
                        st.success(f"**Результат предсказания:** {predicted_class}")
                        st.write("**Вероятности классов:**")
                        proba_df = pd.DataFrame({'Класс': target_encoder.classes_, 'Вероятность': prediction_proba}).sort_values('Вероятность', ascending=False)
                        st.dataframe(proba_df, width='stretch')
                    except Exception as e:
                        st.error(f"Ошибка при предсказании: {e}")
            except Exception as e:
                st.error(f"Ошибка при обучении модели: {e}")
        else:
            st.error("❌ Недостаточно признаков для построения модели")
            st.write("**Возможные причины:**")
            st.write("• Все столбцы содержат только категориальные данные")
            st.write("• Столбцы с датами не удалось преобразовать")
            st.write("• Недостаточно числовых данных")

def advanced_data_analysis(df):
    """Расширенный анализ данных с дополнительными метриками"""
    st.subheader("🔍 Расширенный анализ данных")
    st.subheader("📊 Качество данных")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_cells = df.shape[0] * df.shape[1]
        missing_percentage = (df.isnull().sum().sum() / total_cells) * 100
        st.metric("Пропущенные данные", f"{missing_percentage:.1f}%")
    with col2:
        duplicate_rows = df.duplicated().sum()
        st.metric("Дублирующиеся строки", duplicate_rows)
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Числовые столбцы", numeric_cols)
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Категориальные столбцы", categorical_cols)

    if 'State' in df.columns:
        st.subheader("🗺️ Географический анализ")
        col1, col2 = st.columns(2)
        with col1:
            state_counts = df['State'].value_counts().head(15)
            fig = px.bar(x=state_counts.values, y=state_counts.index, title="Топ-15 штатов по количеству кандидатов", orientation='h')
            st.plotly_chart(fig, width="stretch")
        with col2:
            st.write("**Распределение по штатам:**")
            state_summary = pd.DataFrame({'Штат': state_counts.index, 'Кандидаты': state_counts.values, 'Процент': (state_counts.values / len(df)) * 100})
            st.dataframe(state_summary, width="stretch")

    if 'Last App Date' in df.columns:
        st.subheader("⏰ Временной анализ")
        try:
            df = df.copy()
            df['Last App Date'] = pd.to_datetime(df['Last App Date'], errors='coerce')
            df_time = df.dropna(subset=['Last App Date'])
            if len(df_time) > 0:
                df_time['День недели'] = df_time['Last App Date'].dt.day_name()
                df_time['Месяц'] = df_time['Last App Date'].dt.month_name()
                df_time['Год'] = df_time['Last App Date'].dt.year
                col1, col2 = st.columns(2)
                with col1:
                    day_counts = df_time['День недели'].value_counts()
                    fig = px.pie(values=day_counts.values, names=day_counts.index, title="Активность по дням недели")
                    st.plotly_chart(fig, width="stretch")
                with col2:
                    month_counts = df_time['Месяц'].value_counts()
                    fig = px.bar(x=month_counts.index, y=month_counts.values, title="Активность по месяцам")
                    st.plotly_chart(fig, width="stretch")
                yearly_trend = df_time['Год'].value_counts().sort_index()
                fig = px.line(x=yearly_trend.index, y=yearly_trend.values, title="Тренд найма по годам", labels={'x': 'Год', 'y': 'Количество заявок'})
                st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.warning(f"Не удалось проанализировать временные данные: {e}")

def hiring_effectiveness_analysis(df):
    """Анализ эффективности найма и факторов успеха"""
    st.subheader("🎯 Анализ эффективности найма")
    status_col = None
    for col in df.columns:
        if 'status' in col.lower():
            status_col = col
            break

    if status_col:
        st.write(f"**Анализируем эффективность по столбцу:** {status_col}")
        success_patterns = ['active', 'approved', 'hired', 'success']
        success_statuses = []
        for status in df[status_col].unique():
            if pd.notna(status):
                status_lower = str(status).lower()
                if any(pattern in status_lower for pattern in success_patterns):
                    success_statuses.append(status)

        if success_statuses:
            st.write(f"**Успешные статусы:** {success_statuses}")
            total_candidates = len(df)
            successful_candidates = len(df[df[status_col].isin(success_statuses)])
            overall_effectiveness = (successful_candidates / total_candidates) * 100
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Общая эффективность", f"{overall_effectiveness:.1f}%")
            with col2:
                st.metric("Успешных кандидатов", successful_candidates)
            with col3:
                st.metric("Общее количество", total_candidates)

            if 'Worklist' in df.columns:
                st.subheader("💼 Эффективность по должностям")
                position_effectiveness = {}
                for position in df['Worklist'].unique():
                    if pd.notna(position):
                        position_df = df[df['Worklist'] == position]
                        position_total = len(position_df)
                        position_successful = len(position_df[position_df[status_col].isin(success_statuses)])
                        effectiveness = (position_successful / position_total) * 100 if position_total > 0 else 0
                        position_effectiveness[position] = {'total': position_total, 'successful': position_successful, 'effectiveness': effectiveness}
                sorted_positions = sorted(position_effectiveness.items(), key=lambda x: x[1]['effectiveness'], reverse=True)
                effectiveness_df = pd.DataFrame([{'Должность': pos, 'Всего кандидатов': data['total'], 'Успешных': data['successful'], 'Эффективность (%)': round(data['effectiveness'], 1)} for pos, data in sorted_positions])
                st.dataframe(effectiveness_df, width="stretch")
                fig = px.bar(x=[pos for pos, _ in sorted_positions], y=[data['effectiveness'] for _, data in sorted_positions], title="Эффективность найма по должностям (%)", labels={'x': 'Должность', 'y': 'Эффективность (%)'})
                st.plotly_chart(fig, width="stretch")

            if 'Recruiter' in df.columns:
                st.subheader("👥 Эффективность рекрутеров")
                recruiter_effectiveness = {}
                for recruiter in df['Recruiter'].unique():
                    if pd.notna(recruiter) and recruiter != "":
                        recruiter_df = df[df['Recruiter'] == recruiter]
                        recruiter_total = len(recruiter_df)
                        recruiter_successful = len(recruiter_df[recruiter_df[status_col].isin(success_statuses)])
                        effectiveness = (recruiter_successful / recruiter_total) * 100 if recruiter_total > 0 else 0
                        recruiter_effectiveness[recruiter] = {'total': recruiter_total, 'successful': recruiter_successful, 'effectiveness': effectiveness}
                sorted_recruiters = sorted(recruiter_effectiveness.items(), key=lambda x: x[1]['effectiveness'], reverse=True)
                top_recruiters = sorted_recruiters[:10]
                fig = px.bar(x=[rec for rec, _ in top_recruiters], y=[data['effectiveness'] for _, data in top_recruiters], title="Топ-10 рекрутеров по эффективности (%)", labels={'x': 'Рекрутер', 'y': 'Эффективность (%)'})
                st.plotly_chart(fig, width="stretch")
                recruiter_df = pd.DataFrame([{'Рекрутер': rec, 'Всего кандидатов': data['total'], 'Успешных': data['successful'], 'Эффективность (%)': round(data['effectiveness'], 1)} for rec, data in top_recruiters])
                st.dataframe(recruiter_df, width="stretch")

def trends_and_patterns_analysis(df):
    """Анализ трендов и паттернов в данных"""
    st.subheader("📈 Анализ трендов и паттернов")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        st.subheader("🔗 Корреляции между числовыми переменными")
        correlation_matrix = df[numeric_cols].corr()
        fig = px.imshow(correlation_matrix, title="Корреляционная матрица числовых переменных", color_continuous_scale='RdBu', aspect='auto')
        st.plotly_chart(fig, width="stretch")
        strong_correlations = []
        for i in range(len(numeric_cols)):
            for j in range(i+1, len(numeric_cols)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.5:
                    strong_correlations.append({'Переменная 1': numeric_cols[i], 'Переменная 2': numeric_cols[j], 'Корреляция': round(corr_value, 3)})
        if strong_correlations:
            st.write("**Сильные корреляции (>0.5):**")
            st.dataframe(pd.DataFrame(strong_correlations), width="stretch")

    st.subheader("📊 Анализ распределений")
    if 'Score' in df.columns:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x='Score', title="Распределение оценок кандидатов", nbins=20)
            st.plotly_chart(fig, width="stretch")
        with col2:
            fig = px.box(df, y='Score', title="Распределение оценок (box plot)")
            st.plotly_chart(fig, width="stretch")

    if 'Worklist' in df.columns and 'State' in df.columns:
        st.subheader("🏢 Анализ по должностям и штатам")
        pivot_table = df.groupby(['Worklist', 'State']).size().unstack(fill_value=0)
        st.write("**Топ-5 штатов для каждой должности:**")
        for position in df['Worklist'].unique():
            if pd.notna(position) and position in pivot_table.index:
                position_data = pivot_table.loc[position].sort_values(ascending=False).head(5)
                fig = px.bar(x=position_data.values, y=position_data.index, title=f"Топ-5 штатов для {position}", orientation='h')
                st.plotly_chart(fig, width="stretch")

def create_dashboard(df):
    """Создает информативный дашборд с ключевыми метриками"""
    st.subheader("📊 Дашборд ключевых показателей")
    if len(df) > 0:
        date_col = find_date_column(df)
        if date_col:
            try:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors='coerce')
                min_date = df_temp[date_col].min()
                max_date = df_temp[date_col].max()
                if pd.notna(min_date) and pd.notna(max_date):
                    st.info(f"📅 **Период анализа:** {min_date.strftime('%d.%m.%Y')} - {max_date.strftime('%d.%m.%Y')}")
            except:
                pass

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        total_candidates = len(df)
        st.metric("Всего кандидатов", f"{total_candidates:,}")
    with col2:
        if 'Status' in df.columns:
            active_candidates = len(df[df['Status'].str.contains('Active|Approved', case=False, na=False)])
            st.metric("Активных/Принятых", active_candidates)
        else:
            st.metric("Активных/Принятых", "N/A")
    with col3:
        if 'State' in df.columns:
            unique_states = df['State'].nunique()
            st.metric("Уникальных штатов", unique_states)
        else:
            st.metric("Уникальных штатов", "N/A")
    with col4:
        if 'Worklist' in df.columns:
            unique_positions = df['Worklist'].nunique()
            st.metric("Уникальных должностей", unique_positions)
        else:
            st.metric("Уникальных должностей", "N/A")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if 'Recruiter' in df.columns:
            unique_recruiters = df['Recruiter'].nunique()
            st.metric("Уникальных рекрутеров", unique_recruiters)
        else:
            st.metric("Уникальных рекрутеров", "N/A")
    with col2:
        date_col = 'Last App Date' if 'Last App Date' in df.columns else None
        if date_col:
            try:
                df_tmp = df.copy()
                df_tmp[date_col] = pd.to_datetime(df_tmp[date_col], errors='coerce')
                date_range = df_tmp[date_col].max() - df_tmp[date_col].min()
                st.metric("Диапазон дат", f"{date_range.days} дней")
            except:
                st.metric("Диапазон дат", "N/A")
        else:
            st.metric("Диапазон дат", "N/A")
    with col3:
        missing_data = df.isnull().sum().sum()
        st.metric("Пропущенных значений", f"{missing_data:,}")
    with col4:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
        st.metric("Размер данных", f"{memory_usage:.1f} МБ")

    st.subheader("💡 Быстрые инсайты")
    col1, col2 = st.columns(2)
    with col1:
        if 'Status' in df.columns:
            st.write("**Топ-5 статусов:**")
            top_statuses = df['Status'].value_counts().head(5)
            for status, count in top_statuses.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {status}: {count:,} ({percentage:.1f}%)")
    with col2:
        if 'Worklist' in df.columns:
            st.write("**Топ-5 должностей:**")
            top_positions = df['Worklist'].value_counts().head(5)
            for position, count in top_positions.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {position}: {count:,} ({percentage:.1f}%)")

# ======== НОВОЕ: Сравнение до 4 лет =========
def compare_years_analysis(df, selected_years):
    """Сравнение до 4 лет по ключевым метрикам и трендам."""
    st.subheader("📆 Сравнение по годам")

    if not selected_years:
        st.info("Выберите до 4 лет слева, чтобы сравнить.")
        return
    if len(selected_years) > 4:
        st.warning("Можно выбрать максимум 4 года. Будут использованы первые 4.")
        selected_years = selected_years[:4]

    date_col = find_date_column(df)
    if not date_col:
        st.error("Не найден столбец с датами для сравнения лет.")
        return

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df['__year__'] = df[date_col].dt.year
    df['__month__'] = df[date_col].dt.to_period('M')

    cdf = df[df['__year__'].isin(selected_years)].copy()
    if cdf.empty:
        st.warning("Нет записей для выбранных лет.")
        return

    # Поиск статуса
    status_col = None
    for col in cdf.columns:
        if 'status' in col.lower():
            status_col = col
            break

    # KPI по годам
    kpi_rows = []
    for y in selected_years:
        ydf = cdf[cdf['__year__'] == y]
        total = len(ydf)
        active = eff = None
        if status_col is not None:
            active = ydf[status_col].astype(str).str.contains('Active|Approved', case=False, na=False).sum()
            eff = (active / total * 100) if total else 0
        kpi_rows.append({
            "Год": y,
            "Заявок": total,
            "Активных/Принятых": active if active is not None else "N/A",
            "Эффективность (%)": round(eff, 1) if eff is not None else "N/A"
        })
    st.dataframe(pd.DataFrame(kpi_rows), use_container_width=True)

    totals = cdf.groupby('__year__').size().reset_index(name='Количество')
    fig = px.bar(totals, x='__year__', y='Количество', title="Количество заявок по годам")
    st.plotly_chart(fig, use_container_width=True)

    monthly = (cdf.dropna(subset=['__month__'])
                 .groupby(['__month__', '__year__']).size()
                 .reset_index(name='Заявок')
               ).sort_values('__month__')
    fig = px.line(monthly, x='__month__', y='Заявок', color='__year__',
                  title="Месячная динамика по выбранным годам",
                  labels={'__month__': 'Месяц'})
    st.plotly_chart(fig, use_container_width=True)

    if status_col is not None:
        st.subheader("🧾 Распределение статусов по годам")
        status_df = (cdf.assign(_status=cdf[status_col].astype(str).fillna("Unknown"))
                       .groupby(['__year__', '_status']).size()
                       .reset_index(name='Count'))
        fig = px.bar(status_df, x='__year__', y='Count', color='_status',
                     title="Статусы по годам (стек)", barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

    if 'Worklist' in cdf.columns:
        st.subheader("💼 Топ должностей по годам")
        topN = 5
        for y in selected_years:
            ydf = cdf[cdf['__year__'] == y]
            if ydf.empty:
                continue
            top_positions = ydf['Worklist'].value_counts().head(topN)
            fig = px.bar(x=top_positions.values, y=top_positions.index,
                         orientation='h', title=f"Топ-{topN} должностей — {y}")
            st.plotly_chart(fig, use_container_width=True)

# ==========================
# ОБЩИЙ РЕНДЕР ИНТЕРФЕЙСА
# ==========================

def run_app(df):
    """Единое место для сайдбара, фильтров и роутинга страниц (чтобы не дублировать код)."""
    st.success(f"✅ Данные загружены! Размер: {df.shape[0]} строк × {df.shape[1]} столбцов")

    # Информация о загруженных данных
    st.info(f"""
    📊 **Загруженные данные:**
    - **Записей:** {df.shape[0]:,}
    - **Столбцов:** {df.shape[1]}
    - **Период:** авто-определение по дате
    - **Тип:** Данные о найме сотрудников
    """)

    # ---- Фильтр по годам ----
    st.sidebar.markdown("---")
    st.sidebar.title("📅 Фильтр по годам")

    available_years = get_available_years(df)  # includes "Все время"

    # Single-year filter (for regular pages)
    selected_year = st.sidebar.selectbox(
        "Год для одиночного анализа:",
        available_years,
        help="Выберите конкретный год или 'Все время' для анализа всех данных"
    )

    # Multi-year compare (up to 4)
    years_only = [y for y in available_years if y != "Все время"]
    compare_years_selected = st.sidebar.multiselect(
        "Сравнить годы (до 4):",
        years_only,
        max_selections=4,
        help="Выберите до 4 лет для сравнения"
    )

    # Apply single-year filter for non-compare pages
    filtered_df = apply_year_filter(df, selected_year)

    # ---- Навигация по разделам ----
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Разделы анализа")
    page = st.sidebar.radio(
        "Выберите раздел:",
        [
            "Дашборд", "Общий обзор", "Детальный анализ найма",
            "Эффективность найма", "Расширенный анализ",
            "Тренды и паттерны", "Продолжительность работы",
            "Машинное обучение", "Сравнение лет"
        ]
    )

    if page == "Дашборд":
        create_dashboard(filtered_df)
    elif page == "Общий обзор":
        analyze_data(filtered_df)
    elif page == "Детальный анализ найма":
        detailed_hiring_analysis(filtered_df)
    elif page == "Эффективность найма":
        hiring_effectiveness_analysis(filtered_df)
    elif page == "Расширенный анализ":
        advanced_data_analysis(filtered_df)
    elif page == "Тренды и паттерны":
        trends_and_patterns_analysis(filtered_df)
    elif page == "Продолжительность работы":
        analyze_tenure(filtered_df)
    elif page == "Машинное обучение":
        build_ml_model(filtered_df)
    elif page == "Сравнение лет":
        # Для сравнения используем исходный df (без одиночного фильтра)
        compare_years_analysis(df, compare_years_selected)

# ==========================
# MAIN
# ==========================

def main():
    st.sidebar.title("📁 Данные")

    # 1) Пытаемся автозагрузить встроенный файл
    df = load_builtin_data()

    if df is not None:
        run_app(df)

        # Дополнительные данные (второй файл)
        st.sidebar.markdown("---")
        st.sidebar.title("📤 Дополнительные данные")
        uploaded_file = st.sidebar.file_uploader(
            "Или загрузите свой CSV файл",
            type=['csv'],
            help="Загрузите дополнительный CSV файл для сравнения"
        )
        if uploaded_file is not None:
            st.info("📤 Дополнительный файл загружен. Используйте его для сравнения с основными данными.")
    else:
        st.error("❌ Не удалось загрузить встроенные данные")
        st.info("👆 Попробуйте загрузить CSV файл вручную")

        # 2) Фолбэк: ручная загрузка
        uploaded_file = st.sidebar.file_uploader(
            "Выберите CSV файл",
            type=['csv'],
            help="Загрузите CSV файл с данными о найме сотрудников"
        )
        if uploaded_file is not None:
            df = load_data(uploaded_file)
            if df is not None:
                run_app(df)

if __name__ == "__main__":
    main()

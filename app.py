import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuración de la página
st.set_page_config(
    page_title="Gemelo Digital Turbofan - Universidad Javeriana",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Gemelo Digital Turbofan - Monitoreo de RUL")
st.markdown("---")

# Carga de datos de predicción
@st.cache_data
def load_data():
    df_pred = pd.read_csv('Predicciones_RUL_IA.csv')
    return df_pred

try:
    df_pred = load_data()

    # Identificar nombres de columnas
    col_id = 'engine_id' if 'engine_id' in df_pred.columns else df_pred.columns[0]
    col_real = 'RUL_Real' if 'RUL_Real' in df_pred.columns else df_pred.columns[1]
    col_pred = 'RUL_Predicho' if 'RUL_Predicho' in df_pred.columns else df_pred.columns[2]
    
    # Calcular columna de Error Absoluto de forma segura
    df_pred['Error_Calculado'] = (df_pred[col_real] - df_pred[col_pred]).abs()
    
    # Barra lateral - Selector de Motor
    st.sidebar.header("🕹️ Panel de Control")
    motores_disponibles = sorted(df_pred[col_id].unique())
    
    selected_engine = st.sidebar.selectbox(
        "Seleccionar ID del Motor:",
        options=motores_disponibles
    )

    # Filtrar datos para el motor seleccionado
    motor_data = df_pred[df_pred[col_id] == selected_engine]

    # Métricas en la barra lateral
    rul_real = motor_data[col_real].values[0]
    rul_pred = motor_data[col_pred].values[0]
    error_abs = abs(rul_real - rul_pred)

    st.sidebar.markdown("---")
    st.sidebar.subheader(f"📊 Diagnóstico Motor #{selected_engine}")
    st.sidebar.metric(label="RUL Real (Ciclos)", value=f"{int(rul_real)}")
    st.sidebar.metric(label="RUL Predicho (IA)", value=f"{int(rul_pred)}", delta=f"{int(rul_pred - rul_real)} ciclos error")
    st.sidebar.metric(label="Error Absoluto", value=f"{error_abs:.2f} ciclos")

    # Layout de Gráficas principales en dos columnas
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Comparativa RUL Real vs. Predicho")
        
        fig_line = go.Figure()
        
        # Línea RUL Real
        fig_line.add_trace(go.Scatter(
            x=df_pred[col_id], y=df_pred[col_real],
            mode='lines', name='RUL Real', line=dict(color='#1f77b4', width=1.5)
        ))
        
        # Línea RUL Predicho
        fig_line.add_trace(go.Scatter(
            x=df_pred[col_id], y=df_pred[col_pred],
            mode='lines', name='RUL Predicho', line=dict(color='#aec7e8', width=1.5)
        ))
        
        # Marcador del motor seleccionado (Estrella Roja)
        fig_line.add_trace(go.Scatter(
            x=[selected_engine], y=[rul_pred],
            mode='markers', name=f'Motor #{selected_engine} Seleccionado',
            marker=dict(color='red', size=14, symbol='star')
        ))

        fig_line.update_layout(
            xaxis_title="ID del Motor",
            yaxis_title="Ciclos RUL",
            legend=dict(x=0.55, y=0.99),
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col2:
        st.subheader("📉 Distribución del Error de Predicción")
        
        fig_hist = px.histogram(
            df_pred, 
            x='Error_Calculado',
            nbins=15,
            color_discrete_sequence=['#1f77b4']
        )
        fig_hist.update_layout(
            xaxis_title="Error Absoluto (Ciclos)",
            yaxis_title="Cantidad de Motores",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

except Exception as e:
    st.error(f"Error al cargar la aplicación: {e}")
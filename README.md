# Airline Passenger Satisfaction Prediction ✈️🧳
*(Predicción de Satisfacción de Pasajeros de Aerolíneas)*

This repository contains an end-to-end Machine Learning project focused on analyzing and predicting airline passenger satisfaction based on survey data from 129,880 customers. By implementing and optimization-tuning a Decision Tree classifier, this project identifies the operational features that hold the highest leverage over user sentiment.

Este repositorio contiene un proyecto de Machine Learning de extremo a extremo enfocado en analizar y predecir la satisfacción de los pasajeros de una aerolínea basándose en datos de encuestas de 129,880 clientes. Mediante la implementación y optimización de un clasificador de Árbol de Decisión, este proyecto identifica las características operativas con mayor impacto en el sentimiento de los usuarios.

---

## 📌 Project Overview / Descripción del Proyecto

- **Objective / Objetivo:** Build a predictive classification model to determine whether a future customer will be satisfied or dissatisfied with their flight experience, pinpointing the most critical drivers of satisfaction.
- **Data / Datos:** 129,880 customer survey responses evaluating features such as seat comfort, flight distance, online booking ease, and inflight entertainment.
- **Core Skills / Habilidades Clave:** * Exploratory Data Analysis (EDA) & Data Cleaning.
  * Decision Tree Classification (`scikit-learn`).
  * Hyperparameter Tuning via `GridSearchCV` (evaluating 1,380 combinations).
  * Model evaluation using Confusion Matrices, Accuracy, Precision, Recall, and F1-score.

---

## 📊 Model Performance / Rendimiento del Modelo

Through rigorous cross-validation and hyperparameter tuning ($max\_depth=16$, $min\_samples\_leaf=9$), the optimized model achieved outstanding results:

A través de una rigurosa validación cruzada y ajuste de hiperparámetros ($max\_depth=16$, $min\_samples\_leaf=9$), el modelo optimizado alcanzó resultados sobresalientes:

| Metric / Métrica | Value / Valor | Business Meaning / Significado de Negocio |
| :--- | :---: | :--- |
| **Accuracy** | ~93.85% | Overall correct predictions / Total de predicciones correctas. |
| **Precision** | **95.13%** | Highly reliable "Satisfied" labels; minimal false positives. / Etiquetas de "Satisfecho" altamente confiables. |
| **Recall** | ~93.56% | Effectively captures the majority of truly satisfied passengers. / Captura eficazmente a la mayoría de los pasajeros realmente satisfechos. |
| **F1-Score** | ~94.34% | Robust and balanced harmonic mean. / Media armónica robusta y equilibrada. |

---

## 🎯 Strategic Recommendations / Recomendaciones Estratégicas

### 📈 1. Fleet-wide Inflight Entertainment (IFE) Upgrades
* **EN:** The model identifies `Inflight entertainment` as the single most critical root gatekeeper of passenger sentiment. Improving this touchpoint offers the highest statistical leverage to convert dissatisfied passengers into satisfied ones.
  * *Action:* Allocate capital to upgrade older onboard screen hardware, expand the digital movie/media catalog, and ensure reliable Wi-Fi streaming capabilities across all aircraft.
* **ES:** El modelo identifica a `Inflight entertainment` como el punto de entrada más crítico para el sentimiento del pasajero. Mejorar este único punto de contacto ofrece el mayor impacto estadístico para convertir a pasajeros potencialmente insatisfechos en clientes conformes.
  * *Acción:* Asignar presupuesto para actualizar pantallas obsoletas, expandir el catálogo de películas y contenido digital, y asegurar capacidades de streaming estables mediante Wi-Fi en los aviones.

### 🌗 2. Two-Pronged Service Strategy / Estrategia de Servicio Dual
* **EN:** * *Economy/Value Segment (Fix Comfort):* When entertainment scores are low, **`Seat comfort`** becomes the ultimate make-or-break feature preventing severe passenger dissatisfaction.
  * *Premium/Digital Segment (Optimize Booking):* For passengers already enjoying high-tier entertainment, the final hurdle to securing complete loyalty is **`Ease of Online booking`**.
  * *Action:* Divide efforts between operations (ensuring seat ergonomics) and the digital software product team (simplifying web/mobile interfaces to eliminate friction before boarding).
* **ES:** * *Segmento Económico (Garantizar la Comodidad):* Si el entretenimiento es deficiente, la comodidad del asiento (**`Seat comfort`**) pasa a ser el factor definitivo que evita una caída drástica en la satisfacción.
  * *Segmento Premium (Optimizar la Reserva):* Para los pasajeros que ya disfrutan de un entretenimiento de alto nivel, el último obstáculo para consolidar su lealtad es la facilidad de la reserva en línea (**`Ease of Online booking`**).
  * *Acción:* Dividir esfuerzos entre operaciones (asegurar la ergonomía del asiento) y el equipo de software (simplificar interfaces web/móvil para eliminar fricción antes de abordar).

### 🤖 3. Predictive CRM Deployment / Despliegue de un CRM Predictivo
* **EN:** Our classifier delivers an exceptionally high **Precision of 95.13%**. This means when the model flags a customer segment as "satisfied," the business can trust that prediction with immense confidence.
  * *Action:* Integrate the model into the CRM system. Automatically target predicted high-satisfaction segments for premium loyalty programs, or proactively flag groups at risk of dissatisfaction for automated apology miles, surveys, or targeted service recovery.
* **ES:** Nuestro clasificador optimizado ofrece una **Precisión excepcional del 95.13%**. Esto significa que cuando el modelo etiqueta a un segmento de clientes como "satisfecho", la empresa puede confiar plenamente en esa predicción.
  * *Acción:* Integrar este modelo dentro del sistema de gestión de clientes (CRM). Identificar automáticamente a grupos altamente satisfechos para programas premium, o detectar segmentos en riesgo de insatisfacción para ofrecerles compensaciones proactivas (como millas o beneficios).

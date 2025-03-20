import joblib  
import pandas as pd  
  
# Cargar el modelo y los encoders  
loaded_data = joblib.load('equipment_failure_model.pkl')  
model = loaded_data['model']  
le_equipment = loaded_data['le_equipment']  
  
# Ejemplo de nuevos datos para predecir  
new_data = pd.DataFrame({  
    'temperature': [75.0],  
    'pressure': [10.0],  
    'vibration': [9.5],  
    'humidity': [45.0],  
    'equipment': ['Turbine']  
})  
  
# Preprocesar nuevos datos  
new_data['equipment'] = le_equipment.transform(new_data['equipment'])  
  
# Hacer predicción  
prediction = model.predict(new_data)  
  
print(f'Predicción de fallo: {prediction[0]}')

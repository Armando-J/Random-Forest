import pandas as pd  
from sklearn.model_selection import train_test_split  
from sklearn.ensemble import RandomForestClassifier  
from sklearn.preprocessing import LabelEncoder  
from sklearn.metrics import accuracy_score, classification_report  
import joblib  
  
# Cargar los datos  
df = pd.read_csv('equipos.csv')  
  
#limpiar datos, eliminar columna 'location'  
df.drop(columns=['location'],inplace=True)  
  
# Preprocesamiento  
le_equipment = LabelEncoder()  
  
df['equipment'] = le_equipment.fit_transform(df['equipment'])  
  
# Separar características y objetivo  
x = df.drop('faulty', axis=1)  
y = df['faulty']  
  
# Dividir datos  
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, stratify=y)  
  
# Crear y entrenar el modelo  
model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)  
model.fit(X_train, y_train)  
  
# Evaluación  
y_pred = model.predict(X_test)  
print(f'Accuracy: {accuracy_score(y_test, y_pred):.3f}')  
print(classification_report(y_test, y_pred))  
  
# Guardar el modelo  
joblib.dump({  
    'model': model,  
    'le_equipment': le_equipment }  
    , 'equipment_failure_model.pkl')

import pandas as pd
import numpy as np
import re
from scipy import stats

class VehicleDataImputer:
    def __init__(self, df):
        self.df = df.copy()
        self.original_nulls = df.isnull().sum()
        
    def impute_all(self):
        """Ejecuta toda la estrategia de imputación en orden"""
        print("🚀 Iniciando proceso de imputación...")
        print(f"Valores nulos iniciales: {self.df.isnull().sum().sum()}")
        
        # Orden secuencial según estrategia

        self.impute_manufacturer()
        self.impute_type()
        self.impute_drive()
        self.impute_condition()
        self.impute_cylinders()
        
        self.validate_imputations()
        return self.df
    
    
    def impute_manufacturer(self):
        """Imputa manufacturer extrayendo del model"""
        print("\n5️⃣ Imputando 'manufacturer'...")
        
        mask = self.df['manufacturer'].isnull() & self.df['model'].notna()
        
        # Crear diccionario de manufacturers conocidos
        known_manufacturers = self.df['manufacturer'].dropna().unique()
        
        for idx in self.df[mask].index:
            model = str(self.df.loc[idx, 'model']).lower()
            
            # Buscar manufacturer en el nombre del modelo
            for manufacturer in known_manufacturers:
                if str(manufacturer).lower() in model:
                    self.df.loc[idx, 'manufacturer'] = manufacturer
                    break
            else:
                # Si no se encuentra, usar el más común
                self.df.loc[idx, 'manufacturer'] = self.df['manufacturer'].mode().iloc[0]
        
        print(f"   ✅ Imputados: {mask.sum()} registros")
    
    def impute_type(self):
        """Imputa type basado en manufacturer + model + year"""
        print("\n7️⃣ Imputando 'type'...")
        
        mask = self.df['type'].isnull()
        
        # Palabras clave para identificar tipos
        type_keywords = {
            'sedan': ['sedan', 'camry', 'accord', 'civic', 'corolla'],
            'SUV': ['suv', 'explorer', 'tahoe', 'suburban', 'escalade', 'pilot'],
            'truck': ['truck', 'f-150', 'silverado', 'ram', 'sierra'],
            'coupe': ['coupe', 'mustang', 'camaro', 'challenger'],
            'hatchback': ['hatchback', 'prius', 'fit', 'golf'],
            'wagon': ['wagon', 'forester', 'outback']
        }
        
        for idx in self.df[mask].index:
            model = str(self.df.loc[idx, 'model']).lower()
            manufacturer = self.df.loc[idx, 'manufacturer']
            
            # Buscar por palabras clave en el modelo
            predicted_type = None
            for vehicle_type, keywords in type_keywords.items():
                if any(keyword in model for keyword in keywords):
                    predicted_type = vehicle_type
                    break
            
            if predicted_type:
                self.df.loc[idx, 'type'] = predicted_type
            else:
                # Buscar por manufacturer
                similar_vehicles = self.df[
                    (self.df['manufacturer'] == manufacturer) & 
                    (self.df['type'].notna())
                ]
                
                if len(similar_vehicles) > 0:
                    mode_type = similar_vehicles['type'].mode()
                    self.df.loc[idx, 'type'] = mode_type.iloc[0] if len(mode_type) > 0 else 'sedan'
                else:
                    self.df.loc[idx, 'type'] = 'sedan'  # Más común
        
        print(f"   ✅ Imputados: {mask.sum()} registros")
    
    def impute_drive(self):
        """Imputa drive basado en manufacturer + type + year"""
        print("\n8️⃣ Imputando 'drive'...")
        
        mask = self.df['drive'].isnull()
        
        for idx in self.df[mask].index:
            vehicle_type = self.df.loc[idx, 'type']
            manufacturer = self.df.loc[idx, 'manufacturer']
            
            # Reglas de negocio por tipo
            if pd.notna(vehicle_type):
                if vehicle_type.lower() in ['suv', 'truck']:
                    # SUVs y trucks tienden a tener 4wd
                    self.df.loc[idx, 'drive'] = '4wd'
                elif vehicle_type.lower() in ['sedan', 'hatchback', 'coupe']:
                    # Sedans y similares tienden a ser fwd
                    self.df.loc[idx, 'drive'] = 'fwd'
                else:
                    # Buscar por manufacturer + type
                    similar_vehicles = self.df[
                        (self.df['manufacturer'] == manufacturer) & 
                        (self.df['type'] == vehicle_type) &
                        (self.df['drive'].notna())
                    ]
                    
                    if len(similar_vehicles) > 0:
                        mode_drive = similar_vehicles['drive'].mode()
                        self.df.loc[idx, 'drive'] = mode_drive.iloc[0] if len(mode_drive) > 0 else 'fwd'
                    else:
                        self.df.loc[idx, 'drive'] = 'fwd'
            else:
                self.df.loc[idx, 'drive'] = 'fwd'
        
        print(f"   ✅ Imputados: {mask.sum()} registros")
    
    def impute_condition(self):
        """Imputa condition basado en year + odometer + price - VERSION CORREGIDA"""
        print("\n9️⃣ Imputando 'condition'...")

        # Verificar si la columna es categorical
        is_categorical = self.df['condition'].dtype.name == 'category'

        if is_categorical:
            # Asegurar que todas las categorías necesarias existan
            required_categories = ['excellent', 'good', 'fair', 'poor', 'salvage', 'new', 'like new']
            for cat in required_categories:
                if cat not in self.df['condition'].cat.categories:
                    self.df['condition'] = self.df['condition'].cat.add_categories([cat])

        mask = self.df['condition'].isnull()
        initial_nulls = mask.sum()

        # Vectorizar la imputación para mejor performance
        # Crear una copia de la columna para trabajar
        condition_imputed = self.df['condition'].copy()

        # Obtener los índices donde necesitamos imputar
        null_indices = self.df[mask].index

        for idx in null_indices:
            year = self.df.loc[idx, 'year']
            odometer = self.df.loc[idx, 'odometer']
            price = self.df.loc[idx, 'price']
            
            # Determinar la condición basada en las reglas
            predicted_condition = None
            
            # Primero intentar con year y odometer
            if pd.notna(year) and pd.notna(odometer):
                vehicle_age = 2024 - year
                
                if vehicle_age <= 1 and odometer < 15000:
                    predicted_condition = 'new'
                elif vehicle_age <= 2 and odometer < 30000:
                    predicted_condition = 'like new'
                elif vehicle_age <= 3 and odometer < 45000:
                    predicted_condition = 'excellent'
                elif vehicle_age <= 5 and odometer < 60000:
                    predicted_condition = 'good'
                elif vehicle_age <= 10 and odometer < 120000:
                    predicted_condition = 'fair'
                elif vehicle_age > 15 or odometer > 200000:
                    predicted_condition = 'salvage'
                else:
                    predicted_condition = 'poor'
            
            # Si no se pudo determinar con year/odometer, usar precio
            elif pd.notna(price):
                # Calcular percentil del precio
                price_series = self.df['price'].dropna()
                if len(price_series) > 0:
                    price_percentile = stats.percentileofscore(price_series, price)
                    
                    if price_percentile >= 90:
                        predicted_condition = 'excellent'
                    elif price_percentile >= 70:
                        predicted_condition = 'good'
                    elif price_percentile >= 40:
                        predicted_condition = 'fair'
                    elif price_percentile >= 20:
                        predicted_condition = 'poor'
                    else:
                        predicted_condition = 'salvage'
                else:
                    predicted_condition = 'good'
            
            # Si aún no se pudo determinar, usar valor por defecto
            else:
                # Usar la moda de las condiciones existentes
                mode_condition = self.df['condition'].dropna().mode()
                if len(mode_condition) > 0:
                    predicted_condition = mode_condition.iloc[0]
                else:
                    predicted_condition = 'good'
            
            # Asignar el valor predicho
            condition_imputed.loc[idx] = predicted_condition

        # Actualizar la columna en el DataFrame
        self.df['condition'] = condition_imputed

        print(f"   ✅ Imputados: {initial_nulls} registros")

        # Validación post-imputación
        remaining_nulls = self.df['condition'].isnull().sum()
        if remaining_nulls > 0:
            print(f"   ⚠️  Aún quedan {remaining_nulls} valores nulos")

        # Mostrar distribución de valores imputados
        if initial_nulls > 0:
            print("   📊 Distribución de valores imputados:")
            imputed_values = self.df.loc[null_indices, 'condition'].value_counts()
            for cond, count in imputed_values.items():
                print(f"      - {cond}: {count} ({count/initial_nulls*100:.1f}%)")
    
    def impute_cylinders(self):
        """Imputa cylinders basado en manufacturer + year + type"""
        print("\n🔟 Imputando 'cylinders'...")
        
        mask = self.df['cylinders'].isnull()
        
        # Reglas típicas por tipo de vehículo
        cylinder_rules = {
            'sedan': 4,
            'hatchback': 4,
            'coupe': 6,
            'suv': 6,
            'truck': 8,
            'wagon': 4
        }
        
        for idx in self.df[mask].index:
            manufacturer = self.df.loc[idx, 'manufacturer']
            vehicle_type = self.df.loc[idx, 'type']
            year = self.df.loc[idx, 'year']
            
            # Buscar en vehículos similares
            similar_vehicles = self.df[
                (self.df['manufacturer'] == manufacturer) & 
                (self.df['type'] == vehicle_type) &
                (self.df['cylinders'].notna())
            ]
            
            if len(similar_vehicles) > 0:
                mode_cylinders = similar_vehicles['cylinders'].mode()
                self.df.loc[idx, 'cylinders'] = mode_cylinders.iloc[0] if len(mode_cylinders) > 0 else cylinder_rules.get(vehicle_type, 4)
            else:
                # Usar reglas por tipo
                self.df.loc[idx, 'cylinders'] = cylinder_rules.get(vehicle_type, 4)
        
        print(f"   ✅ Imputados: {mask.sum()} registros")
    
    def validate_imputations(self):
        """Valida la consistencia de las imputaciones"""
        print("\n🔍 Validando imputaciones...")
        
        # Verificar que no hay valores nulos
        final_nulls = self.df.isnull().sum()
        print(f"\nValores nulos finales: {final_nulls.sum()}")
        
        if final_nulls.sum() > 0:
            print("⚠️  Columnas con valores nulos restantes:")
            print(final_nulls[final_nulls > 0])
        
        # Validaciones de consistencia
        print("\n📊 Validaciones de consistencia:")
        
        # 1. Cylinders vs Type
        truck_cylinders = self.df[self.df['type'] == 'truck']['cylinders'].mean()
        sedan_cylinders = self.df[self.df['type'] == 'sedan']['cylinders'].mean()
        print(f"   - Promedio cilindros trucks: {truck_cylinders:.1f}")
        print(f"   - Promedio cilindros sedans: {sedan_cylinders:.1f}")
        
        # 2. Year range
        year_range = self.df['year'].describe()
        print(f"   - Rango de años: {year_range['min']:.0f} - {year_range['max']:.0f}")
        
        # 3. Condition vs Year correlation
        recent_cars = self.df[self.df['year'] >= 2020]['condition'].value_counts()
        print(f"   - Condición de autos recientes (2020+): {recent_cars.head(3).to_dict()}")
        
        print("\n✅ Proceso de imputación completado!")
        
        return {
            'original_nulls': self.original_nulls.to_dict(),
            'final_nulls': final_nulls.to_dict(),
            'total_imputed': self.original_nulls.sum() - final_nulls.sum()
        }

# Función principal para usar el imputador
def impute_vehicle_data(df):
    """
    Función principal para imputar datos de vehículos
    
    Args:
        df: DataFrame con datos de vehículos
    
    Returns:
        DataFrame con datos imputados
    """
    imputer = VehicleDataImputer(df)
    df_imputed = imputer.impute_all()
    
    return df_imputed

# Ejemplo de uso:
# df_imputed = impute_vehicle_data(df)
# print("Imputación completada!")
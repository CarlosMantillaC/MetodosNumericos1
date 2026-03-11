# Importaciones necesarias para el procesamiento numérico y simbólico
import numpy as np  # Biblioteca para cálculos numéricos eficientes
import sympy as sp  # Biblioteca para matemáticas simbólicas
from typing import Callable, Tuple, List  # Tipos para anotaciones de función


class NumericalMethods:
    """
    Clase principal que implementa métodos numéricos para encontrar raíces de ecuaciones.
    
    Esta clase proporciona la funcionalidad base para:
    - Parseo y normalización de funciones matemáticas
    - Conversión entre expresiones simbólicas y funciones numéricas
    - Manejo de errores y validación de entrada
    """
    
    def __init__(self):
        """
        Inicializa la clase con el símbolo x para expresiones simbólicas.
        
        Este símbolo se utiliza para construir y manipular expresiones matemáticas
        con SymPy antes de convertirlas en funciones numéricas evaluables.
        """
        self.symbol = sp.symbols('x')

    def normalize_function_string(self, func_str: str) -> str:
        """
        Normaliza una cadena de función para ser compatible con SymPy.
        
        Esta función realiza las siguientes normalizaciones:
        - Reemplaza operadores matemáticos no estándar (*, ×) por el operador estándar (*)
        - Convierte el operador de potencia (^) a la sintaxis de Python (**)
        - Reemplaza comas decimales (,) por puntos decimales (.)
        - Elimina espacios en blanco al inicio y final
        
        Args:
            func_str (str): Cadena de texto que representa la función matemática
            
        Returns:
            str: Cadena normalizada lista para ser procesada por SymPy
        """
        s = func_str.strip()  # Eliminar espacios en blanco al inicio y final
        s = s.replace("∗", "*")  # Reemplazar asterisco Unicode por asterisco estándar
        s = s.replace("×", "*")  # Reemplazar símbolo de multiplicación por asterisco
        s = s.replace("^", "**")  # Convertir operador de potencia a sintaxis Python
        s = s.replace(",", ".")  # Convertir comas decimales a puntos
        return s

    def parse_function_expression(self, func_str: str) -> Tuple[Callable, sp.Expr]:
        """
        Convierte string de función a función numérica y expresión simbólica.
        
        Este método realiza dos conversiones importantes:
        1. Crea una expresión simbólica con SymPy para manipulación matemática
        2. Genera una función numérica con NumPy para evaluación eficiente
        
        Args:
            func_str (str): Cadena de texto que representa la función matemática
            
        Returns:
            Tuple[Callable, sp.Expr]: 
                - Callable: Función numérica que puede ser evaluada con valores de x
                - sp.Expr: Expresión simbólica para cálculos simbólicos (derivadas, etc.)
                
        Raises:
            ValueError: Si la función no es válida o no puede ser parseada
        """
        try:
            # Normalizar la cadena de entrada para asegurar compatibilidad
            normalized = self.normalize_function_string(func_str)
            
            # Convertir la cadena a una expresión simbólica de SymPy
            expr = sp.sympify(normalized)
            
            # Crear una función numérica a partir de la expresión simbólica
            # Esta función puede ser evaluada eficientemente con arrays de NumPy
            f = sp.lambdify(self.symbol, expr, 'numpy')
            
            return f, expr
        except Exception as e:
            # Si hay error, proporcionar mensaje útil con la versión normalizada
            normalized = self.normalize_function_string(func_str)
            raise ValueError(f"Función no válida: '{func_str}'. Intenta con: '{normalized}'.") from e

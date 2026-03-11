# Importaciones necesarias para el método iterativo
import numpy as np  # Biblioteca para operaciones numéricas y arrays
from typing import List, Tuple  # Tipos para anotaciones de función
from numerical_methods import NumericalMethods  # Clase base para procesamiento de funciones


class IterativeMethods:
    """
    Implementación de métodos iterativos para encontrar raíces de ecuaciones.
    
    Esta clase contiene el método de aproximaciones sucesivas (punto fijo),
    que es un método iterativo simple pero efectivo para encontrar raíces
    cuando la función converge adecuadamente.
    
    El método de punto fijo se basa en reescribir la ecuación f(x) = 0
    como x = g(x) y buscar un punto fijo de g(x).
    """
    
    def __init__(self):
        """
        Inicializa la clase con una instancia de NumericalMethods.
        
        Esta instancia se utiliza para:
        - Parsear y normalizar funciones matemáticas
        - Convertir expresiones simbólicas a funciones numéricas
        - Manejar validación de entrada
        """
        self.numerical_methods = NumericalMethods()
    
    def punto_fijo(
            self,
            func_str: str,           # Función en formato de punto fijo: x = g(x)
            x0: float,               # Valor inicial para la iteración
            tol: float = 1e-6,       # Tolerancia para convergencia
            max_iter: int = 100,     # Máximo número de iteraciones permitidas
            criterio: str = 'error', # Criterio de parada: 'error' o 'iteracion'
            stop_iter: int | None = None,  # Iteración específica para detenerse
            blow_up_limit: float = 1e12,   # Límite para detectar divergencia
            nondecrease_patience: int = 8   # Iteraciones sin mejora antes de declarar divergencia
    ) -> Tuple[List, str]:
        """
        Implementación del método de aproximaciones sucesivas (punto fijo).
        
        El método funciona de la siguiente manera:
        1. Se reescribe la ecuación f(x) = 0 como x = g(x)
        2. Se parte de un valor inicial x0
        3. Se calcula x₁ = g(x₀), x₂ = g(x₁), ..., xₙ₊₁ = g(xₙ)
        4. Se detiene cuando el error es menor que la tolerancia o se alcanza el máximo de iteraciones
        
        Args:
            func_str (str): Función en formato de punto fijo (x = g(x))
            x0 (float): Valor inicial para la iteración
            tol (float): Tolerancia para convergencia (default: 1e-6)
            max_iter (int): Máximo número de iteraciones (default: 100)
            criterio (str): 'error' para detener por tolerancia, 'iteracion' para número fijo
            stop_iter (int | None): Iteración específica para detenerse (opcional)
            blow_up_limit (float): Límite para detectar divergencia (default: 1e12)
            nondecrease_patience (int): Iteraciones sin mejora antes de divergencia (default: 8)
            
        Returns:
            Tuple[List, str]: 
                - List: Lista de diccionarios con datos de cada iteración
                - str: Estado final ('convergio', 'divergio', 'iteraciones', 'max_iter')
                
        Raises:
            ValueError: Si la función no es válida o no puede ser parseada
        """
        # Parsear la función a una forma numérica evaluable
        f, expr = self.numerical_methods.parse_function_expression(func_str)

        # Inicializar variables para el proceso iterativo
        iterations = []              # Lista para almacenar resultados de cada iteración
        current_x = x0              # Valor actual en la iteración
        error = float('inf')        # Error actual (inicialmente infinito)
        status = "max_iter"         # Estado por defecto si se alcanza el máximo
        nondecrease_count = 0       # Contador de iteraciones sin mejora
        previous_error = None       # Error de la iteración anterior para comparación

        # Bucle principal de iteraciones
        for i in range(max_iter):
            # Calcular el siguiente valor usando la función de punto fijo
            next_x = f(current_x)

            # Verificar si el resultado es un número complejo (indica posible divergencia)
            if not np.isfinite(next_x):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'x_siguiente': next_x,
                    'error': float('inf')
                })
                break

            # Verificar si el valor está explotando (demasiado grande)
            if abs(next_x) > blow_up_limit:
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'x_siguiente': next_x,
                    'error': float('inf')
                })
                break

            # Calcular el error real como la diferencia entre iteraciones consecutivas
            actual_error = abs(next_x - current_x)
            
            # Guardar los datos de la iteración actual
            iterations.append({
                'iter': i + 1,
                'x_actual': current_x,
                'x_siguiente': next_x,
                'error': actual_error if criterio == 'error' else i + 1
            })

            # Lógica de convergencia según el criterio seleccionado
            if criterio == 'error':
                # Detener si el error es menor que la tolerancia
                if actual_error < tol:
                    status = "convergio"
                    break

                # Detectar divergencia por falta de mejora en el error
                if previous_error is not None:
                    if actual_error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = actual_error

                # Si el error no disminuye por varias iteraciones, declarar divergencia
                if nondecrease_count >= nondecrease_patience:
                    status = "divergio"
                    break
                    
            elif criterio == 'iteracion':
                # Cuando el criterio es iteración, usamos el error real solo para detectar divergencia
                if previous_error is not None:
                    # Si el error es muy pequeño, no contar como no-disminución (convergencia perfecta)
                    if actual_error <= 1e-15:  # Umbral para convergencia numérica
                        nondecrease_count = 0  # Reiniciar contador
                    elif actual_error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = actual_error

                # Detectar divergencia por falta de mejora
                if nondecrease_count >= nondecrease_patience:
                    status = "divergio"
                    break
                
                # Detener si se alcanza el número máximo de iteraciones
                if (i + 1) >= max_iter:
                    status = "iteraciones"
                    break

            # Actualizar para la siguiente iteración
            current_x = next_x
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        # Generar informe LaTeX con los resultados
        from latex_generator import LaTeXReportGenerator
        latex_gen = LaTeXReportGenerator()
        latex_output = latex_gen.generate_fixed_point_latex(func_str, x0, tol, max_iter, criterio, iterations, status)
        
        return iterations, latex_output

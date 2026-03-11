# Importaciones necesarias para el método de Newton-Raphson
import numpy as np  # Biblioteca para operaciones numéricas y arrays
import sympy as sp  # Biblioteca para cálculo simbólico (derivadas)
from typing import List, Tuple  # Tipos para anotaciones de función
from numerical_methods import NumericalMethods  # Clase base para procesamiento de funciones


class DerivativeMethods:
    """
    Implementación de métodos que usan derivadas para encontrar raíces de ecuaciones.
    
    Esta clase contiene el método de Newton-Raphson, que es uno de los métodos
    más eficientes para encontrar raíces cuando se puede calcular la derivada
    de la función.
    
    El método de Newton-Raphson utiliza la siguiente fórmula de iteración:
    x_{n+1} = x_n - f(x_n) / f'(x_n)
    
    Donde f'(x) es la derivada de la función f(x).
    """
    
    def __init__(self):
        """
        Inicializa la clase con una instancia de NumericalMethods.
        
        Esta instancia se utiliza para:
        - Parsear y normalizar funciones matemáticas
        - Convertir expresiones simbólicas a funciones numéricas
        - Calcular derivadas simbólicamente con SymPy
        """
        self.numerical_methods = NumericalMethods()
    
    def newton_raphson(
            self, 
            func_str: str,              # Función f(x) en formato de cadena
            x0: float,                  # Valor inicial para la iteración
            tol: float = 1e-6,          # Tolerancia para convergencia
            max_iter: int = 100,        # Máximo número de iteraciones permitidas
            criterio: str = 'error',    # Criterio de parada: 'error' o 'iteracion'
            blow_up_limit: float = 1e12, # Límite para detectar divergencia
            nondecrease_patience: int = 8, # Iteraciones sin mejora antes de declarar divergencia
            deriv_eps: float = 1e-14    # Épsilon para detectar derivada cercana a cero
    ) -> Tuple[List, str]:
        """
        Implementación del método de Newton-Raphson para encontrar raíces.
        
        El método funciona de la siguiente manera:
        1. Se parte de un valor inicial x0
        2. Se calcula f(x0) y f'(x0) (la función y su derivada en x0)
        3. Se aplica la fórmula: x1 = x0 - f(x0) / f'(x0)
        4. Se repite hasta alcanzar la tolerancia o el máximo de iteraciones
        
        Ventajas del método:
        - Convergencia cuadrática (muy rápida) cerca de la raíz
        - Requiere menos iteraciones que otros métodos
        
        Desventajas:
        - Necesita calcular la derivada de la función
        - Puede fallar si f'(x) ≈ 0 (división por cero)
        - Sensible al valor inicial
        
        Args:
            func_str (str): Función f(x) en formato de cadena
            x0 (float): Valor inicial para la iteración
            tol (float): Tolerancia para convergencia (default: 1e-6)
            max_iter (int): Máximo número de iteraciones (default: 100)
            criterio (str): 'error' para detener por tolerancia, 'iteracion' para número fijo
            blow_up_limit (float): Límite para detectar divergencia (default: 1e12)
            nondecrease_patience (int): Iteraciones sin mejora antes de divergencia (default: 8)
            deriv_eps (float): Épsilon para detectar derivada cercana a cero (default: 1e-14)
            
        Returns:
            Tuple[List, str]: 
                - List: Lista de diccionarios con datos de cada iteración
                - str: Estado final ('convergio', 'divergio', 'iteraciones', 'max_iter')
                
        Raises:
            ValueError: Si la función no es válida o no puede ser parseada
        """
        # Parsear la función a una forma numérica evaluable
        f, expr = self.numerical_methods.parse_function_expression(func_str)
        
        # Calcular la derivada simbólicamente usando SymPy
        f_prime = sp.diff(expr, self.numerical_methods.symbol)
        
        # Convertir la derivada simbólica a una función numérica evaluable
        df = sp.lambdify(self.numerical_methods.symbol, f_prime, 'numpy')

        # Inicializar variables para el proceso iterativo
        iterations = []              # Lista para almacenar resultados de cada iteración
        current_x = x0              # Valor actual en la iteración
        error = float('inf')        # Error actual (inicialmente infinito)
        status = "max_iter"         # Estado por defecto si se alcanza el máximo
        nondecrease_count = 0       # Contador de iteraciones sin mejora
        previous_error = None       # Error de la iteración anterior para comparación

        # Bucle principal de iteraciones
        for i in range(max_iter):
            # Evaluar la función y su derivada en el punto actual
            fx = f(current_x)
            dfx = df(current_x)

            # Verificar si los resultados son números complejos (indica posible divergencia)
            if np.iscomplexobj(fx) or np.iscomplexobj(dfx):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            # Verificar si los resultados son finitos (no NaN o infinito)
            if not np.isfinite(fx) or not np.isfinite(dfx):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            # Verificar si la derivada es muy cercana a cero (peligro de división por cero)
            if abs(dfx) < deriv_eps:
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': current_x,  # Mantener el valor actual
                    'error': float('inf')
                })
                break

            # Aplicar la fórmula de Newton-Raphson
            # x_{n+1} = x_n - f(x_n) / f'(x_n)
            next_x = current_x - fx / dfx

            # Verificar si el resultado es un número complejo (indica posible divergencia)
            if np.iscomplexobj(next_x):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'fx': fx,
                    'dfx': dfx,
                    'x_siguiente': next_x,
                    'error': float('inf')
                })
                break

            # Verificar si el valor está explotando o es infinito
            if not np.isfinite(next_x) or abs(next_x) > blow_up_limit:
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'fx': fx,
                    'dfx': dfx,
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
                'fx': fx,
                'dfx': dfx,
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
        latex_output = latex_gen.generate_newton_latex(func_str, x0, tol, max_iter, criterio, iterations, status)
        
        return iterations, latex_output

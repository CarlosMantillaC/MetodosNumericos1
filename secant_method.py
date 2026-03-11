# Importaciones necesarias para el método de la secante
import numpy as np  # Biblioteca para operaciones numéricas y arrays
from typing import List, Tuple  # Tipos para anotaciones de función
from numerical_methods import NumericalMethods  # Clase base para procesamiento de funciones


class SecantMethod:
    """
    Implementación del método de la secante para encontrar raíces de ecuaciones.
    
    Esta clase contiene el método de la secante, que es una alternativa al método
    de Newton-Raphson que no requiere calcular la derivada de la función.
    
    El método de la secante utiliza la siguiente fórmula de iteración:
    x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
    
    Este método aproxima la derivada usando una diferencia finita:
    f'(x_n) ≈ (f(x_n) - f(x_{n-1})) / (x_n - x_{n-1})
    
    Ventajas del método:
    - No requiere calcular la derivada de la función
    - Convergencia más rápida que el método de punto fijo
    - Menos sensible al valor inicial que Newton-Raphson
    
    Desventajas:
    - Requiere dos valores iniciales
    - Puede fallar si f(x_n) ≈ f(x_{n-1}) (división por cero)
    - Convergencia más lenta que Newton-Raphson
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
    
    def secante(
            self, 
            func_str: str,              # Función f(x) en formato de cadena
            x0: float,                  # Primer valor inicial
            x1: float,                  # Segundo valor inicial
            tol: float = 1e-6,          # Tolerancia para convergencia
            max_iter: int = 100,        # Máximo número de iteraciones permitidas
            criterio: str = 'error',    # Criterio de parada: 'error' o 'iteracion'
            blow_up_limit: float = 1e12, # Límite para detectar divergencia
            nondecrease_patience: int = 8  # Iteraciones sin mejora antes de declarar divergencia
    ) -> Tuple[List, str]:
        """
        Implementación del método de la secante para encontrar raíces.
        
        El método funciona de la siguiente manera:
        1. Se parten de dos valores iniciales x0 y x1
        2. Se calculan f(x0) y f(x1)
        3. Se aplica la fórmula para obtener x2
        4. Se repite usando los dos valores más recientes
        
        La fórmula se deriva de aproximar la tangente con una secante:
        x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
        
        Args:
            func_str (str): Función f(x) en formato de cadena
            x0 (float): Primer valor inicial
            x1 (float): Segundo valor inicial
            tol (float): Tolerancia para convergencia (default: 1e-6)
            max_iter (int): Máximo número de iteraciones (default: 100)
            criterio (str): 'error' para detener por tolerancia, 'iteracion' para número fijo
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
        previous_x = x0             # Valor anterior en la iteración (x_{n-1})
        current_x = x1              # Valor actual en la iteración (x_n)
        error = float('inf')        # Error actual (inicialmente infinito)
        status = "max_iter"         # Estado por defecto si se alcanza el máximo
        previous_error = None       # Error de la iteración anterior para comparación
        nondecrease_count = 0       # Contador de iteraciones sin mejora

        # Bucle principal de iteraciones
        for i in range(max_iter):
            # Evaluar la función en los dos puntos más recientes
            f_previous = f(previous_x)  # f(x_{n-1})
            f_current = f(current_x)    # f(x_n)

            # Verificar si los resultados son números complejos (indica posible divergencia)
            if np.iscomplexobj(f_previous) or np.iscomplexobj(f_current):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_anterior': previous_x,
                    'x_actual': current_x,
                    'fx_anterior': f_previous,
                    'fx_actual': f_current,
                    'x_siguiente': float('nan'),
                    'error': float('inf')
                })
                break

            # Calcular x_{n+1} usando la fórmula de la secante
            # Manejar el caso especial cuando f(x_n) - f(x_{n-1}) ≈ 0 (división por cero)
            if f_current - f_previous == 0:
                if criterio == 'iteracion':
                    # Cuando el criterio es iteración y hay división por cero,
                    # simplemente continuamos con el valor actual para evitar detener el proceso
                    next_x = current_x
                    actual_error = 0.0  # No hay cambio, error es cero
                else:
                    # Para otros criterios, declaramos divergencia
                    status = "divergio"
                    iterations.append({
                        'iter': i + 1,
                        'x_anterior': previous_x,
                        'x_actual': current_x,
                        'fx_anterior': f_previous,
                        'fx_actual': f_current,
                        'x_siguiente': float('nan'),
                        'error': float('inf')
                    })
                    break
            else:
                # Aplicar la fórmula normal de la secante
                # x_{n+1} = x_n - f(x_n) * (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
                next_x = current_x - f_current * (current_x - previous_x) / (f_current - f_previous)

                # Verificar si el resultado es un número complejo (indica posible divergencia)
                if np.iscomplexobj(next_x):
                    status = "divergio"
                    iterations.append({
                        'iter': i + 1,
                        'x_anterior': previous_x,
                        'x_actual': current_x,
                        'fx_anterior': f_previous,
                        'fx_actual': f_current,
                        'x_siguiente': next_x,
                        'error': float('inf')
                    })
                    break

                # Verificar si el valor está explotando o es infinito
                if not np.isfinite(next_x) or abs(next_x) > blow_up_limit:
                    status = "divergio"
                    iterations.append({
                        'iter': i + 1,
                        'x_anterior': previous_x,
                        'x_actual': current_x,
                        'fx_anterior': f_previous,
                        'fx_actual': f_current,
                        'x_siguiente': next_x,
                        'error': float('inf')
                    })
                    break

                # Calcular el error real como la diferencia entre iteraciones consecutivas
                actual_error = abs(next_x - current_x)
            
            # Determinar el error a reportar según el criterio
            if criterio == 'error':
                error = actual_error
            elif criterio == 'iteracion':
                error = i + 1  # El error es el número de iteración

            # Guardar los datos de la iteración actual
            iterations.append({
                'iter': i + 1,
                'x_anterior': previous_x,
                'x_actual': current_x,
                'fx_anterior': f_previous,
                'fx_actual': f_current,
                'x_siguiente': next_x,
                'error': error
            })

            # Lógica de convergencia según el criterio seleccionado
            if criterio == 'error':
                # Detener si el error es menor que la tolerancia
                if error < tol:
                    status = "convergio"
                    break

                # Detectar divergencia por falta de mejora en el error
                if previous_error is not None:
                    if error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = error

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
            # El valor actual se convierte en el anterior, y el siguiente en el actual
            previous_x = current_x
            current_x = next_x
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        # Generar informe LaTeX con los resultados
        from latex_generator import LaTeXReportGenerator
        latex_gen = LaTeXReportGenerator()
        latex_output = latex_gen.generate_secant_latex(func_str, x0, x1, tol, max_iter, criterio, iterations, status)
        
        return iterations, latex_output

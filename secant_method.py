import numpy as np
from typing import List, Tuple
from numerical_methods import NumericalMethods


class SecantMethod:
    """Implementación del método de la secante."""
    
    def __init__(self):
        self.numerical_methods = NumericalMethods()
    
    def secante(self, func_str: str, x0: float, x1: float, tol: float = 1e-6,
                max_iter: int = 100, criterio: str = 'error', blow_up_limit: float = 1e12,
                nondecrease_patience: int = 8) -> Tuple[List, str]:
        """Método de la secante."""
        f, expr = self.numerical_methods.parse_function_expression(func_str)

        iterations = []
        previous_x = x0
        current_x = x1
        error = float('inf')
        status = "max_iter"
        previous_error = None
        nondecrease_count = 0

        for i in range(max_iter):
            f_previous = f(previous_x)
            f_current = f(current_x)

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

            # Calcular x_siguiente (manejando división por cero)
            if f_current - f_previous == 0:
                if criterio == 'iteracion':
                    # Cuando el criterio es iteracion y hay división por cero,
                    # simplemente continuamos con el valor actual
                    next_x = current_x
                    actual_error = 0.0  # No hay cambio, error es cero
                else:
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
                next_x = current_x - f_current * (current_x - previous_x) / (f_current - f_previous)

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

                # Calcular error real para detección de divergencia
                actual_error = abs(next_x - current_x)
            
            if criterio == 'error':
                error = actual_error
            elif criterio == 'iteracion':
                error = i + 1

            iterations.append({
                'iter': i + 1,
                'x_anterior': previous_x,
                'x_actual': current_x,
                'fx_anterior': f_previous,
                'fx_actual': f_current,
                'x_siguiente': next_x,
                'error': error
            })

            if criterio == 'error':
                if error < tol:
                    status = "convergio"
                    break

                if previous_error is not None:
                    if error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = error

                if nondecrease_count >= nondecrease_patience:
                    status = "divergio"
                    break
            elif criterio == 'iteracion':
                # Cuando el criterio es iteracion, usamos el error real para detectar divergencia
                if previous_error is not None:
                    # Si el error es cero o muy pequeño, no lo contamos como no-disminución (convergencia perfecta)
                    if actual_error <= 1e-15:  # Umbral para convergencia numérica
                        nondecrease_count = 0  # Reiniciar contador, convergencia perfecta
                    elif actual_error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = actual_error

                if nondecrease_count >= nondecrease_patience:
                    status = "divergio"
                    break
                
                if (i + 1) >= max_iter:
                    status = "iteraciones"
                    break

            previous_x = current_x
            current_x = next_x
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        from latex_generator import LaTeXReportGenerator
        latex_gen = LaTeXReportGenerator()
        latex_output = latex_gen.generate_secant_latex(func_str, x0, x1, tol, max_iter, criterio, iterations, status)
        return iterations, latex_output

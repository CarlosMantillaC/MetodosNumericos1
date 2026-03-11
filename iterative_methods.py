import numpy as np
from typing import List, Tuple
from numerical_methods import NumericalMethods


class IterativeMethods:
    """Implementación de métodos iterativos para encontrar raíces."""
    
    def __init__(self):
        self.numerical_methods = NumericalMethods()
    
    def punto_fijo(
            self,
            func_str: str,
            x0: float,
            tol: float = 1e-6,
            max_iter: int = 100,
            criterio: str = 'error',
            stop_iter: int | None = None,
            blow_up_limit: float = 1e12,
            nondecrease_patience: int = 8,
    ) -> Tuple[List, str]:
        """Método de aproximaciones sucesivas (punto fijo)."""
        f, expr = self.numerical_methods.parse_function_expression(func_str)

        iterations = []
        current_x = x0
        error = float('inf')
        status = "max_iter"
        nondecrease_count = 0
        previous_error = None

        for i in range(max_iter):
            next_x = f(current_x)

            if not np.isfinite(next_x):
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'x_siguiente': next_x,
                    'error': float('inf')
                })
                break

            if abs(next_x) > blow_up_limit:
                status = "divergio"
                iterations.append({
                    'iter': i + 1,
                    'x_actual': current_x,
                    'x_siguiente': next_x,
                    'error': float('inf')
                })
                break

            # Calcular error real para detección de divergencia
            actual_error = abs(next_x - current_x)
            
            iterations.append({
                'iter': i + 1,
                'x_actual': current_x,
                'x_siguiente': next_x,
                'error': actual_error if criterio == 'error' else i + 1
            })

            if criterio == 'error':
                if actual_error < tol:
                    status = "convergio"
                    break

                if previous_error is not None:
                    if actual_error >= previous_error:
                        nondecrease_count += 1
                    else:
                        nondecrease_count = 0
                previous_error = actual_error

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

            current_x = next_x
        
        # Si completó todas las iteraciones sin otro estado, marcar como completado
        if status == "max_iter" and criterio == "iteracion":
            status = "iteraciones"

        from latex_generator import LaTeXReportGenerator
        latex_gen = LaTeXReportGenerator()
        latex_output = latex_gen.generate_fixed_point_latex(func_str, x0, tol, max_iter, criterio, iterations, status)
        return iterations, latex_output

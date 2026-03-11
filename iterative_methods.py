# Importaciones necesarias para el método iterativo
import numpy as np  # Biblioteca para operaciones numéricas y arrays
import sympy as sp  # Biblioteca para cálculo simbólico
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
    
    def is_root(self, func_str: str, x: float, tol: float = 1e-10) -> bool:
        """
        Verifica si un valor es una raíz de la función.
        
        Args:
            func_str (str): Función f(x)
            x (float): Valor a verificar
            tol (float): Tolerancia para considerar como raíz
            
        Returns:
            bool: True si x es una raíz, False en caso contrario
        """
        try:
            f, _ = self.numerical_methods.parse_function_expression(func_str)
            fx = f(x)
            return abs(fx) < tol
        except:
            return False
    
    def suggest_initial_values(self, func_str: str, num_suggestions: int = 3) -> List[float]:
        """
        Sugiere valores iniciales que no sean raíces de la función.
        
        Args:
            func_str (str): Función f(x)
            num_suggestions (int): Número de sugerencias a generar
            
        Returns:
            List[float]: Lista de valores iniciales sugeridos
        """
        suggestions = []
        
        # Valores comunes para probar
        test_values = [0.5, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, 0.1, 0.9, 1.1]
        
        for val in test_values:
            if len(suggestions) >= num_suggestions:
                break
                
            if not self.is_root(func_str, val):
                suggestions.append(val)
        
        # Si no encontramos suficientes, generar valores aleatorios
        import random
        while len(suggestions) < num_suggestions:
            val = random.uniform(-2, 5)
            if not self.is_root(func_str, val) and val not in suggestions:
                suggestions.append(round(val, 2))
        
        return suggestions
    
    def convert_to_fixed_point(self, func_str: str) -> str:
        """
        Convierte una función f(x) a la forma g(x) para el método de punto fijo.
        
        Este método implementa varias estrategias para convertir f(x) = 0 a x = g(x):
        1. Para funciones lineales: ax + b = 0 → x = -b/a
        2. Para funciones cuadráticas: ax² + bx + c = 0 → x = (-c - bx)/a
        3. Para funciones con sqrt: sqrt(ax + b) → x = (x² - b)/a
        4. Para funciones generales: x = x - f(x) (forma estándar)
        
        Args:
            func_str (str): Función f(x) en formato de cadena
            
        Returns:
            str: Función g(x) convertida para el método de punto fijo
            
        Raises:
            ValueError: Si la función no puede ser convertida
        """
        try:
            # Parsear la función a expresión simbólica
            f, expr = self.numerical_methods.parse_function_expression(func_str)
            
            # Intentar detectar polinomios cuadráticos directamente
            try:
                poly = sp.Poly(expr, self.numerical_methods.symbol)
                degree = poly.degree()
                
                if degree == 1:
                    # Función lineal: ax + b = 0 → x = -b/a
                    coeffs = poly.all_coeffs()
                    a, b = coeffs[0], coeffs[1]
                    if a != 0:
                        g_expr = -b / a
                        return str(g_expr)
                
                elif degree == 2:
                    # Función cuadrática: usar forma de Newton
                    coeffs = poly.all_coeffs()
                    a, b, c = coeffs[0], coeffs[1], coeffs[2]
                    
                    if a != 0:
                        # x = x - (ax² + bx + c)/(2ax + b)
                        g_expr = self.numerical_methods.symbol - (a*self.numerical_methods.symbol**2 + b*self.numerical_methods.symbol + c)/(2*a*self.numerical_methods.symbol + b)
                        return str(g_expr)
                
                elif degree == 3:
                    # Funciones cúbicas: usar estrategias específicas para mayor estabilidad
                    coeffs = poly.all_coeffs()
                    a, b, c, d = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
                    
                    if a != 0:
                        # Estrategia 1: Forma de Newton (más estable)
                        try:
                            f_derivative = sp.diff(expr, self.numerical_methods.symbol)
                            g_expr_newton = self.numerical_methods.symbol - expr/f_derivative
                            return str(g_expr_newton)
                        except:
                            pass
                        
                        # Estrategia 2: Despejar término cúbico de forma estable
                        # ax³ + bx² + cx + d = 0 → x = -(bx² + cx + d)/(ax²)
                        try:
                            g_expr = -(b*self.numerical_methods.symbol**2 + c*self.numerical_methods.symbol + d)/(a*self.numerical_methods.symbol**2)
                            
                            # Verificar estabilidad
                            f_test, _ = self.numerical_methods.parse_function_expression(str(g_expr))
                            test_values = [0.5, 1.0, 2.0]
                            stable = True
                            for val in test_values:
                                try:
                                    result = f_test(val)
                                    if not (abs(result) < 1e6 and result == result):
                                        stable = False
                                        break
                                except:
                                    stable = False
                                    break
                            
                            if stable:
                                return str(g_expr)
                        except:
                            pass
                        
                        # Estrategia 3: Factorizar si es posible
                        try:
                            roots = sp.nroots(expr)
                            if len(roots) > 0:
                                # Usar la raíz más pequeña en magnitud como referencia
                                root_real = min([abs(complex(r)) for r in roots if abs(complex(r).imag) < 1e-10])
                                if root_real > 0:
                                    # Forma: x = root_real * factor_de_convergencia
                                    g_expr = root_real * 0.9 + 0.1 * self.numerical_methods.symbol
                                    return str(g_expr)
                        except:
                            pass
                
                elif degree >= 4:
                    # Funciones de grado mayor: usar estrategias para asegurar convergencia
                    coeffs = poly.all_coeffs()
                    n = degree
                    
                    # Estrategia 1: Despejar el término de mayor grado
                    # ax^n + ... = 0 → x = (-resto)/(a*x^(n-1))
                    a = coeffs[0]  # Coeficiente del término de mayor grado
                    
                    if a != 0:
                        # Construir el resto de la expresión
                        resto = 0
                        for i in range(1, len(coeffs)):
                            resto += coeffs[i] * self.numerical_methods.symbol**(degree - i)
                        
                        # x = -resto/(a*x^(n-1))
                        if degree - 1 > 0:
                            g_expr = -resto/(a * self.numerical_methods.symbol**(degree - 1))
                        else:
                            g_expr = -resto/a
                        
                        # Verificar si esta forma es estable
                        try:
                            f_test, _ = self.numerical_methods.parse_function_expression(str(g_expr))
                            # Probar con algunos valores para verificar estabilidad
                            test_values = [0.5, 1.0, -0.5, 2.0]
                            stable = True
                            for val in test_values:
                                try:
                                    result = f_test(val)
                                    if not (abs(result) < 1e10 and result == result):  # Verificar finito y no NaN
                                        stable = False
                                        break
                                except:
                                    stable = False
                                    break
                            
                            if stable:
                                return str(g_expr)
                        except:
                            pass
                    
                    # Estrategia 2: Usar forma de Newton general (más estable)
                    # x = x - f(x)/f'(x)
                    try:
                        f_derivative = sp.diff(expr, self.numerical_methods.symbol)
                        g_expr_newton = self.numerical_methods.symbol - expr/f_derivative
                        return str(g_expr_newton)
                    except:
                        pass
                    
                    # Estrategia 3: Dividir por el coeficiente principal y reorganizar
                    # x^n + ... = 0 → x = (-resto)/a)^(1/n)
                    try:
                        if a != 0:
                            resto_dividido = -resto/a
                            if resto_dividido != 0:
                                # Para n impar, podemos usar raíz real
                                if degree % 2 == 1:
                                    g_expr = sp.sign(resto_dividido) * abs(resto_dividido)**(1/degree)
                                    return str(g_expr)
                    except:
                        pass
                        
            except Exception as e:
                pass  # Si no es polinomio, continuar con otras estrategias
            
            # Estrategia 3: Funciones con raíz cuadrada
            if expr.has(sp.sqrt):
                # Si es sqrt(ax + b), asumimos queremos resolver sqrt(ax + b) = x
                # Entonces: ax + b = x² → x = (x² - b)/a
                sqrt_part = list(expr.atoms(sp.sqrt))[0]
                inside = sqrt_part.args[0]
                
                # Extraer coeficientes de ax + b
                a = sp.expand(inside).coeff(self.numerical_methods.symbol, 1)
                b = sp.expand(inside).coeff(self.numerical_methods.symbol, 0)
                
                if a != 0:
                    g_expr = (self.numerical_methods.symbol**2 - b) / a
                    return str(g_expr)
            
            # Estrategia 4: Forma general x = x - f(x)
            g_expr = self.numerical_methods.symbol - expr
            return str(g_expr)
            
        except Exception as e:
            # Si todas las estrategias fallan, usar la forma general
            try:
                f, expr = self.numerical_methods.parse_function_expression(func_str)
                g_expr = self.numerical_methods.symbol - expr
                return str(g_expr)
            except:
                raise ValueError(f"No se pudo convertir '{func_str}' a forma de punto fijo. Error: {e}")
    
    def punto_fijo(
            self,
            func_str: str,           # Función f(x) que se convertirá automáticamente a g(x)
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
        
        Este método ahora acepta una función f(x) y la convierte automáticamente
        a la forma g(x) necesaria para el método de punto fijo.
        
        El proceso es el siguiente:
        1. Se recibe f(x) (ej: x**2 - 5*x + 4)
        2. Se convierte automáticamente a g(x) usando estrategias específicas
        3. Se verifica que x0 no sea ya una raíz
        4. Se verifica que x0 no cause problemas de división por cero
        5. Se aplica el método de punto fijo con g(x)
        
        Estrategias de conversión:
        - Lineal: ax + b = 0 → x = -b/a
        - Cuadrático: ax² + bx + c = 0 → x = (-c - bx)/a  
        - Raíz: sqrt(ax + b) → x = (x² - b)/a
        - General: x = x - f(x)
        
        Args:
            func_str (str): Función f(x) que se convertirá automáticamente a g(x)
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
            ValueError: Si la función no es válida o no puede ser convertida
        """
        # Verificar si x0 ya es una raíz
        if self.is_root(func_str, x0):
            suggestions = self.suggest_initial_values(func_str)
            raise ValueError(
                f"⚠ El valor inicial x0 = {x0} ya es una raíz de la función.\n"
                f"Esto no demuestra el funcionamiento del método de punto fijo.\n"
                f"Prueba con estos valores iniciales: {suggestions[:3]}\n"
                f"Ejemplo: x0 = {suggestions[0] if suggestions else 0.5}"
            )
        
        # Convertir automáticamente f(x) a g(x)
        try:
            g_func_str = self.convert_to_fixed_point(func_str)
        except ValueError as e:
            raise ValueError(f"Error al convertir la función: {e}")
        
        # Verificar que x0 no cause problemas con la función convertida
        try:
            f_test, _ = self.numerical_methods.parse_function_expression(g_func_str)
            # Probar ejecutar una iteración para detectar problemas
            try:
                test_result = f_test(x0)
                if not (abs(test_result) < 1e10 and test_result == test_result):
                    raise ValueError("Resultado numérico inválido")
            except (ZeroDivisionError, ValueError, OverflowError):
                # Si x0 causa problemas, sugerir valores alternativos
                suggestions = self.suggest_initial_values(func_str)
                good_suggestions = []
                for suggestion in suggestions:
                    try:
                        test_result = f_test(suggestion)
                        if abs(test_result) < 1e10 and test_result == test_result:
                            good_suggestions.append(suggestion)
                    except:
                        continue
                
                raise ValueError(
                    f"⚠ El valor inicial x0 = {x0} causa problemas numéricos (división por cero o desbordamiento).\n"
                    f"Prueba con estos valores iniciales: {good_suggestions[:3]}\n"
                    f"Ejemplo: x0 = {good_suggestions[0] if good_suggestions else 0.5}"
                )
        except ValueError:
            raise  # Re-lanzar ValueError personalizado
        except:
            pass  # Si no se puede verificar, continuar con el método original
        
        # Parsear la función convertida a una forma numérica evaluable
        f, expr = self.numerical_methods.parse_function_expression(g_func_str)

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
        latex_output = latex_gen.generate_fixed_point_latex_with_conversion(
            func_str, g_func_str, x0, tol, max_iter, criterio, iterations, status
        )
        
        return iterations, latex_output
    
    def punto_fijo_manual(
            self,
            g_func_str: str,          # Función g(x) directamente (ya convertida)
            x0: float,               # Valor inicial para la iteración
            tol: float = 1e-6,       # Tolerancia para convergencia
            max_iter: int = 100,     # Máximo número de iteraciones permitidas
            criterio: str = 'error', # Criterio de parada: 'error' o 'iteracion'
            blow_up_limit: float = 1e12,   # Límite para detectar divergencia
            nondecrease_patience: int = 8   # Iteraciones sin mejora antes de declarar divergencia
    ) -> Tuple[List, str]:
        """
        Implementación del método de punto fijo con función g(x) directa.
        
        Este método acepta directamente la función g(x) sin necesidad de conversión.
        El usuario debe proporcionar la función en la forma x = g(x).
        
        Args:
            g_func_str (str): Función g(x) directa para el método de punto fijo
            x0 (float): Valor inicial para la iteración
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
        # Parsear directamente la función g(x)
        f, expr = self.numerical_methods.parse_function_expression(g_func_str)

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

        # Generar informe LaTeX con los resultados (modo manual)
        from latex_generator import LaTeXReportGenerator
        latex_gen = LaTeXReportGenerator()
        latex_output = latex_gen.generate_fixed_point_latex_manual(
            g_func_str, x0, tol, max_iter, criterio, iterations, status
        )
        
        return iterations, latex_output

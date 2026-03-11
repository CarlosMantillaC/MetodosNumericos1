# Importaciones necesarias para la generación de informes LaTeX
import sympy as sp  # Biblioteca para cálculo simbólico y conversión a LaTeX
import numpy as np  # Biblioteca para operaciones numéricas
from typing import List  # Tipos para anotaciones de función
from numerical_methods import NumericalMethods  # Clase base para procesamiento de funciones


class LaTeXReportGenerator:
    """
    Generador de informes LaTeX para los métodos numéricos.
    
    Esta clase se encarga de crear documentos LaTeX formateados que contienen:
    - Información sobre el método numérico utilizado
    - Parámetros de entrada (función, valores iniciales, tolerancia, etc.)
    - Tablas detalladas con el proceso iterativo
    - Resultados finales y estado de convergencia
    
    Los informes generados pueden ser compilados a PDF para su visualización
    o impresión, proporcionando un registro profesional de los cálculos.
    
    Estructura del documento LaTeX:
    1. Encabezado con información del método
    2. Parámetros de configuración
    3. Tabla con iteraciones (usando longtable para tablas largas)
    4. Resumen de resultados
    """
    
    def __init__(self):
        """
        Inicializa la clase con una instancia de NumericalMethods.
        
        Esta instancia se utiliza para:
        - Normalizar funciones matemáticas para LaTeX
        - Convertir expresiones a formato LaTeX
        - Manejar validación de entrada
        """
        self.numerical_methods = NumericalMethods()
    
    def format_number_for_latex(self, value, format_type: str) -> str:
        """
        Formatea números para salida LaTeX con manejo especial de casos complejos.
        
        Este método maneja varios tipos de números:
        - Números reales estándar
        - Números complejos (con formato i)
        - Valores infinitos (∞)
        - Valores NaN o no válidos
        - Diferentes formatos (decimal, científico)
        
        Args:
            value: Valor numérico a formatear
            format_type (str): Tipo de formato ('float' para decimal, 'sci' para científico)
            
        Returns:
            str: Número formateado para LaTeX
        """
        try:
            # Caso 1: Valor nulo o no especificado
            if value is None:
                return "-"
            
            # Caso 2: Números complejos representados como cadena
            if isinstance(value, str) and ("j" in value or "J" in value):
                s = value.strip().strip("()")
                s = s.replace("J", "j")
                s = s.replace("j", "")
                return s + "\\,i"  # Formato LaTeX para números imaginarios
            
            # Caso 3: Números complejos nativos de Python/NumPy
            if isinstance(value, (complex, np.complexfloating)) or np.iscomplexobj(value):
                c = complex(value)
                sign = "+" if c.imag >= 0 else "-"
                if format_type == "float":
                    return f"{c.real:.6f} {sign} {abs(c.imag):.2e}\\,i"
                if format_type == "sci":
                    return f"{c.real:.2e} {sign} {abs(c.imag):.2e}\\,i"
                return f"{c.real} {sign} {abs(c.imag)} i"
            
            # Caso 4: Valores infinitos
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return "\\infty" if value > 0 else "-\\infty"
            
            # Caso 5: Formato decimal estándar
            if format_type == "float":
                return f"{float(value):.6f}"
            
            # Caso 6: Formato científico
            if format_type == "sci":
                return f"{float(value):.2e}"
            
            # Caso 7: Formato por defecto
            return str(value)
        except Exception:
            # Si hay algún error, devolver el valor como cadena
            return str(value)
    
    def generate_fixed_point_latex(self, func_str: str, x0: float, tol: float, max_iter: int,
                                     criterio: str, iterations: List, status: str) -> str:
        """
        Genera informe LaTeX para el método de punto fijo.
        
        El informe incluye:
        - Título y descripción del método
        - Función matemática en formato LaTeX
        - Parámetros de configuración
        - Tabla con iteraciones (x_i, x_{i+1}, error)
        - Resumen de resultados
        
        Args:
            func_str (str): Función en formato de punto fijo
            x0 (float): Valor inicial
            tol (float): Tolerancia
            max_iter (int): Máximo de iteraciones
            criterio (str): Criterio de parada
            iterations (List): Lista de iteraciones con datos
            status (str): Estado final del proceso
            
        Returns:
            str: Documento LaTeX completo
        """
        # Convertir la función a formato LaTeX para visualización matemática
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
        
        # Iniciar el documento LaTeX con encabezado y configuración
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de Aproximaciones Sucesivas (Punto Fijo)}}

\\textbf{{Función:}} $g(x) = {func_latex}$

\\textbf{{Valor inicial:}} $x_0 = {x0}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        # Agregar cada iteración a la tabla
        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_actual'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        # Agregar resumen de resultados si hay iteraciones
        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        # Cerrar el documento LaTeX
        latex += "\n\\end{document}"
        return latex
    
    def generate_newton_latex(self, func_str: str, initial_value: float, tol: float, max_iter: int,
                             stopping_criterion: str, iterations: List, status: str) -> str:
        """
        Genera informe LaTeX para el método de Newton-Raphson.
        
        El informe incluye:
        - Título y descripción del método
        - Función matemática y su derivada en formato LaTeX
        - Parámetros de configuración
        - Tabla con iteraciones (x_i, f(x_i), f'(x_i), x_{i+1}, error)
        - Resumen de resultados
        
        Args:
            func_str (str): Función f(x)
            initial_value (float): Valor inicial x0
            tol (float): Tolerancia
            max_iter (int): Máximo de iteraciones
            stopping_criterion (str): Criterio de parada
            iterations (List): Lista de iteraciones con datos
            status (str): Estado final del proceso
            
        Returns:
            str: Documento LaTeX completo
        """
        # Convertir la función a formato LaTeX
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
        
        # Iniciar el documento LaTeX
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de Newton-Raphson}}

\\textbf{{Función:}} $f(x) = {func_latex}$

\\textbf{{Valor inicial:}} $x_0 = {initial_value}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {stopping_criterion}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$f'(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$f'(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        # Agregar cada iteración a la tabla
        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_actual'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['fx'], 'float')}$ & ${self.format_number_for_latex(iteration['dfx'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        # Agregar resumen de resultados
        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        # Cerrar el documento LaTeX
        latex += "\n\\end{document}"
        return latex
    
    def generate_secant_latex(self, func_str: str, initial_value1: float, initial_value2: float, tol: float, max_iter: int,
                              stopping_criterion: str, iterations: List, status: str) -> str:
        """
        Genera informe LaTeX para el método de la secante.
        
        El informe incluye:
        - Título y descripción del método
        - Función matemática en formato LaTeX
        - Parámetros de configuración (dos valores iniciales)
        - Tabla con iteraciones (x_{i-1}, x_i, f(x_{i-1}), f(x_i), x_{i+1}, error)
        - Resumen de resultados
        
        Args:
            func_str (str): Función f(x)
            initial_value1 (float): Primer valor inicial x0
            initial_value2 (float): Segundo valor inicial x1
            tol (float): Tolerancia
            max_iter (int): Máximo de iteraciones
            stopping_criterion (str): Criterio de parada
            iterations (List): Lista de iteraciones con datos
            status (str): Estado final del proceso
            
        Returns:
            str: Documento LaTeX completo
        """
        # Convertir la función a formato LaTeX
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
        
        # Iniciar el documento LaTeX
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\begin{{document}}

\\section*{{Método de la Secante}}

\\textbf{{Función:}} $f(x) = {func_latex}$

\\textbf{{Valores iniciales:}} $x_0 = {initial_value1}$, $x_1 = {initial_value2}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {stopping_criterion}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\textbf{{Estado:}} {status}

\\subsection*{{Proceso iterativo}}

\\begin{{longtable}}{{|c|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i-1}}$}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i-1}})$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endfirsthead
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i-1}}$}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i-1}})$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
\\endhead
\\hline
\\endfoot
\\hline
\\endlastfoot
"""

        # Agregar cada iteración a la tabla
        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_anterior'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['x_actual'], 'float')}$ "
                f"& ${self.format_number_for_latex(iteration['fx_anterior'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['fx_actual'], 'float')}$ "
                f"& ${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & "
                f"${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        # Agregar resumen de resultados
        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        # Cerrar el documento LaTeX
        latex += "\n\\end{document}"
        return latex

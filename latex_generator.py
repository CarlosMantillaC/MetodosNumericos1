import sympy as sp
import numpy as np
from typing import List
from numerical_methods import NumericalMethods


class LaTeXReportGenerator:
    """Generador de informes LaTeX para los métodos numéricos."""
    
    def __init__(self):
        self.numerical_methods = NumericalMethods()
    
    def format_number_for_latex(self, value, format_type: str) -> str:
        """Formatea números para salida LaTeX."""
        try:
            if value is None:
                return "-"
            if isinstance(value, str) and ("j" in value or "J" in value):
                s = value.strip().strip("()")
                s = s.replace("J", "j")
                s = s.replace("j", "")
                return s + "\\,i"
            if isinstance(value, (complex, np.complexfloating)) or np.iscomplexobj(value):
                c = complex(value)
                sign = "+" if c.imag >= 0 else "-"
                if format_type == "float":
                    return f"{c.real:.6f} {sign} {abs(c.imag):.2e}\\,i"
                if format_type == "sci":
                    return f"{c.real:.2e} {sign} {abs(c.imag):.2e}\\,i"
                return f"{c.real} {sign} {abs(c.imag)} i"
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                return "\\infty" if value > 0 else "-\\infty"
            if format_type == "float":
                return f"{float(value):.6f}"
            if format_type == "sci":
                return f"{float(value):.2e}"
            return str(value)
        except Exception:
            return str(value)
    
    def generate_fixed_point_latex(self, func_str: str, x0: float, tol: float, max_iter: int,
                                     criterio: str, iterations: List, status: str) -> str:
        """Genera informe LaTeX para el método de punto fijo."""
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
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

        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_actual'], 'float')}$ & ${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & ${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex
    
    def generate_newton_latex(self, func_str: str, initial_value: float, tol: float, max_iter: int,
                             stopping_criterion: str, iterations: List, status: str) -> str:
        """Genera informe LaTeX para el método de Newton-Raphson."""
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
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

        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_actual'], 'float')}$ & ${self.format_number_for_latex(iteration['fx'], 'float')}$ & ${self.format_number_for_latex(iteration['dfx'], 'float')}$ & ${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & ${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Último iterado:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex
    
    def generate_secant_latex(self, func_str: str, initial_value1: float, initial_value2: float, tol: float, max_iter: int,
                              stopping_criterion: str, iterations: List, status: str) -> str:
        """Genera informe LaTeX para el método de la secante."""
        func_latex = sp.latex(sp.sympify(self.numerical_methods.normalize_function_string(func_str)))
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

        for iteration in iterations:
            latex += (
                f"{iteration['iter']} & ${self.format_number_for_latex(iteration['x_anterior'], 'float')}$ & ${self.format_number_for_latex(iteration['x_actual'], 'float')}$ "
                f"& ${self.format_number_for_latex(iteration['fx_anterior'], 'float')}$ & ${self.format_number_for_latex(iteration['fx_actual'], 'float')}$ "
                f"& ${self.format_number_for_latex(iteration['x_siguiente'], 'float')}$ & ${self.format_number_for_latex(iteration['error'], 'float')}$ \\\\ \\hline\n"
            )

        if iterations:
            latex += f"\\end{{longtable}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${self.format_number_for_latex(iterations[-1]['x_siguiente'], 'float')}$\n"
            latex += f"\\textbf{{Error final:}} ${self.format_number_for_latex(iterations[-1]['error'], 'float')}$\n"

        latex += "\n\\end{document}"
        return latex

import numpy as np
import sympy as sp
from typing import Callable, Tuple, List
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import shutil
import subprocess

class MetodosNumericos:
    def __init__(self):
        self.x = sp.symbols('x')
    
    def parse_function(self, func_str: str) -> Callable:
        """Convierte string de función a función numérica"""
        try:
            expr = sp.sympify(func_str)
            f = sp.lambdify(self.x, expr, 'numpy')
            return f, expr
        except:
            raise ValueError("Función no válida")
    
    def punto_fijo(self, func_str: str, x0: float, tol: float = 1e-6, max_iter: int = 100, 
                   criterio: str = 'error') -> Tuple[List, str]:
        """Método de aproximaciones sucesivas (punto fijo)"""
        f, expr = self.parse_function(func_str)
        
        iteraciones = []
        x_actual = x0
        error = float('inf')
        
        for i in range(max_iter):
            x_siguiente = f(x_actual)
            
            if criterio == 'error':
                error = abs(x_siguiente - x_actual)
            elif criterio == 'iteracion':
                error = i + 1
            
            iteraciones.append({
                'iter': i + 1,
                'x_actual': x_actual,
                'x_siguiente': x_siguiente,
                'error': error
            })
            
            if criterio == 'error' and error < tol:
                break
            
            x_actual = x_siguiente
        
        latex_output = self._latex_punto_fijo(func_str, x0, tol, max_iter, criterio, iteraciones)
        return iteraciones, latex_output
    
    def newton_raphson(self, func_str: str, x0: float, tol: float = 1e-6, max_iter: int = 100,
                      criterio: str = 'error') -> Tuple[List, str]:
        """Método de Newton-Raphson"""
        f, expr = self.parse_function(func_str)
        f_prime = sp.diff(expr, self.x)
        df = sp.lambdify(self.x, f_prime, 'numpy')
        
        iteraciones = []
        x_actual = x0
        error = float('inf')
        
        for i in range(max_iter):
            fx = f(x_actual)
            dfx = df(x_actual)
            
            if dfx == 0:
                break
            
            x_siguiente = x_actual - fx/dfx
            
            if criterio == 'error':
                error = abs(x_siguiente - x_actual)
            elif criterio == 'iteracion':
                error = i + 1
            
            iteraciones.append({
                'iter': i + 1,
                'x_actual': x_actual,
                'fx': fx,
                'dfx': dfx,
                'x_siguiente': x_siguiente,
                'error': error
            })
            
            if criterio == 'error' and error < tol:
                break
            
            x_actual = x_siguiente
        
        latex_output = self._latex_newton(func_str, x0, tol, max_iter, criterio, iteraciones)
        return iteraciones, latex_output
    
    def secante(self, func_str: str, x0: float, x1: float, tol: float = 1e-6, 
                max_iter: int = 100, criterio: str = 'error') -> Tuple[List, str]:
        """Método de la secante"""
        f, expr = self.parse_function(func_str)
        
        iteraciones = []
        x_anterior = x0
        x_actual = x1
        error = float('inf')
        
        for i in range(max_iter):
            fx_anterior = f(x_anterior)
            fx_actual = f(x_actual)
            
            if fx_actual - fx_anterior == 0:
                break
            
            x_siguiente = x_actual - fx_actual * (x_actual - x_anterior) / (fx_actual - fx_anterior)
            
            if criterio == 'error':
                error = abs(x_siguiente - x_actual)
            elif criterio == 'iteracion':
                error = i + 1
            
            iteraciones.append({
                'iter': i + 1,
                'x_anterior': x_anterior,
                'x_actual': x_actual,
                'fx_anterior': fx_anterior,
                'fx_actual': fx_actual,
                'x_siguiente': x_siguiente,
                'error': error
            })
            
            if criterio == 'error' and error < tol:
                break
            
            x_anterior = x_actual
            x_actual = x_siguiente
        
        latex_output = self._latex_secante(func_str, x0, x1, tol, max_iter, criterio, iteraciones)
        return iteraciones, latex_output
    
    def _latex_punto_fijo(self, func_str: str, x0: float, tol: float, max_iter: int, 
                         criterio: str, iteraciones: List) -> str:
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\begin{{document}}

\\section*{{Método de Aproximaciones Sucesivas (Punto Fijo)}}

\\textbf{{Función:}} $g(x) = {sp.sympify(func_str)}$

\\textbf{{Valor inicial:}} $x_0 = {x0}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\subsection*{{Proceso iterativo}}

\\begin{{tabular}}{{|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
"""
        
        for it in iteraciones:
            latex += f"{it['iter']} & {it['x_actual']:.6f} & {it['x_siguiente']:.6f} & {it['error']:.2e} \\\\ \\hline\n"
        
        if iteraciones:
            latex += f"\\end{{tabular}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${iteraciones[-1]['x_siguiente']:.6f}$\n"
            latex += f"\\textbf{{Error final:}} ${iteraciones[-1]['error']:.2e}$\n"
        
        latex += "\n\\end{document}"
        return latex


class MetodosNumericosGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Métodos Numéricos")

        self.mn = MetodosNumericos()

        self.method_var = tk.StringVar(value="punto_fijo")
        self.func_var = tk.StringVar(value="0.5*(x + 2/x)")
        self.x0_var = tk.StringVar(value="1")
        self.x1_var = tk.StringVar(value="2")
        self.tol_var = tk.StringVar(value="1e-6")
        self.max_iter_var = tk.StringVar(value="100")
        self.criterio_var = tk.StringVar(value="error")

        self._build_ui()
        self._on_method_change()

    def _build_ui(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        frm = ttk.Frame(self.root, padding=12)
        frm.grid(row=0, column=0, sticky="nsew")
        frm.columnconfigure(1, weight=1)

        ttk.Label(frm, text="Método:").grid(row=0, column=0, sticky="w")
        method_cb = ttk.Combobox(
            frm,
            textvariable=self.method_var,
            values=["punto_fijo", "newton_raphson", "secante"],
            state="readonly",
        )
        method_cb.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        method_cb.bind("<<ComboboxSelected>>", lambda e: self._on_method_change())

        ttk.Label(frm, text="Función:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.func_var).grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="x0:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.x0_var).grid(row=2, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="x1 (secante):").grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.x1_entry = ttk.Entry(frm, textvariable=self.x1_var)
        self.x1_entry.grid(row=3, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Tolerancia (ε):").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.tol_var).grid(row=4, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Máx. iteraciones:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.max_iter_var).grid(row=5, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        ttk.Label(frm, text="Criterio de parada:").grid(row=6, column=0, sticky="w", pady=(8, 0))
        criterio_cb = ttk.Combobox(frm, textvariable=self.criterio_var, values=["error", "iteracion"], state="readonly")
        criterio_cb.grid(row=6, column=1, sticky="ew", padx=(8, 0), pady=(8, 0))

        btns = ttk.Frame(frm)
        btns.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        btns.columnconfigure(3, weight=1)

        ttk.Button(btns, text="Ejecutar", command=self._run).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Guardar .tex", command=self._save_tex).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(btns, text="Generar PDF", command=self._save_pdf).grid(row=0, column=2, sticky="ew")
        ttk.Button(btns, text="Limpiar", command=self._clear).grid(row=0, column=3, sticky="ew", padx=(8, 0))

        out = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        out.grid(row=1, column=0, sticky="nsew")
        out.rowconfigure(0, weight=1)
        out.columnconfigure(0, weight=1)

        self.output = tk.Text(out, wrap="none")
        self.output.grid(row=0, column=0, sticky="nsew")

        yscroll = ttk.Scrollbar(out, orient="vertical", command=self.output.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        self.output.configure(yscrollcommand=yscroll.set)

        xscroll = ttk.Scrollbar(out, orient="horizontal", command=self.output.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        self.output.configure(xscrollcommand=xscroll.set)

        self._last_latex = ""

    def _on_method_change(self):
        method = self.method_var.get()
        if method == "secante":
            self.x1_entry.state(["!disabled"])
        else:
            self.x1_entry.state(["disabled"])

        if method == "punto_fijo":
            if self.func_var.get().strip() in ("", "x**2 - 2", "x**2-2"):
                self.func_var.set("0.5*(x + 2/x)")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
        else:
            if self.func_var.get().strip() in ("", "0.5*(x + 2/x)"):
                self.func_var.set("x**2 - 2")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
            if method == "secante" and self.x1_var.get().strip() == "":
                self.x1_var.set("2")

    def _clear(self):
        self.output.delete("1.0", "end")
        self._last_latex = ""

    def _run(self):
        func_str = self.func_var.get().strip()
        if not func_str:
            messagebox.showerror("Error", "La función no puede estar vacía")
            return

        try:
            x0 = float(self.x0_var.get())
            tol = float(self.tol_var.get())
            max_iter = int(float(self.max_iter_var.get()))
            criterio = self.criterio_var.get().strip()

            if criterio not in ("error", "iteracion"):
                raise ValueError("Criterio debe ser 'error' o 'iteracion'")

            method = self.method_var.get()
            if method == "punto_fijo":
                iteraciones, latex = self.mn.punto_fijo(func_str, x0, tol, max_iter, criterio)
            elif method == "newton_raphson":
                iteraciones, latex = self.mn.newton_raphson(func_str, x0, tol, max_iter, criterio)
            elif method == "secante":
                x1 = float(self.x1_var.get())
                iteraciones, latex = self.mn.secante(func_str, x0, x1, tol, max_iter, criterio)
            else:
                raise ValueError("Método no soportado")

            self._last_latex = latex
            self.output.delete("1.0", "end")
            self.output.insert("1.0", latex)

            if not iteraciones:
                messagebox.showwarning("Atención", "No se generaron iteraciones (posible división por 0 o problema numérico).")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _save_tex(self):
        if not self._last_latex:
            messagebox.showerror("Error", "No hay salida LaTeX para guardar. Primero ejecuta un método.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX", "*.tex"), ("Todos", "*")],
            initialfile="resultado.tex",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(self._last_latex)
            messagebox.showinfo("OK", f"Archivo guardado en: {path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _open_file(self, path: str):
        try:
            if os.name == "nt":
                os.startfile(path)  # type: ignore[attr-defined]
            else:
                opener = "xdg-open" if shutil.which("xdg-open") else None
                if opener is None:
                    return
                subprocess.run([opener, path], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            return

    def _save_pdf(self):
        if not self._last_latex:
            messagebox.showerror("Error", "No hay salida LaTeX. Primero ejecuta un método.")
            return

        if shutil.which("pdflatex") is None:
            messagebox.showerror(
                "Error",
                "No se encontró 'pdflatex'. Instala TeX Live/MiKTeX para generar PDF.",
            )
            return

        tex_path = filedialog.asksaveasfilename(
            defaultextension=".tex",
            filetypes=[("LaTeX", "*.tex"), ("Todos", "*")],
            initialfile="resultado.tex",
        )
        if not tex_path:
            return

        try:
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(self._last_latex)

            out_dir = os.path.dirname(os.path.abspath(tex_path))
            tex_filename = os.path.basename(tex_path)

            proc = subprocess.run(
                ["pdflatex", "-interaction=nonstopmode", tex_filename],
                cwd=out_dir,
                capture_output=True,
                text=True,
                check=False,
            )

            pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
            if proc.returncode != 0 or not os.path.exists(pdf_path):
                msg = proc.stdout[-2000:] + "\n" + proc.stderr[-2000:]
                messagebox.showerror("Error", "Falló la compilación LaTeX.\n\n" + msg)
                return

            messagebox.showinfo("OK", f"PDF generado en: {pdf_path}")
            self._open_file(pdf_path)
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def _latex_newton(self, func_str: str, x0: float, tol: float, max_iter: int,
                     criterio: str, iteraciones: List) -> str:
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\begin{{document}}

\\section*{{Método de Newton-Raphson}}

\\textbf{{Función:}} $f(x) = {sp.sympify(func_str)}$

\\textbf{{Valor inicial:}} $x_0 = {x0}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\subsection*{{Proceso iterativo}}

\\begin{{tabular}}{{|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$f'(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
"""
        
        for it in iteraciones:
            latex += f"{it['iter']} & {it['x_actual']:.6f} & {it['fx']:.6f} & {it['dfx']:.6f} & {it['x_siguiente']:.6f} & {it['error']:.2e} \\\\ \\hline\n"
        
        if iteraciones:
            latex += f"\\end{{tabular}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${iteraciones[-1]['x_siguiente']:.6f}$\n"
            latex += f"\\textbf{{Error final:}} ${iteraciones[-1]['error']:.2e}$\n"
        
        latex += "\n\\end{document}"
        return latex
    
    def _latex_secante(self, func_str: str, x0: float, x1: float, tol: float, max_iter: int,
                      criterio: str, iteraciones: List) -> str:
        latex = f"""
\\documentclass{{article}}
\\usepackage{{amsmath}}
\\usepackage{{array}}
\\usepackage{{booktabs}}
\\begin{{document}}

\\section*{{Método de la Secante}}

\\textbf{{Función:}} $f(x) = {sp.sympify(func_str)}$

\\textbf{{Valores iniciales:}} $x_0 = {x0}$, $x_1 = {x1}$

\\textbf{{Tolerancia:}} $\\varepsilon = {tol}$

\\textbf{{Criterio de parada:}} {criterio}

\\textbf{{Máximo de iteraciones:}} {max_iter}

\\subsection*{{Proceso iterativo}}

\\begin{{tabular}}{{|c|c|c|c|c|c|c|}}
\\hline
\\textbf{{Iteración}} & \\textbf{{$x_{{i-1}}$}} & \\textbf{{$x_{{i}}$}} & \\textbf{{$f(x_{{i-1}})$}} & \\textbf{{$f(x_{{i}})$}} & \\textbf{{$x_{{i+1}}$}} & \\textbf{{Error}} \\\\ \\hline
"""
        
        for it in iteraciones:
            latex += f"{it['iter']} & {it['x_anterior']:.6f} & {it['x_actual']:.6f} & {it['fx_anterior']:.6f} & {it['fx_actual']:.6f} & {it['x_siguiente']:.6f} & {it['error']:.2e} \\\\ \\hline\n"
        
        if iteraciones:
            latex += f"\\end{{tabular}}\n\n"
            latex += f"\\textbf{{Raíz aproximada:}} ${iteraciones[-1]['x_siguiente']:.6f}$\n"
            latex += f"\\textbf{{Error final:}} ${iteraciones[-1]['error']:.2e}$\n"
        
        latex += "\n\\end{document}"
        return latex

def menu():
    mn = MetodosNumericos()
    
    print("=== MÉTODOS NUMÉRICOS ===")
    print("1. Método de Aproximaciones Sucesivas (Punto Fijo)")
    print("2. Método de Newton-Raphson")
    print("3. Método de la Secante")
    print("4. Salir")
    
    opcion = input("Seleccione un método (1-4): ")
    
    if opcion == '4':
        return
    
    func_str = input("Ingrese la función (ej: x**2 - 2): ")
    
    if opcion == '1':
        x0 = float(input("Valor inicial x0: "))
        tol = float(input("Tolerancia (ej: 0.0001): "))
        max_iter = int(input("Máximo de iteraciones: "))
        criterio = input("Criterio de parada (error/iteracion): ")
        
        iteraciones, latex = mn.punto_fijo(func_str, x0, tol, max_iter, criterio)
        
    elif opcion == '2':
        x0 = float(input("Valor inicial x0: "))
        tol = float(input("Tolerancia (ej: 0.0001): "))
        max_iter = int(input("Máximo de iteraciones: "))
        criterio = input("Criterio de parada (error/iteracion): ")
        
        iteraciones, latex = mn.newton_raphson(func_str, x0, tol, max_iter, criterio)
        
    elif opcion == '3':
        x0 = float(input("Valor inicial x0: "))
        x1 = float(input("Valor inicial x1: "))
        tol = float(input("Tolerancia (ej: 0.0001): "))
        max_iter = int(input("Máximo de iteraciones: "))
        criterio = input("Criterio de parada (error/iteracion): ")
        
        iteraciones, latex = mn.secante(func_str, x0, x1, tol, max_iter, criterio)
    
    else:
        print("Opción no válida")
        return
    
    print("\n=== RESULTADOS ===")
    for it in iteraciones[-5:]:  # Mostrar últimas 5 iteraciones
        print(f"Iteración {it['iter']}: {it}")
    
    print("\n=== SALIDA LATEX ===")
    print(latex)
    
    with open('resultado.tex', 'w') as f:
        f.write(latex)
    print("\nArchivo LaTeX guardado como 'resultado.tex'")

if __name__ == '__main__':
    try:
        root = tk.Tk()
        app = MetodosNumericosGUI(root)
        root.mainloop()
    except Exception:
        while True:
            menu()
            continuar = input("\n¿Desea continuar? (s/n): ")
            if continuar.lower() != 's':
                break

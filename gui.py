import customtkinter as ctk
from tkinter import filedialog, messagebox
import os
import shutil
import subprocess
import tempfile
import webbrowser
from iterative_methods import IterativeMethods
from derivative_methods import DerivativeMethods
from secant_method import SecantMethod


class NumericalMethodsGUI:
    """Interfaz gráfica para la aplicación de métodos numéricos."""
    
    def __init__(self):
        # Configurar customtkinter
        ctk.set_appearance_mode("dark")  # Opciones: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Opciones: "blue", "green", "dark-blue"
        
        self.root = ctk.CTk()
        self.root.title("Métodos Numéricos")
        self.root.geometry("800x600")  # Tamaño inicial

        # Inicializar métodos numéricos
        self.iterative_methods = IterativeMethods()
        self.derivative_methods = DerivativeMethods()
        self.secant_method = SecantMethod()

        # Variables de control
        self.method_var = ctk.StringVar(value="punto_fijo")
        self.function_var = ctk.StringVar(value="0.5*x+1.0/x")
        self.x0_var = ctk.StringVar(value="1")
        self.x1_var = ctk.StringVar(value="2")
        self.tolerance_var = ctk.StringVar(value="1e-6")
        self.max_iterations_var = ctk.StringVar(value="100")
        self.stopping_criterion_var = ctk.StringVar(value="error")

        self._build_ui()
        self.handle_method_change()

    def _build_ui(self):
        """Construye la interfaz de usuario."""
        # Frame principal
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Frame de controles
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(pady=10, padx=10, fill="x")
        
        # Método
        ctk.CTkLabel(control_frame, text="Método:", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
        method_cb = ctk.CTkComboBox(
            control_frame,
            variable=self.method_var,
            values=["punto_fijo", "newton_raphson", "secante"],
            width=200
        )
        method_cb.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))
        method_cb.bind("<<ComboboxSelected>>", lambda e: self.handle_method_change())
        
        # Función
        ctk.CTkLabel(control_frame, text="Función:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        func_entry = ctk.CTkEntry(control_frame, textvariable=self.function_var, width=200)
        func_entry.grid(row=1, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # x0
        ctk.CTkLabel(control_frame, text="x0:", font=("Arial", 12, "bold")).grid(row=2, column=0, sticky="w", pady=5)
        x0_entry = ctk.CTkEntry(control_frame, textvariable=self.x0_var, width=200)
        x0_entry.grid(row=2, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # x1
        ctk.CTkLabel(control_frame, text="x1 (secante):", font=("Arial", 12, "bold")).grid(row=3, column=0, sticky="w", pady=5)
        self.x1_entry = ctk.CTkEntry(control_frame, textvariable=self.x1_var, width=200)
        self.x1_entry.grid(row=3, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Tolerancia
        ctk.CTkLabel(control_frame, text="Tolerancia (ε):", font=("Arial", 12, "bold")).grid(row=4, column=0, sticky="w", pady=5)
        tol_entry = ctk.CTkEntry(control_frame, textvariable=self.tolerance_var, width=200)
        tol_entry.grid(row=4, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Máximo de iteraciones
        ctk.CTkLabel(control_frame, text="Máx. iteraciones:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        max_iter_entry = ctk.CTkEntry(control_frame, textvariable=self.max_iterations_var, width=200)
        max_iter_entry.grid(row=5, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Criterio de parada
        ctk.CTkLabel(control_frame, text="Criterio de parada:", font=("Arial", 12, "bold")).grid(row=6, column=0, sticky="w", pady=5)
        criterio_cb = ctk.CTkComboBox(
            control_frame,
            variable=self.stopping_criterion_var,
            values=["error", "iteracion"],
            width=200
        )
        criterio_cb.grid(row=6, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Botones
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.grid(row=7, column=0, columnspan=2, pady=20, sticky="ew")
        
        run_button = ctk.CTkButton(
            button_frame,
            text="Ejecutar",
            command=self.execute_method,
            width=100,
            height=40
        )
        run_button.pack(side="left", padx=5)
        
        clear_button = ctk.CTkButton(
            button_frame,
            text="Limpiar",
            command=self.clear_output,
            width=100,
            height=40
        )
        clear_button.pack(side="left", padx=5)
        
        # Área de salida
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        ctk.CTkLabel(output_frame, text="Salida LaTeX:", font=("Arial", 14, "bold")).pack(pady=5)
        
        # Crear un textbox con scroll usando tkinter Text dentro de CTkScrollableFrame
        self.output = ctk.CTkTextbox(output_frame, height=300)
        self.output.pack(pady=5, padx=5, fill="both", expand=True)
        
        self.last_latex = ""

    def handle_method_change(self):
        """Maneja el cambio de método seleccionado."""
        method = self.method_var.get()
        if method == "secante":
            self.x1_entry.configure(state="normal")
        else:
            self.x1_entry.configure(state="disabled")

        if method == "punto_fijo":
            if self.function_var.get().strip() in ("", "x**2 - 2", "x**2-2"):
                self.function_var.set("0.5*(x + 2/x)")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
        else:
            if self.function_var.get().strip() in ("", "0.5*(x + 2/x)"):
                self.function_var.set("x**2 - 2")
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
            if method == "secante" and self.x1_var.get().strip() == "":
                self.x1_var.set("2")

    def clear_output(self):
        """Limpia el área de salida."""
        self.output.delete("0.0", "end")
        self.last_latex = ""

    def execute_method(self):
        """Ejecuta el método numérico seleccionado."""
        func_str = self.function_var.get().strip()
        if not func_str:
            messagebox.showerror("Error", "La función no puede estar vacía")
            return

        try:
            x0 = float(self.x0_var.get())
            tol = float(self.tolerance_var.get())
            max_iter = int(float(self.max_iterations_var.get()))
            stopping_criterion = self.stopping_criterion_var.get().strip()

            if stopping_criterion not in ("error", "iteracion"):
                raise ValueError("Criterio debe ser 'error' o 'iteracion'")

            method = self.method_var.get()
            if method == "punto_fijo":
                iterations, latex = self.iterative_methods.punto_fijo(func_str, x0, tol, max_iter, stopping_criterion)
            elif method == "newton_raphson":
                iterations, latex = self.derivative_methods.newton_raphson(func_str, x0, tol, max_iter, stopping_criterion)
            elif method == "secante":
                x1 = float(self.x1_var.get())
                iterations, latex = self.secant_method.secante(func_str, x0, x1, tol, max_iter, stopping_criterion)
            else:
                raise ValueError("Método no soportado")

            self.last_latex = latex
            self.output.delete("1.0", "end")
            self.output.insert("1.0", latex)

            if not iterations:
                messagebox.showwarning("Atención",
                                       "No se generaron iteraciones (posible división por 0 o problema numérico).")

            self.save_pdf_report()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_pdf_report(self):
        """Guarda el resultado como PDF."""
        if not self.last_latex:
            messagebox.showerror("Error", "No hay salida LaTeX. Primero ejecuta un método.")
            return

        if shutil.which("pdflatex") is None:
            messagebox.showerror(
                "Error",
                "No se encontró 'pdflatex'. Instala TeX Live/MiKTeX para generar PDF.",
            )
            return

        try:
            pdf_path = filedialog.asksaveasfilename(
                defaultextension=".pdf",
                filetypes=[("PDF", "*.pdf"), ("Todos", "*")],
                initialfile="resultado.pdf",
            )
            if not pdf_path:
                return

            with tempfile.TemporaryDirectory(prefix="mn_pdf_") as tmpdir:
                tex_filename = "resultado.tex"
                tex_path = os.path.join(tmpdir, tex_filename)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(self.last_latex)

                proc = None
                for _ in range(2):
                    proc = subprocess.run(
                        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_filename],
                        cwd=tmpdir,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc.returncode != 0:
                        break

                tmp_pdf = os.path.join(tmpdir, "resultado.pdf")
                if proc is None or proc.returncode != 0 or not os.path.exists(tmp_pdf):
                    stdout = "" if proc is None else proc.stdout
                    stderr = "" if proc is None else proc.stderr
                    msg = stdout[-2000:] + "\n" + stderr[-2000:]
                    messagebox.showerror("Error", "Falló la compilación LaTeX.\n\n" + msg)
                    return

                shutil.copyfile(tmp_pdf, pdf_path)

                # Abrir el PDF automáticamente
                try:
                    webbrowser.open(f"file://{os.path.abspath(pdf_path)}")
                except Exception as e:
                    print(f"No se pudo abrir el PDF automáticamente: {e}")

            messagebox.showinfo("OK", f"PDF generado en: {pdf_path}")
        except Exception as e:
            messagebox.showerror("Error", str(e))

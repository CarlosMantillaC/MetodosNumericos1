# Importaciones necesarias para la interfaz gráfica y funcionalidades del sistema
import customtkinter as ctk  # Biblioteca moderna para interfaces gráficas en Python
from tkinter import filedialog, messagebox  # Diálogos estándar de tkinter para archivos y mensajes
import os  # Funcionalidades del sistema operativo para manejo de rutas
import shutil  # Utilidades para operaciones de archivos de alto nivel
import subprocess  # Ejecución de comandos del sistema (para pdflatex)
import tempfile  # Creación de directorios temporales para compilación LaTeX
import webbrowser  # Apertura de archivos PDF en el navegador por defecto
from iterative_methods import IterativeMethods  # Método de punto fijo
from derivative_methods import DerivativeMethods  # Método de Newton-Raphson
from secant_method import SecantMethod  # Método de la secante


class NumericalMethodsGUI:
    """
    Interfaz gráfica para la aplicación de métodos numéricos.
    
    Esta clase proporciona una interfaz de usuario completa para ejecutar tres métodos
    numéricos de búsqueda de raíces:
    1. Método de Aproximaciones Sucesivas (Punto Fijo)
    2. Método de Newton-Raphson
    3. Método de la Secante
    
    Características principales:
    - Entrada de datos intuitiva con validación
    - Selección dinámica del método numérico
    - Configuración de parámetros (tolerancia, iteraciones, criterio de parada)
    - Generación automática de informes LaTeX
    - Compilación y visualización de PDFs
    - Manejo de errores con mensajes informativos
    
    La interfaz utiliza customtkinter para un aspecto moderno y profesional,
    con tema oscuro por defecto y controles responsivos.
    """
    
    def __init__(self):
        """
        Inicializa la interfaz gráfica y configura los componentes principales.
        
        Este método realiza las siguientes tareas:
        1. Configura el tema y apariencia de customtkinter
        2. Crea la ventana principal con título y tamaño
        3. Inicializa las instancias de los métodos numéricos
        4. Configura las variables de control para los widgets
        5. Construye la interfaz de usuario completa
        6. Establece los valores iniciales según el método seleccionado
        """
        # Configurar customtkinter con tema oscuro y color azul
        ctk.set_appearance_mode("dark")  # Opciones: "dark", "light", "system"
        ctk.set_default_color_theme("blue")  # Opciones: "blue", "green", "dark-blue"
        
        # Crear ventana principal
        self.root = ctk.CTk()
        self.root.title("Métodos Numéricos")
        self.root.geometry("800x600")  # Tamaño inicial de la ventana

        # Inicializar instancias de los métodos numéricos
        self.iterative_methods = IterativeMethods()
        self.derivative_methods = DerivativeMethods()
        self.secant_method = SecantMethod()

        # Variables de control para los widgets de la interfaz
        self.method_var = ctk.StringVar(value="punto_fijo")  # Método numérico seleccionado
        self.function_var = ctk.StringVar(value="sqrt(5*x - 4)")  # Función matemática
        self.function_mode_var = ctk.StringVar(value="auto")  # Modo: auto (f(x)) o manual (g(x))
        self.x0_var = ctk.StringVar(value="1")  # Valor inicial x0
        self.x1_var = ctk.StringVar(value="2")  # Valor inicial x1 (para secante)
        self.tolerance_var = ctk.StringVar(value="1e-6")  # Tolerancia para convergencia
        self.max_iterations_var = ctk.StringVar(value="100")  # Máximo de iteraciones
        self.stopping_criterion_var = ctk.StringVar(value="error")  # Criterio de parada

        # Construir la interfaz de usuario y establecer valores iniciales
        self._build_ui()
        self.handle_method_change()

    def _build_ui(self):
        """
        Construye la interfaz de usuario completa con todos los componentes.
        
        Este método crea el layout principal de la aplicación:
        1. Frame principal contenedor
        2. Frame de controles con campos de entrada
        3. Frame de botones de acción
        4. Frame de salida para mostrar resultados LaTeX
        
        La interfaz está organizada verticalmente con un diseño limpio
        y espaciado adecuado para facilitar el uso.
        """
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
            width=200,
            command=lambda choice: self.handle_method_change()
        )
        method_cb.grid(row=0, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # También bind para el evento de cambio
        method_cb.bind("<<ComboboxSelected>>", lambda e: self.handle_method_change())
        
        # Función
        ctk.CTkLabel(control_frame, text="Función:", font=("Arial", 12, "bold")).grid(row=1, column=0, sticky="w", pady=5)
        
        # Frame para función y modo (solo visible para punto fijo)
        function_frame = ctk.CTkFrame(control_frame)
        function_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=5)
        
        # Selector de modo (solo para punto fijo)
        self.mode_label = ctk.CTkLabel(function_frame, text="Modo:", font=("Arial", 10, "bold"))
        self.mode_label.grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        self.mode_cb = ctk.CTkComboBox(
            function_frame,
            variable=self.function_mode_var,
            values=["auto", "manual"],
            width=120,
            command=lambda e: self.handle_function_mode_change()
        )
        self.mode_cb.grid(row=0, column=1, sticky="w", padx=5, pady=2)
        
        # Etiqueta de descripción del modo
        self.mode_desc_label = ctk.CTkLabel(function_frame, text="", font=("Arial", 9, "italic"))
        self.mode_desc_label.grid(row=0, column=2, sticky="w", padx=10, pady=2)
        
        # Campo de entrada de función
        self.function_entry = ctk.CTkEntry(function_frame, textvariable=self.function_var, width=400)
        self.function_entry.grid(row=1, column=0, columnspan=3, sticky="ew", pady=5, padx=5)
        
        # Etiqueta informativa para punto fijo
        self.info_label = ctk.CTkLabel(control_frame, text="", font=("Arial", 10, "italic"))
        self.info_label.grid(row=4, column=0, columnspan=3, sticky="w", pady=2)
        
        # x0
        ctk.CTkLabel(control_frame, text="x0:", font=("Arial", 12, "bold")).grid(row=5, column=0, sticky="w", pady=5)
        x0_entry = ctk.CTkEntry(control_frame, textvariable=self.x0_var, width=200)
        x0_entry.grid(row=5, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # x1
        ctk.CTkLabel(control_frame, text="x1 (secante):", font=("Arial", 12, "bold")).grid(row=6, column=0, sticky="w", pady=5)
        self.x1_entry = ctk.CTkEntry(control_frame, textvariable=self.x1_var, width=200)
        self.x1_entry.grid(row=6, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Inicialmente deshabilitar x1 (solo para secante)
        self.x1_entry.configure(state="disabled")
        
        # Tolerancia
        ctk.CTkLabel(control_frame, text="Tolerancia (ε):", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky="w", pady=5)
        tol_entry = ctk.CTkEntry(control_frame, textvariable=self.tolerance_var, width=200)
        tol_entry.grid(row=7, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Máximo de iteraciones
        ctk.CTkLabel(control_frame, text="Máx. iteraciones:", font=("Arial", 12, "bold")).grid(row=8, column=0, sticky="w", pady=5)
        max_iter_entry = ctk.CTkEntry(control_frame, textvariable=self.max_iterations_var, width=200)
        max_iter_entry.grid(row=8, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Criterio de parada
        ctk.CTkLabel(control_frame, text="Criterio de parada:", font=("Arial", 12, "bold")).grid(row=9, column=0, sticky="w", pady=5)
        criterio_cb = ctk.CTkComboBox(
            control_frame,
            variable=self.stopping_criterion_var,
            values=["error", "iteracion"],
            width=200
        )
        criterio_cb.grid(row=9, column=1, sticky="w", pady=5, padx=(10, 0))
        
        # Botones
        button_frame = ctk.CTkFrame(control_frame)
        button_frame.grid(row=10, column=0, columnspan=2, pady=20, sticky="ew")
        
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

    def handle_function_mode_change(self):
        """
        Maneja el cambio entre modo automático y manual para el método de punto fijo.
        
        Modo automático: Ingresa f(x) y el software lo convierte a g(x)
        Modo manual: Ingresa directamente g(x) para el método de punto fijo
        """
        mode = self.function_mode_var.get()
        method = self.method_var.get()
        
        if method == "punto_fijo":
            if mode == "auto":
                self.mode_desc_label.configure(text="Ingresa f(x) y el software la convertirá a g(x)")
                self.info_label.configure(
                    text="💡 Ingresa f(x) y el software la convertirá automáticamente a g(x) para el método de punto fijo"
                )
                # Ejemplo de función f(x)
                if self.function_var.get().strip() in ("", "0.5*(x + 2/x)"):
                    self.function_var.set("sqrt(5*x - 4)")
            else:  # manual
                self.mode_desc_label.configure(text="Ingresa directamente g(x) para el método de punto fijo")
                self.info_label.configure(
                    text="💡 Ingresa directamente la función g(x) en formato x = g(x)"
                )
                # Ejemplo de función g(x)
                if self.function_var.get().strip() in ("", "sqrt(5*x - 4)"):
                    self.function_var.set("0.5*(x + 2/x)")

    def handle_method_change(self):
        """
        Maneja el cambio de método numérico seleccionado.
        
        Este método se ejecuta cuando el usuario selecciona un método diferente
        del menú desplegable y realiza las siguientes acciones:
        
        1. Habilita/deshabilita el campo x1 según el método:
           - Método de la secante: habilita x1 (requiere dos valores iniciales)
           - Otros métodos: deshabilita x1 (solo necesitan un valor inicial)
        
        2. Ajusta automáticamente la función y valores iniciales según el método:
           - Punto fijo: usa función f(x) original, el software la convierte a g(x)
           - Newton-Raphson/Secante: usa función f(x) directamente
           - Secante adicionalmente: x1 = 2
        
        3. Muestra información útil sobre el método seleccionado
        """
        method = self.method_var.get()
        
        # Controlar la visibilidad del campo x1 según el método seleccionado
        if method == "secante":
            self.x1_entry.configure(state="normal")  # Habilitar para secante
            # Forzar actualización de la interfaz
            self.x1_entry.update()
        else:
            self.x1_entry.configure(state="disabled")  # Deshabilitar para otros métodos
            # Forzar actualización de la interfaz
            self.x1_entry.update()

        # Actualizar etiqueta informativa según el método
        if method == "punto_fijo":
            # Mostrar controles de modo para punto fijo
            self.mode_label.grid()
            self.mode_desc_label.grid()
            self.mode_cb.grid()  # Asegurar que el combobox también se muestre
            
            # Actualizar según el modo actual
            self.handle_function_mode_change()
            
            # Forzar actualización del frame
            function_frame = self.mode_cb.master
            function_frame.update()
            
        elif method == "newton_raphson":
            # Ocultar completamente controles de modo para otros métodos
            self.mode_label.grid_remove()
            self.mode_desc_label.grid_remove()
            self.mode_cb.grid_remove()  # Ocultar también el combobox
            # Forzar actualización del frame
            function_frame = self.mode_cb.master
            function_frame.update()
            self.info_label.configure(
                text="💡 Ingresa f(x) directamente. El método calculará la derivada automáticamente."
            )
        elif method == "secante":
            # Ocultar completamente controles de modo para otros métodos
            self.mode_label.grid_remove()
            self.mode_desc_label.grid_remove()
            self.mode_cb.grid_remove()  # Ocultar también el combobox
            # Forzar actualización del frame
            function_frame = self.mode_cb.master
            function_frame.update()
            self.info_label.configure(
                text="💡 Ingresa f(x) directamente. El método no requiere derivada."
            )
        else:
            # Ocultar completamente controles de modo
            self.mode_label.grid_remove()
            self.mode_desc_label.grid_remove()
            self.mode_cb.grid_remove()  # Ocultar también el combobox
            self.info_label.configure(text="")

        # Ajustar automáticamente la función según el método
        if method == "punto_fijo":
            # Para punto fijo, ahora aceptamos cualquier función f(x)
            if self.function_var.get().strip() == "":
                self.function_var.set("sqrt(5*x - 4)")  # Ejemplo de función
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
        else:
            # Para Newton-Raphson y secante, usar funciones estándar
            if self.function_var.get().strip() == "":
                self.function_var.set("x**2 - 2")  # f(x) = x^2 - 2
            if self.x0_var.get().strip() == "":
                self.x0_var.set("1")
            # Para la secante, también establecer x1
            if method == "secante" and self.x1_var.get().strip() == "":
                self.x1_var.set("2")

    def clear_output(self):
        """
        Limpia el área de salida de resultados.
        
        Este método realiza las siguientes acciones:
        1. Elimina todo el contenido del área de texto LaTeX
        2. Reinicia la variable que almacena el último código LaTeX generado
        3. Prepara la interfaz para una nueva ejecución
        
        Se utiliza cuando el usuario presiona el botón "Limpiar" o antes
        de ejecutar un nuevo método numérico.
        """
        self.output.delete("0.0", "end")  # Limpiar todo el contenido del textbox
        self.last_latex = ""  # Reiniciar el código LaTeX almacenado

    def execute_method(self):
        """
        Ejecuta el método numérico seleccionado con los parámetros proporcionados.
        
        Este es el método principal que coordina la ejecución completa:
        
        1. Validación de entrada:
           - Verifica que la función no esté vacía
           - Convierte los parámetros a los tipos correctos
           - Valida el criterio de parada
        
        2. Ejecución del método numérico:
           - Selecciona el método apropiado según la elección del usuario
           - Pasa los parámetros configurados
           - Obtiene las iteraciones y el código LaTeX
        
        3. Presentación de resultados:
           - Muestra el código LaTeX en el área de salida
           - Genera y abre automáticamente el PDF
           - Muestra advertencias si no hay iteraciones
        
        4. Manejo de errores:
           - Captura y muestra errores de forma amigable
           - Proporciona mensajes informativos sobre problemas comunes
        """
        func_str = self.function_var.get().strip()
        if not func_str:
            messagebox.showerror("Error", "La función no puede estar vacía")
            return

        try:
            method = self.method_var.get()
            x0 = float(self.x0_var.get())
            tol = float(self.tolerance_var.get())
            max_iter = int(float(self.max_iterations_var.get()))
            stopping_criterion = self.stopping_criterion_var.get().strip()
            
            # Limpiar salida anterior
            self.output.delete("1.0", "end")
            
            # Ejecutar el método numérico según la selección
            if method == "punto_fijo":
                # Verificar si es modo automático o manual
                mode = self.function_mode_var.get()
                
                if mode == "auto":
                    # Modo automático: convertir f(x) a g(x)
                    iterations, latex = self.iterative_methods.punto_fijo(
                        func_str, x0, tol, max_iter, stopping_criterion
                    )
                else:  # manual
                    # Modo manual: usar directamente g(x)
                    iterations, latex = self.iterative_methods.punto_fijo_manual(
                        func_str, x0, tol, max_iter, stopping_criterion
                    )
                    
            elif method == "newton_raphson":
                iterations, latex = self.derivative_methods.newton_raphson(
                    func_str, x0, tol, max_iter, stopping_criterion
                )
                
            elif method == "secante":
                # Validar que x1 no esté vacío
                x1_str = self.x1_var.get().strip()
                if not x1_str:
                    messagebox.showerror("Error", "El campo x1 no puede estar vacío para el método de la secante")
                    return
                
                try:
                    x1 = float(x1_str)
                except ValueError:
                    messagebox.showerror("Error", f"El valor de x1 '{x1_str}' no es un número válido")
                    return
                
                iterations, latex = self.secant_method.secante(
                    func_str, x0, x1, tol, max_iter, stopping_criterion
                )
            
            # Almacenar el código LaTeX para uso posterior
            self.last_latex = latex
            
            # Mostrar el código LaTeX en el área de salida
            self.output.insert("1.0", latex)

            if not iterations:
                messagebox.showwarning("Atención",
                                       "No se generaron iteraciones (posible división por 0 o problema numérico).")

            self.save_pdf_report()

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def save_pdf_report(self):
        """
        Genera y guarda un informe PDF a partir del código LaTeX.
        
        Este método implementa el flujo completo de compilación LaTeX:
        
        1. Verificación de requisitos:
           - Comprueba que haya código LaTeX disponible
           - Verifica que pdflatex esté instalado en el sistema
           - Muestra mensajes de error si faltan requisitos
        
        2. Selección de archivo:
           - Abre un diálogo para que el usuario seleccione la ubicación
           - Sugiere un nombre por defecto: "resultado.pdf"
           - Permite cancelar la operación
        
        3. Compilación LaTeX:
           - Crea un directorio temporal para los archivos intermedios
           - Escribe el código LaTeX a un archivo .tex
           - Ejecuta pdflatex dos veces para referencias cruzadas correctas
           - Captura errores de compilación si ocurren
        
        4. Manejo de resultados:
           - Copia el PDF generado a la ubicación seleccionada
           - Abre automáticamente el PDF en el visor por defecto
           - Limpia los archivos temporales
        
        5. Manejo de errores:
           - Muestra errores de compilación LaTeX de forma clara
           - Proporciona sugerencias para problemas comunes
           - Limpia recursos temporales incluso en caso de error
        
        Nota: Este método requiere que LaTeX (pdflatex) esté instalado
        en el sistema para funcionar correctamente.
        """
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

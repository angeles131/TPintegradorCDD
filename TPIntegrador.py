import tkinter as tk
from tkinter import filedialog, messagebox 
from tkinter import ttk 
import numpy as np # Importa NumPy para operaciones numéricas eficientes, especialmente con arreglos de datos de audio.
import sounddevice as sd # Importa SoundDevice para grabar y reproducir audio.
import wavio # Importa wavio para guardar arrays de NumPy como archivos WAV.
from pydub import AudioSegment # Importa AudioSegment de pydub para manipular archivos de audio de forma sencilla.
import librosa # Importa Librosa para análisis de audio, espectrogramas.
import librosa.display # Importa librosa.display para visualizar espectrogramas.
import matplotlib.pyplot as plt # Importa Matplotlib para crear gráficos.
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # Importa esta clase para integrar gráficos en Tkinter.
import os # Importa el módulo os para interactuar con el sistema operativo (borrar archivos temporales).

# Define la tasa de muestreo para la grabación de audio.
# Una tasa de 44100 Hz es estándar para audio de calidad, imita analogico.
SAMPLE_RATE_REC = 44100  
# Nombre del archivo temporal donde se guardará la grabación antes de procesarla.
TEMP_REC_FILENAME = "temp_recording.wav"

class AudioConverterApp:
    """
    Clase principal que define la aplicación de conversión de audio.
    Maneja la interfaz de usuario, la grabación, conversión y exportación de audio,
    además de la visualización de espectrogramas.
    """
    def __init__(self, root):
        
        self.root = root 
        self.root.title("Conversor de Audio Analógico a Digital")
        self.root.geometry("1700x800")
        self.root.configure(bg="#fefae0")
        
        # Variables de instancia para almacenar los datos de audio y la ruta del archivo.
        self.audio_original = None      # Almacena el objeto AudioSegment del audio grabado original.
        self.audio_digitalizado = None  # Almacena el objeto AudioSegment del audio después de la conversión.
        self.ruta_archivo = None        # Guarda la ruta del archivo temporal de la grabación.

        # Frames para organizar la Interfaz Gráfica (GUI)
        # Un frame para los controles
        control_frame = tk.Frame(root, padx=10, pady=10, bg="#8d99ae")
        control_frame.pack(side=tk.TOP, fill=tk.X) # Empaqueta el frame en la parte superior, expandiéndose horizontalmente.

        # Un frame para los gráficos de espectro.
        plot_frame = tk.Frame(root, padx=10, pady=10, bg="#8d99ae")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Empaqueta el frame en la parte superior, llenando el espacio restante.
        
        # --- Controles de Grabación de Duración Personalizada ---
        # Etiqueta para indicar al usuario qué ingresar.
        tk.Label(control_frame, text="Duración (segundos):", font=("Arial",15,"bold"), bg="#8d99ae").pack(side=tk.LEFT, padx=(5, 5))
        # Campo de entrada (Entry) donde el usuario puede escribir la duración de la grabación.
        self.entry_duracion = tk.Entry(control_frame, width=8, font=("Arial", 16))
        self.entry_duracion.insert(0, "5") # Inserta "5" como valor predeterminado al inicio.
        self.entry_duracion.pack(side=tk.LEFT, padx=5)
        
        # Botón para iniciar la grabación. Llama al método grabar_audio_personalizado.
        tk.Button(control_frame, text="Grabar Audio", padx=10,
                pady=2,bg="#e5989b", fg="white", activebackground="#588157",font=("Arial", 14, "bold"),command=self.grabar_audio_personalizado).pack(side=tk.LEFT, padx=6)

        # --- Controles de Conversión (Muestreo y Cuantización) ---
        # Etiqueta para el control de muestreo.
        tk.Label(control_frame, text="Muestreo (Hz):",font=("Arial",14,"bold"),bg="#8d99ae").pack(side=tk.LEFT, padx=(15, 5))
        # Combobox para seleccionar la tasa de muestreo.
        self.combo_muestreo = ttk.Combobox(control_frame, width=10, font=("Arial", 16), values=["44100", "22050", "16000", "8000"], style="Estilos.TCombobox")
        self.combo_muestreo.set("22050") # Establece "22050" como opción preseleccionada.
        self.combo_muestreo.pack(side=tk.LEFT, padx=5)

        # Etiqueta para el control de cuantización.
        tk.Label(control_frame, text="Cuantización (bits):",font=("Arial",15,"bold"),bg="#8d99ae").pack(side=tk.LEFT, padx=(15, 5))
        # Combobox para seleccionar la profundidad de bits (cuantización).
        self.combo_cuantizacion = ttk.Combobox(control_frame,width=10, font=("Arial", 16), values=["16", "8"], style="Estilos.TCombobox")
        self.combo_cuantizacion.set("16") # Establece "16" como opción preseleccionada.
        self.combo_cuantizacion.pack(side=tk.LEFT, padx=5)

        # Botón para iniciar el proceso de conversión.
        tk.Button(control_frame, text="Convertir",bg="#e5989b",fg="white",activebackground="#588157",font=("Arial", 14, "bold"), command=self.convertir_audio).pack(side=tk.LEFT, padx=15)
        # Botón para exportar el audio digitalizado.
        tk.Button(control_frame, text="Exportar (WAV/MP3)",bg="#e5989b",fg="white",activebackground="#588157",font=("Arial", 14, "bold") ,command=self.exportar_audio).pack(side=tk.LEFT,padx=5)

        # --- Área de Gráficos (Matplotlib) ---
        # Crea una figura de Matplotlib con dos subplots (ax1 para original, ax2 para digitalizado).
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        # Crea un lienzo (canvas) para integrar la figura de Matplotlib en el frame de Tkinter.
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True) # Empaqueta el lienzo.
        self.inicializar_graficos() # Llama a la función para configurar los títulos iniciales de los gráficos.

    def inicializar_graficos(self):
        """
        Prepara los gráficos de Matplotlib con títulos y etiquetas iniciales.
        Esto asegura que los ejes estén etiquetados y los títulos sean visibles
        incluso antes de que se grabe o convierta el audio.
        """
        self.ax1.set_title("Espectro de Frecuencia - Audio Original (Grabado)")
        self.ax1.set_ylabel("Frecuencia (Hz)")
        self.ax2.set_title("Espectro de Frecuencia - Audio Digitalizado")
        self.ax2.set_xlabel("Tiempo (s)")
        self.ax2.set_ylabel("Frecuencia (Hz)")
        self.fig.tight_layout() # Ajusta automáticamente los parámetros del subplot para un diseño ajustado.
        self.canvas.draw() # Dibuja la figura en el lienzo de Tkinter.

    def grabar_audio_personalizado(self):
        """
        Graba audio desde el micrófono por una duración especificada por el usuario.
        Esta función:
        1. Obtiene la duración del campo de entrada.
        2. Valida la entrada para asegurar que sea un número positivo.
        3. Muestra un mensaje al usuario para indicar que la grabación ha comenzado.
        4. Utiliza sounddevice para grabar el audio.
        5. Guarda la grabación en un archivo WAV temporal.
        6. Carga el audio grabado en un objeto AudioSegment de pydub.
        7. Muestra el espectrograma del audio original.
        8. Informa al usuario que la grabación fue exitosa.
        9. Maneja errores si la entrada no es válida o si la grabación falla.
        """
        try:
            duracion_str = self.entry_duracion.get() # Obtiene el texto del campo de entrada de duración.
            duracion = float(duracion_str) # Intenta convertir el texto a un número flotante (permite decimales).

            # Validación de la duración ingresada.
            if duracion <= 0:
                messagebox.showerror("Error de Entrada", "La duración debe ser un número positivo.")
                return # Sale de la función si la duración no es válida.

            messagebox.showinfo("Grabando", f"Se grabará audio durante {duracion:.1f} segundos.")
            # Inicia la grabación usando sounddevice.
            # int(duracion * SAMPLE_RATE_REC) calcula el número total de fotogramas a grabar.
            # channels=1 para grabación mono, dtype='float64' para alta precisión.
            grabacion = sd.rec(int(duracion * SAMPLE_RATE_REC), samplerate=SAMPLE_RATE_REC, channels=1, dtype='float64')
            sd.wait()  # Espera hasta que la grabación (el array 'grabacion') se haya completado.

            # Guarda los datos de la grabación (NumPy array) en un archivo WAV temporal.
            # sampwidth=3 significa que se guardará como 24-bit PCM (float64 se mapea a esto).
            wavio.write(TEMP_REC_FILENAME, grabacion, SAMPLE_RATE_REC, sampwidth=3) 

            # Carga el archivo WAV temporal en un objeto AudioSegment de pydub.
            self.ruta_archivo = TEMP_REC_FILENAME
            self.audio_original = AudioSegment.from_file(self.ruta_archivo)
            
            # Muestra el espectrograma del audio recién grabado en el primer subplot.
            self.mostrar_espectro(self.audio_original, self.ax1, "Espectro de Frecuencia - Audio Grabado")
            messagebox.showinfo("Éxito", f"Audio de {duracion:.1f} segundos grabado y listo para convertir.")

        except ValueError:
            # Captura el error si el usuario no ingresa un número válido.
            messagebox.showerror("Error de Entrada", "Por favor, ingrese un número válido para la duración.")
        except Exception as e:
            # Captura cualquier otro error que pueda ocurrir durante la grabación.
            messagebox.showerror("Error de Grabación", f"No se pudo grabar el audio: {e}")

    def convertir_audio(self):
        """
        1. Verifica si hay un audio grabado para convertir.
        2. Obtiene los valores de muestreo y cuantización seleccionados por el usuario.
        3. Realiza el remuestreo (sampling) del audio usando pydub.
        4. Cambia la profundidad de bits (cuantización) del audio remuestreado.
        5. Muestra el espectrograma del audio digitalizado.
        """
        if not self.audio_original:
            messagebox.showwarning("Advertencia", "Primero grabe un audio.")
            return # Sale de la función si no hay audio grabado.

        try:
            # Obtiene la nueva frecuencia de muestreo
            nueva_frecuencia = int(self.combo_muestreo.get())
            # Obtiene la nueva profundidad de bits
            nueva_profundidad_bits = int(self.combo_cuantizacion.get())
            
            # 1. Muestreo: Cambia la tasa de muestreo del audio.
            audio_muestreado = self.audio_original.set_frame_rate(nueva_frecuencia)

            # 2. Cuantización: Cambia la profundidad de bits del audio.
            # pydub.AudioSegment.set_sample_width espera el ancho en bytes, por eso se divide por 8.
            self.audio_digitalizado = audio_muestreado.set_sample_width(nueva_profundidad_bits // 8)

            # Muestra el espectrograma del audio digitalizado en el segundo subplot.
            self.mostrar_espectro(self.audio_digitalizado, self.ax2, "Espectro de Frecuencia - Audio Digitalizado")
            
            messagebox.showinfo("Éxito", "Audio convertido. Ahora puede exportarlo.")

        except Exception as e:
            messagebox.showerror("Error de Conversión", f"Ocurrió un error: {e}")

    def mostrar_espectro(self, audio_segment, ax, title):
        """
        Se usa tanto para el audio original como para el digitalizado.
        1. Convierte el AudioSegment a un array de NumPy.
        2. Normaliza las muestras si es necesario.
        3. Limpia el eje del gráfico.
        4. Calcula el espectrograma Mel (Mel-spectrogram) usando librosa.
        5. Convierte la potencia a decibelios (dB) para una mejor visualización.
        6. Dibuja el espectrograma en el eje de Matplotlib.
        7. Actualiza el lienzo de Tkinter para mostrar los cambios.


            audio_segment (pydub.AudioSegment): El segmento de audio a analizar.
            ax (matplotlib.axes.Axes): El objeto de eje de Matplotlib donde se dibujará el espectrograma.
          
        """
        # Convierte el AudioSegment a un array de NumPy de tipo float32.
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        
        # Normaliza las muestras si la profundidad de bits es mayor que 1 byte (8 bits).
        # Esto es importante para que librosa interprete correctamente la amplitud.
        if audio_segment.sample_width > 1:
            samples /= np.iinfo(f"int{audio_segment.sample_width*8}").max # Divide por el valor máximo para el tipo de entero.

        ax.clear() # Limpia el contenido previo del eje del gráfico.
        # Calcula el espectrograma Mel. `n_mels` es el número de bandas Mel.
        S = librosa.feature.melspectrogram(y=samples, sr=audio_segment.frame_rate, n_mels=128)
        # Convierte la potencia del espectrograma a decibelios (dB) para una visualización logarítmica.
        S_DB = librosa.power_to_db(S, ref=np.max)
        # Muestra el espectrograma. x_axis='time' y y_axis='mel' configuran los ejes.
        librosa.display.specshow(S_DB, sr=audio_segment.frame_rate, x_axis='time', y_axis='mel', ax=ax)
        ax.set_title(title) # Establece el título del espectrograma.
        self.fig.tight_layout() # Ajusta el diseño de la figura.
        self.canvas.draw() 

    def exportar_audio(self):
        """
        Exporta el audio digitalizado a un archivo WAV o MP3, según la elección del usuario.
        1. Verifica si hay un audio digitalizado para exportar.
        2. Abre un cuadro de diálogo para que el usuario elija la ubicación y el formato del archivo.
        3. Exporta el audio usando pydub al formato y ruta seleccionados.
        4. Informa al usuario sobre el éxito de la exportación.
        5. Maneja errores si la exportación falla.
        """
        if not self.audio_digitalizado:
            messagebox.showwarning("Advertencia", "Primero debe convertir un audio.")
            return # Sale de la función si no hay audio digitalizado.

        # Abre un cuadro de diálogo para guardar archivo.
        # defaultextension: la extensión por defecto si el usuario no la especifica.
        # filetypes: los tipos de archivo permitidos.
        ruta_exportacion = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3")],
            title="Guardar audio digitalizado"
        )
        if not ruta_exportacion:
            return 

        try:
            # Extrae la extensión del archivo de la ruta elegida
            formato = os.path.splitext(ruta_exportacion)[1][1:] 
            # Exporta el AudioSegment digitalizado al formato y ruta especificados.
            self.audio_digitalizado.export(ruta_exportacion, format=formato)
            messagebox.showinfo("Éxito", f"Audio exportado como {ruta_exportacion}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar: {e}")

    def on_closing(self):
        """
        Función que se llama cuando el usuario intenta cerrar la ventana de la aplicación.
        Su propósito principal es limpiar el archivo de grabación temporal.
        """
        # Verifica si el archivo temporal existe y, si es así, lo elimina.
        if os.path.exists(TEMP_REC_FILENAME):
            os.remove(TEMP_REC_FILENAME)
        self.root.destroy() # Cierra la ventana de Tkinter y termina la aplicación.

if __name__ == "__main__":
    
    root = tk.Tk() # Crea la ventana principal de Tkinter.
    
    #VER 
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Estilos.TCombobox", 
                background="#e5989b",    
                foreground="black",
                font=("Arial", 13))
    app = AudioConverterApp(root) # Crea una instancia de la aplicación.
    # Configura un protocolo para manejar el evento de cierre de ventana.
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop() # Inicia el bucle principal de Tkinter, que espera eventos (clics, entradas, etc.).

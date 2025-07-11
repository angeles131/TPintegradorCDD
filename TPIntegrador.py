import tkinter as tk
from tkinter import filedialog, messagebox 
from tkinter import ttk 
import numpy as np
import sounddevice as sd
import wavio
from pydub import AudioSegment
import librosa
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os

SAMPLE_RATE_REC = 44100  
TEMP_REC_FILENAME = "temp_recording.wav"

class AudioConverterApp:
    def __init__(self, root):
        self.root = root 
        self.root.title("Conversor de Audio Analógico a Digital")
        self.root.geometry("1700x800")
        self.root.configure(bg="#fefae0")
        
        self.audio_original = None
        self.audio_digitalizado = None
        self.ruta_archivo = None

        control_frame = tk.Frame(root, padx=10, pady=10, bg="#8d99ae")
        control_frame.pack(side=tk.TOP, fill=tk.X)

        plot_frame = tk.Frame(root, padx=10, pady=10, bg="#8d99ae")
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        tk.Label(control_frame, text="Duración (segundos):", font=("Arial",15,"bold"), bg="#8d99ae").pack(side=tk.LEFT, padx=(5, 5))
        self.entry_duracion = tk.Entry(control_frame, width=8, font=("Arial", 16))
        self.entry_duracion.insert(0, "5")
        self.entry_duracion.pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Grabar Audio", padx=10,
                pady=2,bg="#e5989b", fg="white", activebackground="#588157",font=("Arial", 14, "bold"),command=self.grabar_audio_personalizado).pack(side=tk.LEFT, padx=6)

        tk.Label(control_frame, text="Muestreo (Hz):",font=("Arial",14,"bold"),bg="#8d99ae").pack(side=tk.LEFT, padx=(15, 5))
        self.combo_muestreo = ttk.Combobox(control_frame, width=10, font=("Arial", 16), values=["44100", "22050", "16000", "8000"], style="Estilos.TCombobox")
        self.combo_muestreo.set("22050")
        self.combo_muestreo.pack(side=tk.LEFT, padx=5)

        tk.Label(control_frame, text="Cuantización (bits):",font=("Arial",15,"bold"),bg="#8d99ae").pack(side=tk.LEFT, padx=(15, 5))
        self.combo_cuantizacion = ttk.Combobox(control_frame,width=10, font=("Arial", 16), values=["16", "8"], style="Estilos.TCombobox")
        self.combo_cuantizacion.set("16")
        self.combo_cuantizacion.pack(side=tk.LEFT, padx=5)

        tk.Button(control_frame, text="Convertir",bg="#e5989b",fg="white",activebackground="#588157",font=("Arial", 14, "bold"), command=self.convertir_audio).pack(side=tk.LEFT, padx=15)
        tk.Button(control_frame, text="Exportar (WAV/MP3)",bg="#e5989b",fg="white",activebackground="#588157",font=("Arial", 14, "bold") ,command=self.exportar_audio).pack(side=tk.LEFT,padx=5)

        # Configuración de los gráficos para mostrar espectros de frecuencia
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.inicializar_graficos()

    def inicializar_graficos(self):
        """Configura los gráficos iniciales para mostrar espectros de frecuencia."""
        self.ax1.set_title("Espectro de Frecuencia - Audio Original (Grabado)")
        self.ax1.set_ylabel("Amplitud")
        self.ax1.set_xlabel("Frecuencia (Hz)")
        
        self.ax2.set_title("Espectro de Frecuencia - Audio Digitalizado")
        self.ax2.set_ylabel("Amplitud")
        self.ax2.set_xlabel("Frecuencia (Hz)")
        
        self.fig.tight_layout()
        self.canvas.draw()

    def grabar_audio_personalizado(self):
        try:
            duracion_str = self.entry_duracion.get()
            duracion = float(duracion_str)

            if duracion <= 0:
                messagebox.showerror("Error de Entrada", "La duración debe ser un número positivo.")
                return

            messagebox.showinfo("Grabando", f"Se grabará audio durante {duracion:.1f} segundos.")
            grabacion = sd.rec(int(duracion * SAMPLE_RATE_REC), samplerate=SAMPLE_RATE_REC, channels=1, dtype='float64')
            sd.wait()

            wavio.write(TEMP_REC_FILENAME, grabacion, SAMPLE_RATE_REC, sampwidth=3)
            self.ruta_archivo = TEMP_REC_FILENAME
            self.audio_original = AudioSegment.from_file(self.ruta_archivo)
            
            # Mostrar espectro de frecuencia en lugar de espectrograma
            self.mostrar_espectro_frecuencia(self.audio_original, self.ax1, "Espectro de Frecuencia - Audio Grabado")
            messagebox.showinfo("Éxito", f"Audio de {duracion:.1f} segundos grabado y listo para convertir.")

        except ValueError:
            messagebox.showerror("Error de Entrada", "Por favor, ingrese un número válido para la duración.")
        except Exception as e:
            messagebox.showerror("Error de Grabación", f"No se pudo grabar el audio: {e}")

    def convertir_audio(self):
        if not self.audio_original:
            messagebox.showwarning("Advertencia", "Primero grabe un audio.")
            return

        try:
            nueva_frecuencia = int(self.combo_muestreo.get())
            nueva_profundidad_bits = int(self.combo_cuantizacion.get())
            
            audio_muestreado = self.audio_original.set_frame_rate(nueva_frecuencia)
            self.audio_digitalizado = audio_muestreado.set_sample_width(nueva_profundidad_bits // 8)

            # Mostrar espectro de frecuencia en lugar de espectrograma
            self.mostrar_espectro_frecuencia(self.audio_digitalizado, self.ax2, "Espectro de Frecuencia - Audio Digitalizado")
            
            messagebox.showinfo("Éxito", "Audio convertido. Ahora puede exportarlo.")

        except Exception as e:
            messagebox.showerror("Error de Conversión", f"Ocurrió un error: {e}")

    def mostrar_espectro_frecuencia(self, audio_segment, ax, title):
        """Muestra el espectro de frecuencia (FFT) del audio en el eje especificado."""
        samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32)
        
        if audio_segment.sample_width > 1:
            samples /= np.iinfo(f"int{audio_segment.sample_width*8}").max

        # Calcular la FFT
        n = len(samples)
        fft_result = np.fft.fft(samples)
        fft_freq = np.fft.fftfreq(n, d=1/audio_segment.frame_rate)
        
        # Tomar solo la mitad (frecuencias positivas)
        half_n = n // 2
        fft_freq = fft_freq[:half_n]
        fft_magnitude = np.abs(fft_result[:half_n])

        ax.clear()
        ax.plot(fft_freq, fft_magnitude)
        ax.set_title(title)
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Amplitud")
        ax.set_xlim(0, audio_segment.frame_rate / 2)  # Mostrar hasta la frecuencia Nyquist
        ax.grid(True)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def exportar_audio(self):
        if not self.audio_digitalizado:
            messagebox.showwarning("Advertencia", "Primero debe convertir un audio.")
            return

        ruta_exportacion = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("MP3 files", "*.mp3")],
            title="Guardar audio digitalizado"
        )
        if not ruta_exportacion:
            return 

        try:
            formato = os.path.splitext(ruta_exportacion)[1][1:] 
            self.audio_digitalizado.export(ruta_exportacion, format=formato)
            messagebox.showinfo("Éxito", f"Audio exportado como {ruta_exportacion}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo exportar: {e}")

    def on_closing(self):
        if os.path.exists(TEMP_REC_FILENAME):
            os.remove(TEMP_REC_FILENAME)
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Estilos.TCombobox", 
                background="#e5989b",    
                foreground="black",
                font=("Arial", 13))
    app = AudioConverterApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

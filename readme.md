# Herramientas de Python para laboratorios del Departamento de Óptica y Dimensional

La clase `DataReader` está diseñada para leer y procesar datos de un archivo de texto con formato de ancho fijo, convirtiéndolos en un DataFrame de pandas, y proporcionando métodos para recuperar y visualizar los datos.Si .SiS


## Flecha e interfranja

### Análisis

El archivo `flecha_interfranja.py` contiene funciones para analizar y visualizar imagenes de flechas e interfranjas. A su vez, se puede correr como script para analizar un directorio de imagenes o una imagen en particular.

La función `analyze_interference` es la que realiza el procesamiento de las imagenes.

1. Puede procesar un directorio de imagenes o una imagen en particular.
2. Se gira la imagen para alinear las franjas más verticalmente.
3. Desenfoca la imagen girada para reducir el ruido.
4. Calcula el perfil de intensidad de la imagen desenfocada.
5. Encuentra los mínimos locales en el perfil de intensidad, que corresponden a las franjas. Se obtiene un punto por franja.
6. Detecta el círculo principal que contiene las franjas.
7. Descarta las franjas cercanas al borde del círculo. Descarta puntos sospechosos que no correspondan a franjas.
8. Normaliza la imagen.
9. Identifica puntos equiespaciados verticalmente que corresponden a los valles de las franjas.
10. Mediante una optimización, encuentra las líneas paralelas y equiespaciadas que representan las franjas para una muestra plana.
11. Calcula la desviación máxima de las líneas de las franjas respecto a las líneas ideales.
12. Plotea las franjas y las líneas que las representan.

La función devuelve dos valores: `interfringe_distance` y `flecha`. `interfringe_distance` es la distancia promedio entre las líneas que representan las franjas, y `flecha` es un valor estadístico que representa la mayor desviación de las líneas de las franjas respecto a las líneas ideales.

La función también tiene varios parámetros opcionales que controlan qué se muestra y se guarda durante el análisis, como `show`, `show_result` y `save`.

Hay parámetros que no son recibidos de entrada, pero se pueden configurar:

* Búsqueda de franjas:
    * `ROTATION_IGNORE_LOW_FREQ_PIXELS`: El número de píxeles de baja frecuencia a ignorar al rotar una imagen.
    * `ROTATION_RANGE_ANGLE_DEG`: El rango de ángulos de rotación en grados para la búsqueda de rotación por contraste en el perfil de intensidad.
    * `ROTATION_N_RANGE_ANGLE`: El número de ángulos de rotación a considerar en el rango.
    * `MINIMUM_DISTANCE_PEAKS`: La distancia mínima tolerada entre picos en una señal para búsqueda de franjas en el perfil de intensidad.
    * `PROMINENCE_PEAKS`: La prominencia mínima de los picos en el perfil de intensidad.
* Blurring para analizar franjas:
    * `GAUSSIAN_BLUR_SIGMA`: La desviación estándar del kernel de suavizado gaussiano.
    * `GAUSSIAN_BLUR_SIGMAY_FACTOR`: El factor utilizado para calcular la desviación estándar del kernel de suavizado gaussiano en la dirección "y". Es útil para el análisis de franjas verticales.
    * `GAUSSIAN_BLUR_KERNEL_SIZE`: El tamaño del kernel de suavizado gaussiano.
* Detección del círculo correspondiente a la muestra (plano óptico circular):
    * `GAUSSIAN_BLUR_SIGMA_CIRCLE`: La desviación estándar del kernel de suavizado gaussiano circular.
    * `GAUSSIAN_BLUR_KERNEL_SIZE_CIRCLE`: El tamaño del kernel de suavizado gaussiano circular.
    * `HOUGH_PARAM1`: El primer parámetro para la transformada de Hough.
    * `HOUGH_PARAM2`: El segundo parámetro para la transformada de Hough.  
* Detección de puntos en franjas encontradas:  
    * `FRACTION_OF_SEPARATION_TO_SEARCH_FRINGES`: La fracción de la distancia de separación utilizada para buscar puntos de borde.
    * `MINIMUM_DISTANCE_FROM_EDGES`: La distancia mínima desde los bordes de la imagen para considerar al buscar puntos de borde.
    * `DISCARD_EDGE_POINTS`: El número de puntos a descartar cercanos a los bordes.
    * `FIND_FRINGES_STEP`: El tamaño del paso utilizado al buscar puntos de las franjas.
    * `FIND_FRINGES_APERTURE_IN_SEARCH`: Proporción de la ventana de búsqueda entre franjas para buscar puntos en franjas.
    * `MAX_NUMBER_POINTS_FIT`: El número máximo de puntos a ajustar con una cuadrática para encontrar un extremo.
* Cálculo de desviación máxima de cada franja:
    * `IQR_FACTOR_POINTS_IN_FRINGES`: El factor utilizado para descartar puntos detectados de franjas que son sospechosos.
* General:
    * `REQUIRED_IMS`: El número mínimo de imágenes requeridas para el análisis.
    * `RESULTS_DIR`: El directorio donde se guardan los resultados.
    * `IQR_FACTOR_IMS`: El factor utilizado para descartar puntos en la estimación final de la desviación máxima.    


### Monte Carlo

El archivo `montecarlo_fei.py` contiene funciones para calcular el montecarlo de flechas e interfranjas. A su vez, se puede correr como script para calcular el montecarlo de flechas e interfranjas. Los siguientes parámetros se pueden configurar:

* MULTIPROCESSING: Booleano que indica si se deben usar procesos multiprocesos para calcular el Monte Carlo (default: False)
* SAVE_PATH: Ruta donde se guardan las figuras de resultados.
* LOAD_PATH: Ruta donde se cargan los resultados guardados. Usar "" para no cargar nada.
* N_MC_SAMPLES: Número de muestras Monte Carlo a generar (default: 200)
* N_IMS_PER_SAMPLE: Número de imágenes a procesar con distinto espacio interfranja por muestra Monte Carlo (default: 10)
* MIN_N_FRINGES: Mínimo número de franjas para cada muestra Monte Carlo (default: 7)
* MAX_N_FRINGES: Máximo número de franjas para cada muestra Monte Carlo (default: 20)
* MAX_ROTATION_DEG: Máximo ángulo de rotación simulado en grados (default: 5)
* VISIBILITY_RATIO: Coeficiente de visibilidad de las franjas (default: 0.5)
* NOISE_LEVEL: Nivel de ruido relativo a la visibilidad (default: 0.001)
* WAVELENGTH_NM: Longitud de onda en nanómetros (default: 632.8)
* MAX_DEVIATION_nm: Desviación máxima permitida en nanómetros (default: 150.0)
* PLOT_INTERFEROGRAMS: Booleano que indica si se debe mostrar las imágenes de interferogramas (default: False)
* PLOT_SURFACES: Booleano que indica si se deben mostrar las superficies simuladas (default: False)
* PLOT_FITS: Booleano que indica si se debe mostrar el ajuste que calcula la máxima desviación a partir de los pares (desviación de franja, espacio interfranja) (default: False)
* PLOT_FRINGE_DETECTION: Booleano que indica si se deben mostrar la detección de franjas (default: False)
* PLOT_RESULTS: Booleano que indica si se deben mostrar los resultados finales (default: True)
* SIMULATION_MODE: Modo de simulación de las desviaciones, puede ser "random" o "gaussian" (default: "random")

Se utiliza el objeto `FlatInterferogramGenerator` de `interferogram_generation.py` para generar los interferogramas. El método `generate` de este objeto se utiliza como un iterador, el cual se inicializa solo cada vez que comienza un bucle de *for* y devuelve interferogramas con distintas frecuencias de franjas para la misma superficie en cada iteración. Adicionalmente, calcula la ubicación de las curvas de valles ideales en las franjas. La superficie simulada no contiene componentes de inclinación para no alterar la frecuencia espacial de franjas simulada.

La función que hace el procesamiento de los interferogramas puede ofrecer como salida los espacios interfranja (en píxeles) y las máximas desviaciones con respecto a lineas rectas equiespaciadas como ufloats (con incertidumbre). Por otro lado, puede usar el diccionario `debugging_info` para recibir las ubicaciones de las curvas reales de valles en las franjas, y ofrecer como resultado lo siguiente:

* `rotation_angle_estimated`: ángulo estimado de rotación de las franjas en grados (primera estimación).
* `rotation_angle_estimated_corrected`: ángulo estimado de rotación de las franjas en grados (estimación refinada por optimización).
* `rmsd_to_valley_curves`: distancia RMS a las curvas reales de valles en las franjas.

Los workers (trabajadores en paralelo) obtienen las siguientes salidas:

* `simulated_interfringe_spacings`: Espacios interfranja simulados en píxeles.
* `measured_interfringe_spacings`: Espacios interfranja medidos en píxeles.
* `simulated_deviation_nm`: Desviación simulada en nanómetros.
* `measured_max_deviation`: Máxima desviación estimada en nanómetros.
* `simulated_rotation_angle`: Ángulo de rotación simulado en grados para cada imagen de cada muestra del Monte Carlo.
* `error_rotation_estimation`: Error entre el ángulo de rotación simulado y el ángulo de rotación estimado (primera estimación).
* `error_rotation_estimation_corrected`: Error absoluto entre el ángulo de rotación simulado y el ángulo de rotación estimado (estimación refinada por optimización).
* `mean_rms_distance_to_valley`: Distancia RMS media de los puntos de valles encontrados a las curvas reales de valles en las franjas.

En el caso de querer obtener las figuras de los resultados (`PLOT_RESULTS`), se realizan los siguientes plots:

* Figura 1 (2 x 2):
    * RMSE de espacios interfranja medidos para cada imagen de cada muestra del Monte Carlo. Las primeras imágenes contienen menor cantidad de franjas.
    * Idem anterior pero con valores relativos al espacio interfranja simulado.
    * RMSE de espacios interfranja medidos para cada muestra del Monte Carlo con respecto a la desviación máxima simulada.
    * Boxplot de espacios interfranja medidos y relativos a lo simulado para distintos rangos de rotaciones simuladas.
* Figura 2 (2 x 3):
    * Boxplot de error absoluto de estimación optimizada de rotación para cada cantidad de franjas simuladas.
    * RMSE de estimaciones de rotación con respecto a la desviación máxima simulada.
    * Boxplot de error absoluto de estimación optimizada de rotación para distintos rangos de rotaciones simuladas.
    * Boxplot de distancias RMS a curvas reales de valles para distintos rangos de rotaciones simuladas.
    * RMS de distancias RMSEs a curvas reales de valles con respecto a la desviación máxima simulada.
    * Boxplot de distancias RMS a valles para distintos rangos de rotaciones simuladas.
* Figura 3:
    * Desviación máxima estimada con respecto a la simulada. Se agrega el ajuste lineal.

#### Modificicaciones necesarias para adaptar a otros métodos de análisis al Monte Carlo

En el caso de querer realizar el Monte Carlo para otra función de análisis, probablemente se tengan que realizar plots distintos. La Figura 2 probablemente no tenga sentido en otra función de análisis. Los plots de Figura 1 y 3 pueden ser modificados para que se adapten a la función de análisis.
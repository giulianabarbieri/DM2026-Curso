---
title: No10 - PINNs Pt2
---

# Tips para entrenar PINNs

**Fecha:** 13/05/2026

:::{iframe} https://www.youtube.com/embed/5-SWMQoxbKs
:width: 100%
:::

### Recap de la clase pasada:

PINNs: estamos optimizando sobre dos variables: \(x\) y \(\theta\).

*(Aquí insertamos la **Figura 1**: PINN con \(\theta^*\) y las curvas naranjas de zonas de \(\theta\).
Representa dos coordenadas con una función suave que tiene un conjunto de círculos ovalados que convergen en el punto óptimo \(\theta^*\))*.

Y teníamos una variedad que satisfacía las restricciones (*constraints*):

\[D[x(\cdot, \theta)] = 0\]

Donde lo de adentro de los corchetes podría ser cualquier operador.
En el ejemplo de la ecuación del calor, podríamos tener:

\[\left[ \begin{matrix} \dfrac{dx}{dt} - D \nabla^2 x \\ x(t, \theta) - x_0 \\ \vdots \end{matrix} \right] = 0\]

Donde la primera fila representa el calor, la segunda la condición de contorno, etc.
Podemos seguir agregando componentes si así lo quisiéramos.

Entonces, estamos tratando de encontrar un parámetro \(\theta\) óptimo:

\[\theta^*\]

Pero como la optimización se realiza en más variables, el problema se veía más como una "zona" de parámetros \(\theta\) (ver **Figura 1**: son las formas ovaladas de color naranja que, dependiendo del problema, pueden estar más estiradas o no).

## Desafíos de entrenar una PINN

La clase pasada vimos por qué esto se puede volver muy complejo para dimensiones altas.

### (1) Mal condicionamiento:

Uno de los motivos por los que esto puede fallar es cuando el problema de optimización está mal condicionado (pequeños cambios dan resultados muy distintos).
Las curvas naranjas de la **Figura 1** pueden estar alargadas en una dirección y comprimidas en otra.
En estos casos, al optimizador le cuesta llegar al \(\theta\) óptimo.
Por este motivo, siempre que entrenemos una PINN vamos a usar un algoritmo de gradiente descendiente de al menos primer orden con métodos de tipo pseudo-Newton (que intentan aproximar un método de segundo orden).

### (2) Cómo balancear las distintas componentes de la función de costo

Habíamos visto que la función de costo total se armaba de esta manera:

\[\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{emp}} + \sum_{i=1}^{k} \lambda_i \mathcal{L}_{\text{física}}(\theta, x)\]

La pregunta es: ¿cómo elegimos estos valores de \(\lambda\)?
Si están conectados con los datos, podríamos tantear haciendo prueba y error o (lo cual es más interesante y popular) utilizar *hiperparámetros adaptativos*.
Aquí los \(\lambda\) dependen de la iteración y uno los va actualizando con un criterio específico; por ejemplo, que todos los términos de la función de costo tengan el mismo peso o que los gradientes de los términos de la función de costo estén igualados.

##### Observación:

Existe una herramienta llamada *La curva L*, que consiste en graficar la función de costo física en un eje y la función de costo empírica en el otro.
La curva resultante (véase **Figura 2**) tiene forma de "L".
Se suele tomar el \(\lambda\) que esté más cerca del ángulo o vértice, ya que es el que mejor nos aproxima a nuestro \(\theta\) óptimo:

\[\theta^*\]

*(Aquí insertamos la **Figura 2**: Gráfico de la Curva L)*.

### (3) Puntos de colocación

*(Aquí insertamos la **Figura 3**: Dibujo de la ecuación del calor con bordes, más un dominio en \(\mathbb{R}^n\) con los puntitos y la condición de borde)*.

Decidir cómo vamos a elegir cada uno de estos puntos (triángulos, círculos o cruces en el dominio) es clave.
Nos interesa saber qué pasa en el interior, decidir cuántos puntos tomaremos y si están bien distribuidos.
Esto determina qué tan denso se evalúa la ecuación diferencial; además, los puntos interiores siempre tienen una dimensión mayor que los que están en los bordes, y es allí donde se está imponiendo la restricción de la ecuación diferencial.

Nos interesa conocer esta distribución y qué tan distantes están los puntos entre sí, porque esta distancia controla nuestra precisión.
Estamos discretizando el espacio con la distancia que establezcamos; es decir, no podremos resolver el problema con mayor resolución que la distancia seleccionada.
El compromiso (*trade-off*) radica en que si aumentamos la cantidad de puntos (disminuyendo lo más posible la distancia entre ellos), estamos incrementando el costo computacional.
Cada evaluación en estos puntos requiere diferenciar la red neuronal y todo ese cálculo impacta directamente en la función de costo.

#### Estrategias de muestreo de puntos de colocación

##### (1) Grilla uniforme

Grillamos el dominio y las intersecciones de las líneas son los puntos de colocación.

*(Aquí insertamos la **Figura 4**: El mismo dominio que mostramos en la Figura 3, pero con una grilla.
Nota: No olvidar mencionar el Hipercubo Latino y agregar al pie de la figura que esto es lo que utilizan en el *paper* original).*

##### (2) Muestreo uniforme

Consiste en un muestreo aleatorio que sigue una distribución uniforme dentro del dominio.

##### (3) Muestreo uniforme distinto en cada época (*epoch*)

Es una estrategia muy útil para evitar el sobreajuste (*overfitting*) en puntos específicos.

##### (4) Muestreo por importancia (*Importance sampling*)

En pocas palabras, consiste en realizar un muestreo con la probabilidad ajustada.
Nuestro objetivo ideal no es simplemente seleccionar puntos de colocación individuales, sino que uno querría calcular la integral exacta sobre el dominio \(\Omega\):

\[\int_{\Omega} \mathcal{L}_{\text{física}}(x) \, dx\]

Pero como no la tenemos, lo que hacemos es aproximarla mediante puntos discretos.
Dado que nos interesa minimizar la función de costo física, observamos en qué regiones esta función toma valores más altos y tomamos más muestras allí.

Para lograrlo, calculamos la función de costo física para cada punto candidato.
Luego, muestreamos sujetos a una función de probabilidad que le asignará un peso mayor si esos puntos presentan valores de error elevados:

\[P(x^{\text{col}}_i, k+1) \propto e^{\mathcal{L}_{\text{física}}(x)}\]

Donde el símbolo \(\propto\) se lee como "proporcional a".
Al colocar la función de costo física como exponente de la constante \(e\), le otorgamos una prioridad exponencialmente mayor a las regiones con un error alto.

### (4) Sesgo Espectral (*Spectral Bias*)

Las redes neuronales aprenden muy bien las bajas frecuencias, pero les cuesta mucho más capturar las frecuencias altas.

*(Aquí insertamos la **Figura 5**: Gráfico que ilustra cómo la red neuronal se aproxima secuencialmente a las curvas según sus frecuencias).*

Cuando entrenamos una red, vemos que a lo largo de las épocas (*epochs*) converge primero a "la curva general" (la frecuencia baja) y luego, con un mejor ajuste y más épocas, comienza a capturar las frecuencias altas.
Por lo tanto, si queremos aproximar una función con componentes de alta frecuencia, la variable \(x\) de entrada por sí sola presentará problemas; las redes convencionales son malas aproximando frecuencias altas y, cuando lo intentan, suelen caer en *overfitting*.

**¿Por qué representa un problema?**

La PINN está resolviendo una ecuación diferencial, y las frecuencias de esa solución *tienen un significado físico real*.
No queremos descartar las frecuencias altas si representan un fenómeno físico válido.
Si la red neuronal es incapaz de "ver" nada más allá de cierta frecuencia, nunca podrá aproximar dicha solución, lo cual es crítico principalmente para **ecuaciones en derivadas parciales (EDPs)**.

Otra manera más heurística de pensarlo es la siguiente: en una red neuronal convencional, las operaciones dentro de las capas consisten en una transformación lineal seguida de una función de activación (como la sigmoide) multiplicada por una matriz grande.
Este proceso actúa intrínsecamente como un filtro de suavizado.
De esta forma, a las capas subsiguientes ya les van llegando frecuencias cada vez más bajas.

¿Cómo podríamos solucionar el problema del sesgo espectral para que la red sea capaz de aprender frecuencias más altas?

---

#### Solución 1: Fourier features

Es una solución que se usa mucho e implica una precalibración de la red neuronal.
Consiste en que antes de pasar el input a la red vamos a forzar una primera capa (es decir, le agregamos una capa más, adelante de todo, para que sea la primera en recibir el input).
Esta nueva capa lleva todo al plano espectral.

##### Esquema Matemático de la Capa

El mapeo de la señal de entrada \(x\) a través de las frecuencias de Fourier se expresa formalmente como la transformación \(\gamma(x)\), la cual alimenta a la primera capa oculta de la red neuronal:

\[x \longrightarrow \gamma(x) = \begin{bmatrix} \cos(2\pi w_1 x) \\ \sin(2\pi w_1 x) \\ \cos(2\pi w_2 x) \\ \sin(2\pi w_2 x) \\ \vdots \\ \cos(2\pi w_i x) \\ \sin(2\pi w_i x) \end{bmatrix} \longrightarrow \boxed{\text{Red Neuronal}}\]

##### Operación en la Primera Capa Oculta

Si denotamos a la capa de características de Fourier como la capa \(0\) (o capa de entrada modificada), la preactivación de la siguiente capa de la red se calcula mediante el producto tensorial con la matriz de pesos \(W^{(1)}\) y el vector de sesgo \(b^{(1)}\):

\[z^{(1)} = W^{(1)} \cdot \gamma(x) + b^{(1)}\]

Desarrollando los componentes para cada neurona \(j\) de la capa oculta, la operación con símbolos puros de LaTeX se representa así:

\[z_j^{(1)} = \sum_{i=1}^{N} \left[ w_{j, 2i-1}^{(1)} \cos(2\pi w_i x) + w_{j, 2i}^{(1)} \sin(2\pi w_i x) \right] + b_j^{(1)}\]

Donde:
* **\(x\)**: Es el input escalar original.
* **\(w_i\)**: Son las frecuencias fijas o aprendibles en el plano espectral.
* **\(w_{j, k}^{(1)}\)**: Son los pesos entrenables de la neurona \(j\) que conectan con la característica de Fourier \(k\).
* **\(b_j^{(1)}\)**: Es el sesgo de la neurona \(j\).

#### Solución 2: Otro tipo de arquitectura (Escalamiento de Entrada / Bagging de Frecuencias)

Otro método es aumentar los valores de entrada para mapear y capturar de mejor manera las frecuencias bajas que tenga la señal.

##### Esquema Matemático de la Arquitectura
En este diseño, el input escalar \(x\) se multiplica por factores enteros sucesivos en la primera capa modificada.
Cada una de estas entradas escaladas alimenta a una red neuronal independiente (arquitectura tipo *Ensemble/Bagging*), cuyas salidas parciales se combinan finalmente en una neurona de salida global \(u\):

\[x \longrightarrow \begin{bmatrix} 1x \\ 2x \\ 3x \\ \vdots \\ nx \end{bmatrix} \longrightarrow \begin{aligned} &\boxed{\text{Red}_1(\theta_1)} \longrightarrow u_1 \\ &\boxed{\text{Red}_2(\theta_2)} \longrightarrow u_2 \\ &\boxed{\text{Red}_3(\theta_3)} \longrightarrow u_3 \\ &\ \vdots \\ &\boxed{\text{Red}_n(\theta_n)} \longrightarrow u_n \end{aligned} \longrightarrow \boxed{\sum} \longrightarrow u_{\text{final}}\]

##### Expresión Formal del Modelo

La salida final del sistema \(u(x)\) se define como la combinación (frecuentemente un promedio o una suma ponderada) de \(n\) redes neuronales "vainilla" independientes, donde cada red \(i\) está parametrizada por sus propios pesos \(\theta_i\):

\[u_{\text{final}}(x) = \sum_{i=1}^{n} \text{Red}_i(i \cdot x ; \theta_i)\]

##### Intuición de la Versión Bagging

Si la señal original contiene una frecuencia extremadamente baja o "chiquita" del orden de:

\[f_{\text{baja}} = \frac{1}{n}\]

Una red neuronal estándar (vainilla) tendría severas dificultades para aprenderla debido al sesgo hacia las altas frecuencias.
Al forzar la multiplicación de \(x\) por el factor \(n\) en la \(n\)-ésima neurona de la primera capa, la frecuencia efectiva se transforma:

\[f_{\text{efectiva}} = n \cdot \left(\frac{1}{n}\right) = 1\]

Al convertirla en una frecuencia fundamental (\(1\)), la red asociada a ese bloque ya tiene la capacidad de aprender esa componente de la señal sin problemas.
Finalmente, todas estas capacidades predictivas de diferentes escalas de frecuencia se combinan en la neurona de salida general \(u\).
```{note}
**Nota al margen: Ensembles en Machine Learning**
* **Bagging (Bootstrap Aggregating):** Consiste en entrenar múltiples modelos en paralelo de forma independiente y promediar sus predicciones. La **Solución 2** es de este tipo (paralela).
* **Boosting:** Consiste en un proceso secuencial donde cada nuevo modelo se entrena específicamente para corregir los errores (residuos) de los modelos anteriores. La **Solución 3** aplica este concepto a redes neuronales.
```

#### Solución 3: Multistage Neural Networks (Versión Boosting)

Este enfoque utiliza el concepto de *Boosting* de manera secuencial: se entrena una primera red neuronal y luego se entrenan redes sucesivas de forma iterativa sobre el residuo (el error) que las anteriores no lograron explicar.

##### Esquema de la Arquitectura

A diferencia de la arquitectura en paralelo de la solución anterior, aquí las redes se encadenan secuencialmente en etapas (*multistage*).
Cada red aprende la señal residual de la etapa previa:

\[x \longrightarrow \boxed{\text{Red}^{(1)}} \longrightarrow \text{Residuo}^{(1)} \longrightarrow \boxed{\text{Red}^{(2)}} \longrightarrow \text{Residuo}^{(2)} \longrightarrow \boxed{\text{Red}^{(3)}} \longrightarrow \dots \longrightarrow \boxed{\sum} \longrightarrow \hat{y}\]

##### Expresión Formal del Modelo

La predicción final del modelo compuesto por \(K\) etapas es la suma acumulada de las salidas de cada una de las sub-redes entrenadas individualmente:

\[\text{Red}_{\Theta}(x) = \sum_{i=1}^{K} \text{Red}^{(i)}_{\theta_i}(x)\]

##### Mecanismo de Aprendizaje por Residuos

El entrenamiento se realiza de forma estrictamente secuencial bajo la siguiente lógica:

1. **Etapa 1:** Se entrena la primera red \(\text{Red}^{(1)}\) con los datos originales.
Esta red suele capturar los patrones más fáciles y dominantes de la señal (por ejemplo, las **frecuencias bajas**).
2. **Etapa 2:** Se calcula el primer residuo, es decir, lo que la primera red no pudo aprender:
   \[r^{(1)} = y - \text{Red}^{(1)}(x)\]
   Luego, la segunda red \(\text{Red}^{(2)}\) se entrena utilizando \(r^{(1)}\) como su objetivo (*target*).
   Esta red se ve forzada a capturar detalles más finos (como **frecuencias intermedias o altas**).
3. **Etapa \(i+1\):** De manera general, cada red subsiguiente aprende el residuo acumulado de todas las anteriores:
   \[r^{(i)} = y - \sum_{j=1}^{i} \text{Red}^{(j)}(x)\]

Al finalizar el proceso, la combinación de todas las etapas permite reconstruir tanto la estructura global de los datos como sus detalles de alta frecuencia.



# Laten Diffusion Models Explicación Teórica

## Introducción

Los difusores son modelos de Inteligencia Artificial (IA) que generan nueva información en base a inputs del usuario. Se llaman difusores porque funcionan de manera similar a como lo hacen los difusores en química.

## Diffusion Models

Es una manera de generar imágenes pixel a pixel. Requiere de un gran esfuerzo computacional debido al tamaño de la imágenes.

### Forward Diffusion

Se le añade **ruido** a una imagen **paso a paso**. A cada paso, se le agrega más ruido. La técnica utilizada para agregar ruido se llama [_Ruido Gaussiano_](https://www.sfu.ca/sonic-studio-webdav/handbook/Gaussian_Noise.html). Básicamente, esta técnica añade aleatoriedad a los datos ingresados, siguiendo un esquema de Gauss (Campana).

![Forward Diffusion Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/forward-diffusion.webp")

Esto sirve para **entrenar** una _CNN_ de **segmentación de imágenes** llamada [_U-Net_](https://towardsdatascience.com/understanding-u-net-61276b10f360), la cual debería predecir cuanto ruido se le añadió a una imagen.

![Arquitectura U-Net]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/u-net-architecture.png")

### Reverse Diffusion

Se busca **predecir** cuanto **ruido** se **añadió** a una imagen para posteriormente **sustraerlo** de la misma, para obtener la **imagen sin ruido** alguno. Para lograr esto, se utiliza el predictor de ruido que entrenamos con [Forward Diffusion](###forward-diffusion) para hacer una estimación sobre la cantidad de ruido añadido. Una vez que se sabe esto, es posible sustraer esta cantidad de ruido a la imagen para obtener la imagen original.

![Reverse Diffusion Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/reverse-diffusion.webp")

1. Se genera una imagen con ruido de manera aleatoria.
2. El modelo trata de predecir el ruido de la imagen
3. Se sustrae el ruido predicho por el modelo de la imagen
4. Obtenemos una imagen con menos ruido
5. Se repiten los pasos del 2 al 4 una X cantidad de veces
6. Obtenemos la imagen final

Podemos verlo como una **analogía matemática**. La imagen con ruido seria el numero 36, puesto que conocemos su información, mientras que el ruido agregado seria la incognita "R" y la imagen original seria la incognita "I", ya que no sabemos ninguno de sus valores. Nosotros buscamos descubrir el valor de I. Sabemos que I = 36 - R, ya que la imagen original se le añadió R cantidad de ruido y se obtuvo 36. Para resolver este problema, necesitamos descubrir el valor de la incognita R. Es ahi cuando el predictor de ruido entre en acción. Este modelo nos dice que su predicción de ruido para la imagen con valor 36 es de 15, por lo que ya podemos terminar la ecuación. Entonces, I = 36 - 15, lo que nos da un resultado de I = 21. De esta manera, sabemos que si sabemos cuanto se le añadió a la imagen, podemos regresar a la imagen original.

## Latent Diffusion Models

Sin embargo, _Stable Diffusion_ no funciona de esa manera. Stable Diffusion es un modelo de _difusión latente_. Esto significa que funciona por gracias a la creación de una **imagen latente**. Se utiliza esta técnica puesto que **acelera enormemente los procesos** de inferencia del modelo. Este espacio latente termina siendo una **imagen 48 veces más chica** que la original, acelerando enormemente los tiempos.

### VAE

El VAE o “Variational Autoencoder”, es la herramienta encargada de **transformar** una imagen a un **espacio latente**. Esta red neuronal se divide en dos partes:

- Encoder (Codificador): Le **baja la resolución** a la imagen en el espacio latente.
- Decoder (Decodificador): **Restaura la imagen a su resolución** original antes de pasar por el codificador.

![VAE Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/VAE.webp")
![Encoder & Decoder Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/encoder-decoder.png")

El espacio latente para imágenes con una resolución de salida de 512x512 es de 4x64x64. Por lo tanto, vemos que es mucho **más rápido** aplicar todos los procesos de forward diffusion y reverse diffusion en este espacio.

Es posible usar esta técnica ya que las imágenes reales contienen objetos del mundo real que siguen patrones, haciendo mucho más fácil la comprensión de su información. Por ejemplo, los humanos siempre tenemos una cabeza en nuestra parte superior, tenemos un brazo a cada lado, una mano con cinco dedos al final de cada brazo, etc. Esta teoría se llama “Manifold Hypothesis”.

### Denoising

Una vez que entendemos el espacio latente, podemos entender cómo realmente funciona la creación de imágenes con reverse difusión.

1. Se genera una matriz random en el espacio latente (Ruido latente)
2. El predictor de ruido trata de predecir el ruido de la matriz
3. Se sustrae el ruido predicho de la matriz
4. Se repiten los pasos 2 y 3 por una X cantidad de pasos
5. El decodificador del VAE convierte la matriz resultante a la imagen final

### Conditioning

Ya sabemos cómo se crean las imágenes, pero no como **condicionar la creación** de estas con texto. Ahí es cuando sale a la luz el “conditioning” del modelo.

Lo que buscamos con esto es **condicionar al predictor** de ruido, para que **prediga un ruido X** que luego de sustraerlo, nos dé el resultado deseado.

#### Text Conditioning

¿Cómo condicionamos al predictor de ruido con texto? Bueno, se siguen los siguientes pasos:

1. Se crean varios tokens a partir del texto. A cada token le corresponde un ID numérico especifico. Se utiliza una herramienta llamada “tokenizer”
2. Cada token se convierte en un vector con 768 valores llamado “embedding”
3. Cada embedding es procesado por un “text transformer”
4. Se pasa este resultado al predictor de ruido

![Text Encoder Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/text-encoder.webp")

##### Tokenizer

Es una herramienta que le **asigna un número**, conocido como **ID**, a palabras que **conozca**. Esto significa que solo es capaz de **atribuirle un ID** a una palabra con la que haya sido **entrenado**.

###### Tokenization

Antes de asignar IDs, hay que pasar por el proceso de _tokenization_, el cual es **dividir el texto en unidades llamadas _tokens_**. Existen **diversas maneras** de tokenizar un texto, las cuales varían en resultados y en complejidad computacional.

###### Métodos

###### Sentence Tokenization

Se divide el texto en distintas **sub-oraciones o frases**. Tiene un **mejor rendimiento** para **textos periodísticos, literarios o documentos**.

###### Word Tokenization

Se divide el texto en distintas **palabras o mismo sub-palabras**. Tiene varios métodos.

###### Character

Cada **letra** del texto se **transforma en un token**. Este método puede ser útil para identificar palabras que tengan algún **error ortográfico**, pero a causa de esto, requiere un **gran poder** computacional. Por lo tanto, no suele ser muy usado.

Ejemplo: “Ethereum is a safe investment” → ['E', 't', 'h', 'e', 'r', 'e', 'u', 'm', ' ', 'i', 's', ' ', 'a', ' ', 's', 'a', 'f', 'e', ' ', 'i', 'n', 'v', 'e', 's', 't', 'm', 'e', 'n', 't']

```python
text = "Ethereum is a safe investment"
tokenized_text = list(text)
print(tokenized_text)
# ['E', 't', 'h', 'e', 'r', 'e', 'u', 'm', ' ', 'i', 's', ' ', 'a', ' ', 's', 'a', 'f', 'e', ' ', 'i', 'n', 'v', 'e', 's', 't', 'm', 'e', 'n', 't']
```

###### Word

Se divide el texto en **palabras individuales**. Esta técnica requiere de una **gran cantidad computacional** puesto que implica que el modelo pueda reconocer todas las palabras. Es posible **acotar el rango** de palabras.

Ejemplo: “Ethereum is a safe investment” → ['Ethereum', 'is', 'a', 'safe', 'investment']

```python
text = "Estoy en relajación"
tokenized_text = text.split(" ")
print(tokenized_text)
# ['Ethereum', 'is', 'a', 'safe', 'investment']
```

###### Subword

Aplica **ambas técnicas**, puesto que mantiene a las **palabras normales como un token**, mientras que a las **palabras menos comunes las separa en sub-palabras**. Esto trae los beneficios de los dos métodos anteriores, puesto que es capaz de analizar correctamente **faltas ortográficas y vocabulario inusual** al igual que tener un rango de palabras manejable.

Ejemplo: “Ethereum is a safe investment” → ['ether', '##eum', 'is', 'a', 'safe', 'investment']

El doble ¨#¨ significa que ese token debe unir con el token anterior.

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
tokenizer_output = tokenizer.tokenize("Ethereum is a safe investment")
print(tokenizer_output)
# ['ether', '##eum', 'is', 'a', 'safe', 'investment']

```

Por ejemplo, en el caso de que el modelo no se haya entrenado con la palabra **limpiaparabrisas**, muy probablemente termine **separando** esta palabra en dos, “limpia” y “parabrisas”, asignándole a cada uno su respectivo **token**. Por lo tanto, **una palabra no equivale siempre a un token**.

Stable Diffusion utiliza el **tokenizer de CLIP**, el cual es un modelo de inteligencia artificial de **image to text**, siendo capaz de hacer descripciones de una imagen que se le ingrese como input.

El **límite de tokens** por prompt son de **75**, a menos que se modifique en los parámetros del modelo.

##### Embedding

Es un proceso en el que se busca **representar información no vectorizada** en **vectores** que sean utilizables por modelos de inteligencia artificial.

En este caso, buscamos convertir una **palabra a un vector**. Esto se logra con el mapeo de tales palabras a una forma vectorial, en la cual se almacenan de manera **numérica varios atributos y características**. Debido a que las características de los objetos se miden numéricamente, mientras **mayor similitud y relación en su significado tenga**, **más parecidos serán los valores** de los vectores.

![Embedding Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/embedding.png")

![Word Vectors Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/word-vectors.png")

Podemos ver como los vectores de Rey y de Hombre son parecidos, al igual que Reina y Mujer lo son.

Esta forma de representar información nos posibilita propiedades tan interesantes como la siguiente.

![Vector Comparison Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/vector-comparison.png")

Vemos que si sustraemos el vector de Hombre del de Rey y a este resultado le sumamos el vector de mujer, obtenemos el vector de Reina.

[Embeddings 3D Comparison](projector.tensorflow.org)

##### Text Transformer

Se encarga de procesar y acondicionar todos los inputs al modelo, sean vectores, imágenes, etc.

### Proceso

1. Se genera una **matriz de ruido latente** en base a la **seed**, la cual si no se especifica, es aleatoria, mientras que si se le indica un número, generará una versión específica de la matriz de ruido latente. Si se utiliza la misma seed y prompt, la imagen tenderá a ser muy igual o idéntica.

![Matriz Ruido Latente Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/random-tensor-latent.webp")

2. El **predictor de ruido** ingresa la **matriz de ruido latente** y utiliza como **condición** o guía el resultado del text-conditioning sobre la prompt ingresada. Se obtiene el **ruido predicho** por la U-Net.

![Predicción Ruido Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/predicted-noise.webp")

3. Se **sustrae** o resta el **ruido predicho** sobre la matriz de ruido latente.

![Sustraer Ruido Predicho Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/substracted-noise.webp")

4. Se **repiten los pasos 2 y 3** por una **X cantidad de veces**, llamadas “Steps”.

5. El **decodificador** del VAE convierte la **matriz** del espacio latente a una **imagen normal** con la resolución original.

![Decoder Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/decoder.webp")

#### Ilustración Gráfica

![Proceso 1 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/stable-diffusion-1.png")
![Proceso 2 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/stable-diffusion-2.png")
![Proceso 3 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/stable-diffusion-3.png")
![Proceso 4 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/img/stable-diffusion-4.png")

![Creación 1 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/SD-GIF-1.gif")
![Creación 2 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/SD-GIF-2.gif")
![Creación 3 Ilustración]("https://github.com/LucaFedericoMarty/Desaigner-AI/blob/dev/resource/SD-GIF-3.gif")

## Fuentes

https://www.youtube.com/watch?v=sFztPP9qPRc

https://www.youtube.com/watch?v=KVaJKzr4a8c

https://stable-diffusion-art.com/how-stable-diffusion-work/

https://medium.com/@steinsfu/stable-diffusion-clearly-explained-ed008044e07e

https://saschametzger.com/blog/what-are-tokens-vectors-and-embeddings-how-do-you-create-them

https://huggingface.co/docs/transformers/tokenizer_summary

https://claritynlp.readthedocs.io/en/latest/developer_guide/algorithms/sentence_tokenization.html#:~:text=Overview,corpus%20of%20formal%20English%20text.

https://blog.acolyer.org/2016/04/21/the-amazing-power-of-word-vectors/

https://towardsdatascience.com/understanding-u-net-61276b10f360

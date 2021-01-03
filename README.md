# QLearningFallas

Repo para ir probando in algoritmo de aprendizaje por refuerzo para la materia Fallas 2 de FIUBA

## Instalacion
Requiere (pip3 install):
    gym
    gym[atari]
    numpy
    matplotlib
    pyyaml

Correr con:
    `python3 run.py`

## Problemas
ModuleNotFoundError: No module named 'skbuild': updatear pip
## Proceso
Primero se siguio el tutorial en https://www.youtube.com/playlist?list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7 para entender como funciona q learning, y luego se le hizo "mejoras" pasandolo a clases.

Luego se le agrego un manejo de configs mediante yaml, para centralizar todos los parametros variables.

Una vez se tuvo el modelo funcionando se lo refino agregando el factor epsilon y graficos.

Luego se cambio el entorno al de breakout ram que es un juego de atari. Ni bien se cambio el entorno se encontraron problemas. Primero que nada, los estados a observar eran muchisimos mas, por que lo que se observa es la ram. Ademas, cambia el sistema de puntaje, ya que es enos discreto. Claramente es necesario re evaluar.
Primero porcedimos a permitir que el escenario corra, sin una tabla de q para entenderlo que valores de la ram significaban cada cosa. Entender los 128 bytes podia ser complejo por lo que primero miramos los bytes que cambiaban.
Las acciones:
    0 -> nada
    1 -> comienza el juego
    2 -> derecha
    3 -> izquierda

Se aprovecharon las acciones de nada, y el hecho que la bola no aparece al principio para establecer que bytes SIEMPRE cambian. El byte 90, siempre cambia. El byte 91 cambia cada x iteraciones. esto significa que dichos bytes son una manera de tomar el tiempo de juego. En las primeras 12 iteraciones, cambian multiples bytes, independiente de las acciones que cambian.
Al usar la accion de comenzar juego se pudo determinar que los bytes 99 y 101 llevan la informacion de donde se encuentre la bola. En particular
99 es la posicion en eje x (de ~50 a ~190 izq y derecha) y eje y la 101 (mas alto valor es mas abajo en la pantalla).
El byte 57 son las vidas.
Usando accion de movimiento random, determinamos que los bytes 70 (184 pared izq, 0 pared derecha) y 72 (55 pared izq y 191 pared derecha) son de la posicion de la plataforma.
Ya teniamos la posicion de la bola, la plataforma, las vidas, solo nos falta el puntuaje. Al hacer un punto el byte 77 aumenta en uno y el 84 en 5

Ahora era necesario adaptar el sistema.

Primero se intento de reward el puntaje del juego. El problema fue que si lograba hacer un punto en la primer vida, luego esto recompensaria NO jugar una vez se tenia unos pocos puntos, cosa que ocurrio. Por esto se le agrego penalizacion por la pelota estar perdida, y recompensa por estar cerca de la bola. Pero esto no era suficiente (no aprendia), por lo que seguimos probando con distintos valores de buckets, epsilon, factor de afrendizaje y descuento.

Como no venia aprendiendo, se probo nada mas penalizar la caida de la bola, para ver si aprendia a sobrevivir.

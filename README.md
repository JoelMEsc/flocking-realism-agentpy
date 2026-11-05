# Aumentando el realismo en modelos de AgentPy: Flocking Behavior

Extensión del modelo *Flocking behavior* de la [AgentPy Model Library](https://agentpy.readthedocs.io/en/latest/agentpy_flocking.html) (modelo de *boids* de Reynolds, 1987) para aumentar su realismo, con parámetros calibrados contra parvadas reales de estornino.

TC2008B · Tec de Monterrey

**Integrantes:** Carlos Eduardo Arias Capetillo · Emilio Moreno Pérez · Ilan David Narváez Martínez · Joel Martínez Escalera · Mario Giovanni González López · Oswaldo García Torres

## Contenido

El trabajo completo está en el notebook `flocking_realism.ipynb`:

| Sección | Contenido |
|---|---|
| 1 | Modelo base de AgentPy y respuestas a las preguntas guía |
| 2 | Mejoras candidatas, priorización y mejoras elegidas |
| 3 | Modelo extendido: escala realista (3.0), percepción (3.1), roles (3.2), energía (3.3) y modelo completo (3.4) |
| 4 | Calibración y análisis de sensibilidad |
| 5 | KPIs y registro de datos |
| 6 | Experimentos E1 a E6, con 20 réplicas por configuración |
| 7 | Verificación del código y validación con datos reales |
| 8 | Conclusiones, limitaciones y trabajo futuro |
| 9 | Referencias |

```
flocking-realism-agentpy/
├── flocking_realism.ipynb   # Notebook con el modelo, los experimentos y el análisis
├── flocking_realism.py      # Solo el código del modelo, los KPIs, los experimentos y las pruebas
├── resultados/              # Resúmenes de los experimentos de la sección 6 (CSV)
├── reporte.pdf              # Reporte del trabajo (4 páginas)
├── requirements.txt         # Dependencias con versiones fijas
└── README.md
```

## Requisitos

- Python 3.12 (probado con 3.12.3 en Linux).
- Las dependencias de `requirements.txt`: agentpy, numpy, pandas, matplotlib, scipy y jupyter.

## Instalación

Desde la carpeta del repositorio, crear un entorno virtual e instalar las dependencias.

Linux y macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Ejecución

Con el entorno virtual activado:

```bash
jupyter notebook flocking_realism.ipynb
```

y ejecutar todas las celdas (*Run All*). En VS Code, abrir el notebook y elegir el kernel de `.venv`.

Para ejecutarlo completo sin abrirlo:

```bash
jupyter nbconvert --to notebook --execute flocking_realism.ipynb --output ejecutado.ipynb --ExecutePreprocessor.timeout=3600
```

**Tiempo de ejecución:** ~17 minutos en una computadora de 24 núcleos (las simulaciones con réplicas se reparten entre todos los núcleos disponibles).

**Experimentos guardados:** los experimentos de la sección 6 suman 1760 corridas (~100 minutos). Sus resúmenes están en `resultados/` y el notebook los carga si existen. Para volver a correrlos, cambiar `RECALCULAR = False` a `True` en la celda de infraestructura de la sección 6.

### Solo el código

`flocking_realism.py` contiene el mismo código del notebook (modelo base, modelo extendido por etapas, KPIs, configuraciones de los experimentos y pruebas), sin el análisis ni las figuras. Se genera a partir de las celdas del notebook, que es la fuente.

```bash
python flocking_realism.py                             # una corrida del modelo completo y sus KPIs
python flocking_realism.py --animacion                 # animación con slider (300 pasos) en animacion.html
python flocking_realism.py --animacion 1000            # la misma animación, con 1000 pasos
python flocking_realism.py --pruebas                   # pruebas de verificación del código
python flocking_realism.py --experimentos              # carga los experimentos E1 a E6 y muestra un resumen
python flocking_realism.py --experimentos --recalcular # vuelve a correr los experimentos (~100 min)
```

La animación es la misma de la sección 3.4 del notebook: un slider para recorrer los pasos, con una cámara fija que encuadra todo el recorrido, estelas con la trayectoria reciente de cada ave, el color según el rol y la opacidad según la energía. Se guarda como página web y se abre en el navegador.

También se puede importar:

```python
from flocking_realism import ModeloExtendido, parametros_extendido
salida = ModeloExtendido(parametros_extendido).run()
```

## Reproducibilidad

- Cada corrida usa el parámetro `seed` (definido en `parametros_ref`, sección 3.0). Los experimentos derivan de él las semillas de cada réplica, y todas las configuraciones de un experimento usan las mismas semillas.
- Todo el azar del modelo viene de los generadores del modelo (`model.random` y `model.nprandom`), así que la misma semilla da exactamente el mismo resultado. La sección 7.1 lo verifica.
- Las versiones de las dependencias están fijas en `requirements.txt`.

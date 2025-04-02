import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import numpy as np
import dask.dataframe as dd
import uvicorn

app = FastAPI()

# Definir la ruta del dataset
dataset_path = "dataset_limpio"

# Cargar los datasets y manejar excepciones
try:
    movies_df = pd.read_parquet(os.path.join(dataset_path, "movies_dataset_cleaned.parquet"))
    cast_df = pd.read_parquet(os.path.join(dataset_path, "cast.parquet"))
    crew_df = pd.read_parquet(os.path.join(dataset_path, "crew.parquet"))
except FileNotFoundError:
    raise Exception("Error: algún archivo parquet no fue encontrado.")

@app.get("/")
def read_root():
    return {"message": "API en Render funcionando correctamente"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Cambia el puerto a uno dinámico
    uvicorn.run(app, host="0.0.0.0", port=port)

# Asegurar que las columnas de ID sean tipo string para evitar problemas de join
movies_df["id"] = movies_df["id"].astype(str)
cast_df["id"] = cast_df["id"].astype(str)
crew_df["id"] = crew_df["id"].astype(str)

@app.get("/cantidad_filmaciones_mes/{mes}")
def cantidad_filmaciones_mes(mes: str):
    meses = {
        "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
        "julio": 7, "agosto": 8, "septiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12
    }
    mes_lower = mes.lower()
    if mes_lower not in meses:
        raise HTTPException(status_code=400, detail="Mes inválido")
    count = movies_df[movies_df["release_date"].dt.month == meses[mes_lower]].shape[0]
    return {"mensaje": f"{count} películas fueron estrenadas en el mes de {mes}"}

@app.get("/cantidad_filmaciones_dia/{dia}")
def cantidad_filmaciones_dia(dia: str):
    dias = {"lunes": 0, "martes": 1, "miércoles": 2, "jueves": 3, "viernes": 4, "sábado": 5, "domingo": 6}
    dia_lower = dia.lower()
    if dia_lower not in dias:
        raise HTTPException(status_code=400, detail="Día inválido")
    count = movies_df[movies_df["release_date"].dt.dayofweek == dias[dia_lower]].shape[0]
    return {"mensaje": f"{count} películas fueron estrenadas en los días {dia}"}

@app.get("/score_titulo/{titulo}")
def score_titulo(titulo: str):
    titulo_lower = titulo.lower()
    movie = movies_df[movies_df["title"].str.lower() == titulo_lower]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    return {
        "mensaje": f"La película {titulo} fue estrenada en el año {movie.iloc[0]['release_year']} con un score/popularidad de {movie.iloc[0]['popularity']}"
    }

@app.get("/votos_titulo/{titulo}")
def votos_titulo(titulo: str):
    titulo_lower = titulo.lower()
    movie = movies_df[movies_df["title"].str.lower() == titulo_lower]
    if movie.empty:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    if movie.iloc[0]["vote_count"] < 2000:
        raise HTTPException(status_code=400, detail="La película no cumple con el mínimo de 2000 valoraciones")
    return {
        "mensaje": f"La película {titulo} fue estrenada en el año {movie.iloc[0]['release_year']}. La misma cuenta con un total de {movie.iloc[0]['vote_count']} valoraciones, con un promedio de {movie.iloc[0]['vote_average']}"
    }

@app.get("/get_actor/{nombre_actor}")
def get_actor(nombre_actor: str):
    global cast_df, movies_df  # Asegurar que las variables son accesibles

    nombre_actor_lower = nombre_actor.strip().lower()

    # Crear 'name_lower' si no existe para evitar sobreescribir en cada solicitud
    if "name_lower" not in cast_df.columns:
        cast_df["name_lower"] = cast_df["name"].astype(str).str.strip().str.lower()

    # Filtrar películas donde aparece el actor
    actor_movies = cast_df[cast_df["name_lower"] == nombre_actor_lower].copy()
    
    if actor_movies.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas para el actor {nombre_actor}")

    # Convertir ID a string para evitar errores en el merge
    actor_movies["id"] = actor_movies["id"].astype(str)
    movies_df["id"] = movies_df["id"].astype(str)

    # Merge con movies_df
    df_merge = actor_movies.merge(movies_df, on="id", how="left")

    # Rellenar valores nulos en budget y revenue con 0 y convertir a int
    df_merge["budget"] = df_merge["budget"].fillna(0).astype(int)
    df_merge["revenue"] = df_merge["revenue"].fillna(0).astype(int)

    # Calcular 'return' usando np.where para mejor rendimiento
    df_merge["return"] = np.where(df_merge["budget"] > 0, df_merge["revenue"] / df_merge["budget"], 0.0)
    
    # Convertir a float de Python para evitar errores en FastAPI
    df_merge["return"] = df_merge["return"].astype(float)

    # Depuración (opcional): Ver valores después del cálculo
    print(df_merge[["title", "budget", "revenue", "return"]])

    # Cálculos finales
    retorno_total = float(df_merge["return"].sum())  # Asegurar float nativo
    cantidad_peliculas = len(df_merge)
    promedio_retorno = retorno_total / cantidad_peliculas if cantidad_peliculas > 0 else 0.0

    return {
        "actor": nombre_actor.title(),
        "cantidad_peliculas": cantidad_peliculas,
        "retorno_total": round(retorno_total, 2),
        "promedio_retorno": round(promedio_retorno, 2)
    }


@app.get("/get_director/{nombre_director}")
def get_director(nombre_director: str):
    global crew_df, movies_df  # Asegurar que las variables están definidas

    # Crear columna 'name_lower' si no existe
    if "name_lower" not in crew_df.columns:
        crew_df["name_lower"] = crew_df["name"].astype(str).str.lower().str.strip()

    nombre_director_lower = nombre_director.strip().lower()

    # Filtrar películas dirigidas por el director
    peliculas_dirigidas = crew_df[crew_df["name_lower"] == nombre_director_lower]

    if peliculas_dirigidas.empty:
        raise HTTPException(status_code=404, detail=f"No se encontraron películas dirigidas por {nombre_director}")

    # Ajustar el ID para la unión
    key_merge = "id" if "id" in movies_df.columns else "movie_id"

    # Realizar el merge con movies_df
    df_merge = peliculas_dirigidas.merge(movies_df, on=key_merge, how="left")

    # Rellenar valores nulos en las columnas importantes
    df_merge.fillna({
        "title": "Título desconocido",
        "release_date": "Fecha desconocida",
        "budget": 0,
        "revenue": 0
    }, inplace=True)

    # Calcular retorno (revenue / budget, evitando división por cero)
    df_merge["return"] = df_merge.apply(lambda row: row["revenue"] / row["budget"] if row["budget"] > 0 else 0, axis=1)

    # Construcción de la respuesta
    resultado = [
        {
            "titulo": row["title"],
            "fecha_lanzamiento": row["release_date"],
            "retorno": round(row["return"], 2),
            "costo": int(row["budget"]),
            "ganancia": int(row["revenue"])
        }
        for _, row in df_merge.iterrows()
    ]

    return {
        "director": nombre_director.title(),
        "cantidad_peliculas": len(resultado),
        "peliculas": resultado
    }

# Vectorización del contenido
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies_df["overview"].fillna("")).astype(np.float32)

# Reducir dimensionalidad
svd = TruncatedSVD(n_components=100, random_state=42)
tfidf_reduced = svd.fit_transform(tfidf_matrix)

# Modelo de vecinos más cercanos
nn_model = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
nn_model.fit(tfidf_reduced)

@app.get("/recomendacion/{titulo}")
def recomendacion(titulo: str):
    titulo_lower = titulo.lower()
    if titulo_lower not in movies_df["title"].str.lower().values:
        raise HTTPException(status_code=404, detail="Película no encontrada")
    
    idx = movies_df[movies_df["title"].str.lower() == titulo_lower].index[0]
    distances, indices = nn_model.kneighbors(tfidf_reduced[idx].reshape(1, -1))
    top_indices = indices.flatten()[1:6]  # Obtener las 5 más similares
    recommended_movies = movies_df.iloc[top_indices]["title"].tolist()
    
    return {"peliculas_recomendadas": recommended_movies}







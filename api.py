from flask import Flask, request, jsonify
from flask_cors import CORS
from pyproj import CRS, Transformer
from pulp import *
from shapely.geometry import MultiPoint
from shapely.geometry import Point
import geojson
import pyproj
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union
from shapely.geometry import mapping
from shapely.ops import transform
from functools import partial
from pyproj import Transformer
import numpy as np

trans = Transformer.from_crs( 
    "epsg:4326",
    "+proj=utm +zone=23 +ellps=WGS84",
    always_xy=True,
)

def distance_cal(a, b) -> float:
    """ Função para retornar a distância entre dois pontos.

    Args:
        a (np.array): lista com coordenadas que descreve um ponto.
        b (np.array): lista com coordenadas que descreve um ponto.

    Returns:
        float: Distância entre os dois pontos como parâmetro.


    Exemplos:
    ---------

    >>> distance([1. , 1.], [4.333332, 24])
    23.240290493499085
    """ 
    return abs(np.linalg.norm(a-b))

app = Flask(__name__)
CORS(app)

def calculate_distance(point1, point2):
    # Define a projeção UTM para realizar a conversão de coordenadas
    crs_utm = CRS.from_string('+proj=utm +zone=21 +datum=WGS84')

    # Cria o transformador para realizar a conversão de coordenadas
    transformer = Transformer.from_crs('EPSG:4326', crs_utm, always_xy=True)

    # Converte as coordenadas dos pontos para UTM
    x1, y1 = transformer.transform(point1[0], point1[1])
    x2, y2 = transformer.transform(point2[0], point2[1])

    # Calcula a distância linear em UTM entre os pontos
    distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    return distance

# Definir função para projetar círculos em metros
def create_circle_in_meters(point, radius_in_meters):
    local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"
    wgs84_to_aeqd = partial(pyproj.transform,
                            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
                            pyproj.Proj(local_azimuthal_projection),
    )
    aeqd_to_wgs84 = partial(pyproj.transform,
                            pyproj.Proj(local_azimuthal_projection),
                            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
    )
    point_transformed = transform(wgs84_to_aeqd, point)
    buffer = point_transformed.buffer(radius_in_meters)
    buffer_wgs84 = transform(aeqd_to_wgs84, buffer)
    return buffer_wgs84

@app.route('/medianas', methods=['POST'])
def process_geojson():
    try:
        data = request.get_json()
        
        # Verifica se o objeto recebido é um GeoJSON válido do tipo 'FeatureCollection'
        if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
            raise ValueError('Entrada inválida. O objeto deve ser um GeoJSON do tipo "FeatureCollection".')

        features = data.get('features')

        # Verifica se existem exatamente duas features dentro do FeatureCollection
        if not isinstance(features, list) or len(features) != 2:
            raise ValueError('Entrada inválida. O FeatureCollection deve conter exatamente duas features.')

        # Verifica se as duas features são do tipo MultiPoint
        for feature in features:
            if not isinstance(feature, dict) or feature.get('type') != 'Feature':
                raise ValueError('Entrada inválida. Cada feature deve ser um objeto GeoJSON do tipo "Feature".')
            if feature.get('geometry', {}).get('type') != 'MultiPoint':
                raise ValueError('Entrada inválida. As features devem ser do tipo "MultiPoint".')

        # Obtém os pontos das duas features
        points1 = features[0]['geometry']['coordinates'] 
        points2 = features[1]['geometry']['coordinates'] 

        # Calcula a matriz de distâncias em UTM
        distances = []
        for point1 in points1:
            row = []
            for point2 in points2:
                distance = calculate_distance(point1, point2)
                row.append(distance)
            distances.append(row)

        # Constrói o objeto de resposta
        response = {
            'type': 'distances',
            'distances': distances
        }

        facilidades = points2
        pontos_atendimento = points1

        # Suponha que a quantidade de facilidades disponíveis seja uma variável 'p'.
        p = features[1]['properties']["p"]

        # Crie o problema de minimização.
        prob = LpProblem("P_Median_Problem", LpMinimize)

        # Crie as variáveis de decisão.
        x = LpVariable.dicts("X", [(i, j) for i in range(len(pontos_atendimento)) for j in range(len(facilidades))], 0, 1, LpBinary)
        y = LpVariable.dicts("Y", [j for j in range(len(facilidades))], 0, 1, LpBinary)

        # Obtém os pesos da feature
        pesos = features[0]['properties']['novoCampo']

        # Adicione a função objetivo.
        prob += lpSum([lpSum([x[(i, j)] * distances[i][j] * pesos[i]['pop'] for j in range(len(facilidades))]) for i in range(len(pontos_atendimento))])

        # Adicione a restrição de que cada ponto deve ser atribuído a exatamente uma facilidade.
        for i in range(len(pontos_atendimento)):
            prob += lpSum([x[(i, j)] for j in range(len(facilidades))]) == 1

        # Adicione a restrição de que o número de facilidades abertas deve ser exatamente 'p'.
        prob += lpSum([y[j] for j in range(len(facilidades))]) == p

        # Adicione a restrição de que se um ponto é atribuído a uma facilidade, a facilidade deve estar aberta.
        for i in range(len(pontos_atendimento)):
            for j in range(len(facilidades)):
                prob += x[(i, j)] <= y[j]

        # Solucione o problema.
        prob.solve()

        # Imprima o status da solução.
        print("Status:", LpStatus[prob.status])

        linhas = []

        # Percorremos todas as variáveis do problema
        for v in prob.variables():
            # Consideramos apenas as variáveis que representam uma ligação entre uma facilidade e um ponto de atendimento
            if v.name.startswith('X') and v.varValue == 1.0:
                # Extraímos os índices das facilidades e dos pontos de atendimento a partir do nome da variável
               
                _, i, j = v.name.split("_")
                i =i.replace("(","")
                i =i.replace(")","")
                i =i.replace(",","")
                j =j.replace("(","")
                j =j.replace(",","")
                j =j.replace(")","")
                i = int(i)
                j = int(j)
                
                # Criamos uma linha entre a facilidade e o ponto de atendimento
                linha = geojson.LineString((facilidades[j], pontos_atendimento[i]))
                
                # Adicionamos a linha à lista
                linhas.append(linha)

        # Criamos a MultiLineString com todas as linhas
        multilinestring = geojson.MultiLineString(linhas)

        # Criamos a Feature com a MultiLineString
        feature = geojson.Feature(geometry=multilinestring)

        # Imprimimos a Feature
        geojson_feature = geojson.dumps(feature, indent=2)
        print(geojson_feature)

        FeatureCollection = {
            "type": "FeatureCollection",
            "features": [geojson_feature, geojson_feature]
        }

        return FeatureCollection

    except ValueError as e:
        print(str(e))
        return str(e), 400

@app.route('/maxcover', methods=['POST'])
def process_mclp():
    try:
        data = request.get_json()

        # Verifica se o objeto recebido é um GeoJSON válido do tipo 'FeatureCollection'
        if not isinstance(data, dict) or data.get('type') != 'FeatureCollection':
            raise ValueError('Entrada inválida. O objeto deve ser um GeoJSON do tipo "FeatureCollection".')

        features = data.get('features')
        
        # Verifica se existem exatamente duas features dentro do FeatureCollection
        if not isinstance(features, list) or len(features) != 2:
            raise ValueError('Entrada inválida. O FeatureCollection deve conter exatamente duas features.')

        # Verifica se as duas features são do tipo MultiPoint
        for feature in features:
            if not isinstance(feature, dict) or feature.get('type') != 'Feature':
                raise ValueError('Entrada inválida. Cada feature deve ser um objeto GeoJSON do tipo "Feature".')
            if feature.get('geometry', {}).get('type') != 'MultiPoint':
                raise ValueError('Entrada inválida. As features devem ser do tipo "MultiPoint".')

        # Obtém os pontos das duas features
        points1 = features[0]['geometry']['coordinates'] 
        points2 = features[1]['geometry']['coordinates'] 
        pesos = features[0]['properties']['novoCampo']
        S = 2000
        I = list(range(0, len(points1)))
        J = list(range(0, len(points2)))
        P = features[1]['properties']['p']
        distances = []
        for point1 in points1:
            row = []

            for point2 in points2:
                distance = calculate_distance(point1, point2)
                row.append(distance)

            distances.append(row)
        

        N = [[j for j in J if distances[i][j] < S] for i in I]

        prob = LpProblem("MCLP", LpMaximize)
        x = LpVariable.dicts("x", J, 0)
        y = LpVariable.dicts("y", I, 0)

        # Objective
        prob += lpSum([y[i]*pesos[i]['impacto']*pesos[i]['pop'] for i in I])

        # Constraints
        for i in I:
            prob += lpSum([x[j] for j in N[i]]) >= y[i]

        for j in J:
            prob += x[j] <= 1
            prob += x[j] >= 0

        for i in I:
            prob += y[i] <= 1
            prob += y[i] >= 0

        prob += lpSum([x[j] for j in J]) == P


        # Solve problem
        prob.solve()

        x_soln = np.array([x[j].varValue for j in J])
        print(x_soln)

        # Criando uma lista de círculos
        circles = []
        # Resultado
        for i in range(len(x_soln)):
            print(x[i].varValue)
            if x_soln[i] > 0.5:
                print((f"Facilidade alocada na localização {points2[i]}"))
                point = Point(points2[i][0], points2[i][1])
                circle = create_circle_in_meters(point, 2000)  # Cria um círculo de raio 2000 metros
                circles.append(circle)

        # Unindo todos os círculos em um MultiPolygon
        multi_polygon = unary_union(circles)
        feature = geojson.Feature(geometry=mapping(multi_polygon))
        FeatureCollection = {
            "type": "FeatureCollection",
            "features": [feature, feature]
        }
        return FeatureCollection, 200

    except ValueError as e:
        return str(e), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000)

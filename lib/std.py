import requests

def find_food_places(manager, radius, keyword, location):
    google_api_key = manager.get_key("PLACES_API_KEY")
    lat, lng = get_lat_lng(google_api_key, address=location)
    location = f"{lat},{lng}"  # Formato "latitud,longitud"
    endpoint = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "key": google_api_key,
        "location": location,
        "radius": radius,
        "keyword": keyword
    }

    response = requests.get(endpoint, params=params)

    response = response.json()

    food_places = api_formatting(response)

    results_list = []

    # Obtener detalles de los lugares
    for place in food_places:
        try:
            details = get_place_details(google_api_key, place["place_id"])
            results_list.append(details)
        except Exception as e:
            continue

    results = api_formatting_2(results_list)

    return {"search_results": results}

def get_lat_lng(api_key, address):
    """
    Obtiene latitud y longitud a partir de una dirección usando la API de Google Maps Geocoding.
    """
    endpoint = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {
        "key": api_key,
        "address": address
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'OK':
            # Extrae latitud y longitud del primer resultado
            location = data['results'][0]['geometry']['location']
            return location['lat'], location['lng']
        else:
            raise Exception(f"Error de Geocoding API: {data['status']}")
    else:
        raise Exception(f"La solicitud a la API falló con el código de estado {response.status_code}")
    
def get_place_details(api_key, place_id):
    """
    Obtiene información detallada sobre un lugar específico usando su Place ID.
    """
    endpoint = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "key": api_key,
        "place_id": place_id,
        "fields": "name,rating,price_level,formatted_address,opening_hours,website,formatted_phone_number,user_ratings_total,editorial_summary,reviews"
    }

    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"La solicitud a la API falló con el código de estado {response.status_code}")
    
def api_formatting(data):
    """
    Filtra y organiza los resultados según estén dentro o fuera del radio solicitado.
    """
    parsed_data = []

    for result in data.get('results', []):
        # Arma la información básica de cada lugar
        place_info = {
            "name": result.get("name"),
            "place_id": result.get("place_id"),
            "rating": result.get("rating"),
            "price_level": result.get("price_level"),
            "formatted_address": result.get("vicinity"),
            "opening_hours": result.get("opening_hours", {}).get("open_now", "N/A"),
            "website": result.get("website", "No Disponible"),
            "formatted_phone_number": result.get("formatted_phone_number", "No Disponible")
        }

        # Mantén la estructura original en parsed_data
        parsed_data.append(place_info)

    # Devuelve tanto los datos clasificados como la estructura original
    return parsed_data

def api_formatting_2(results, max_results = 5):
    formatted_results = []

    for result in results:
        # Extraer campos relevantes
        data = result.get('result', {})
        formatted_result = {
            'name': data.get('name'),
            'formatted_address': data.get('formatted_address'),
            'rating': data.get('rating', 0),
            'user_ratings_total': data.get('user_ratings_total', 0),  # Por defecto 0 si no está disponible
            'price_level': data.get('price_level', 'N/A'),  # 'N/A' si no está disponible
            'opening_hours': data.get('opening_hours', {}).get('weekday_text', []),
            'open_now': data.get('opening_hours', {}).get('open_now', 'N/A'),
            'phone_number': data.get('formatted_phone_number', 'N/A'),
            'website': data.get('website', 'N/A'),
            'editorial_summary': data.get('editorial_summary', 'N/A'),
            'reviews': data.get('reviews', [])[:3]
        }

        # Añadir cada resultado formateado a la lista
        formatted_results.append(formatted_result)
        
        if len(formatted_results) >= max_results:
          break

    return formatted_results
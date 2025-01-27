



import mercadopago
from datetime import datetime, timedelta, timezone
import os

def read_google_doc(manager, messages):
    import requests

    api_key = manager.get_key("GOOGLE_DRIVE_API")
    doc_id = manager.get_key("GOOGLE_DOC_ID")
    
    print(api_key)
    print(doc_id)

    # Google Drive API endpoint for exporting a Google Doc as plain text
    url = f"https://www.googleapis.com/drive/v3/files/{doc_id}/export"
    
    # Parameters for the request
    params = {
        "mimeType": "text/plain",  # Export as plain text
        "key": api_key             # API key for authentication
    }
    
    # Make the GET request
    response = requests.get(url, params=params)
    
    # Check if the request was successful
    if response.status_code == 200:
        content = response.content.decode('utf-8-sig')
        return {"doc_text": content}  # Return the plain text content
    else:
        return {"doc_text": f"Error: {response.status_code}, {response.text}"}

def weather_query(manager, location: str, unit: str = "metric") -> tuple:
    import requests

    API_KEY = manager.get_key("WEATHER_API_KEY")
    BASE_URL = "http://api.weatherapi.com/v1/current.json"

    if unit == "metric":
        temp_unit = "C"
    elif unit == "imperial":
        temp_unit = "F"
    else:
        raise ValueError("Invalid unit. Use 'metric' or 'imperial'.")

    params = {
        "key": API_KEY,
        "q": location,
        "aqi": "no"
    }

    try:
        response = requests.get(BASE_URL, params=params)
        response.raise_for_status()
        data = response.json()

        temperature = data["current"]["temp_" + temp_unit.lower()]
        condition = data["current"]["condition"]["text"]
        wind_speed = data["current"]["wind_kph" if unit == "metric" else "wind_mph"]
        location_name = f"{data['location']['name']}, {data['location']['country']}"

        return (temperature, condition, wind_speed, location_name)

    except requests.RequestException as e:
        return (None, "Unavailable", None, location)

def markdown_to_pdf(messages):
    import markdown
    import pdfkit
    import os

    md_text = messages[-1]["message"]["markdown"]

    title = messages[-1]["message"]["title"]

    # Convert markdown to HTML
    html_content = markdown.markdown(md_text)

    # Wrap the HTML content with proper encoding meta tag
    html_with_meta = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Document</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Path to wkhtmltopdf executable
    path_to_kit = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    config = pdfkit.configuration(wkhtmltopdf=path_to_kit)

    # Determine the output file name
    base_name = title
    extension = ".pdf"
    i = 0
    output_file = f"{base_name}{extension}"

    while os.path.exists(output_file):
        i += 1
        output_file = f"{base_name}-{i}{extension}"


    # Save the HTML as a PDF file
    pdfkit.from_string(html_with_meta, output_file, configuration=config)
    
    return {"pdf_file_path": output_file}

def create_payment_url(manager, name, price, currency, qty):
    import mercadopago
    from datetime import datetime, timedelta, timezone

    ACCESS_TOKEN = manager.get_key("MERCADOPAGO_ACCESS_TOKEN")
    sdk = mercadopago.SDK(ACCESS_TOKEN)

    expiration_date = datetime.now(timezone.utc) + timedelta(days=1)
    expiration_date_iso = expiration_date.strftime("%Y-%m-%dT%H:%M:%SZ")

    preference_data = {
        "items": [
            {
                "title": name,
                "quantity": price,
                "currency_id": currency,
                "unit_price": qty
            }
        ],
        "expiration_date_to": expiration_date_iso
    }

    preference = sdk.preference().create(preference_data)

    url = preference["response"]["init_point"]
    return {"payment_url": url}

def get_video_transcript(video_id):
    from youtube_transcript_api import YouTubeTranscriptApi

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        # Combine all transcript parts into a single string
        full_transcript = ' '.join([entry['text'] for entry in transcript])
        return full_transcript
    except Exception as e:
        return f"Could not retrieve transcript."

def youtube_search(manager, query, max_results=3):
    from googleapiclient.discovery import build

    YOUTUBE_API_SERVICE_NAME = 'youtube'
    YOUTUBE_API_VERSION = 'v3'
    API_KEY = manager.get_key('YOUTUBE_API_KEY')

    youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=API_KEY)

    # Perform the search
    search_response = youtube.search().list(
        q=query,
        part='id,snippet',
        type='video',
        maxResults=max_results
    ).execute()

    results = []
    for item in search_response['items']:
        transcript = get_video_transcript(item['id']['videoId'])

        video_data = {
            'title': item['snippet']['title'],
            'description': item['snippet']['description'],
            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
            'transcript': transcript
        }
        results.append(video_data)

    results = {"video_list": results}
    return results

def get_current_date(messages):
    from datetime import datetime, timedelta, timezone

    return {"current_date": str(datetime.now(timezone.utc))}

def get_calendar(manager, start_date: str, end_date: str, calendar_id: str):
    from datetime import datetime, timezone
    import pytz
    from googleapiclient.discovery import build

    # Convertir fechas de string a objetos timezone-aware
    timezone = pytz.UTC
    start_datetime = timezone.localize(datetime.strptime(start_date, "%Y-%m-%d"))
    end_datetime = timezone.localize(datetime.strptime(end_date, "%Y-%m-%d"))

    # Crear un servicio para acceder a la API de Google Calendar
    API_KEY = manager.get_key('CALENDAR_API_KEY')
    service = build('calendar', 'v3', developerKey=API_KEY)

    try:
        # Llamar a la API para obtener eventos
        events_result = service.events().list(
            calendarId=calendar_id,
            timeMin=start_datetime.isoformat(),
            timeMax=end_datetime.isoformat(),
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        # Obtener eventos
        events = events_result.get('items', [])

        # Crear la lista de eventos en el formato solicitado
        event_list = []
        for event in events:
            event_data = {
                "summary": event.get("summary", "Sin título"),
                "description": event.get("description", "No disponible"),
                "start": event['start'].get('dateTime', event['start'].get('date')),
                "end": event['end'].get('dateTime', event['end'].get('date')),
                "location": event.get("location", "No especificada")
            }
            event_list.append(event_data)

        return {"event_list": event_list}

    except Exception as e:
        print(f"Error al obtener eventos: {e}")
        return {"error": str(e)}
    
def find_food_places(manager, radius, keyword, location):
    import requests

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
    import requests

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
    import requests
    
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
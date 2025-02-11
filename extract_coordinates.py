import re

def extract_coordinates(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = f.read()

        longitude_pattern = re.compile(r"<longitude>(.*?)</longitude>", re.DOTALL)
        latitude_pattern = re.compile(r"<latitude>(.*?)</latitude>", re.DOTALL)

        longitudes = [lon.strip() for lon in longitude_pattern.findall(data)]
        latitudes = [lat.strip() for lat in latitude_pattern.findall(data)]

        coordinates = list(zip(latitudes, longitudes))


        return coordinates
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    file_path = 'coordinates_of_archaeological_sites.kml'
    coordinates = extract_coordinates(file_path)
    with open('coordinates.txt', 'w', encoding='utf-8') as f:
        for lon, lat in coordinates:
            f.write(f"{lon}, {lat}\n")

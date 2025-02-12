import time
import webbrowser
import pyautogui
import pyperclip


def coordinate_to_clipboard(latitude, longitudes):
    coordinate = f'{latitude} {longitudes}'
    pyperclip.copy(coordinate)


def get_coordinates_list():

    with open('coordinates.txt', 'r') as f:
        coordinates_list = [
            (float(x), float(y))
            for line in f if (parts := line.strip().split(',')) and len(parts) == 2
            if (x := parts[0].strip()) and (y := parts[1].strip())
        ]

    return coordinates_list


def generate_google_earth_url(lat, lon, altitude=754.10716168, tilt=699.046938, heading=35, roll=0, twist=0, rotation=90):
    """   
    lat: Latitude of the location
    lon: Longitude of the location
    altitude: Altitude in meters
    tilt: Tilt angle in degrees
    heading: Heading direction in degrees
    roll: Roll angle
    twist: Twist angle
    rotation: Rotation angle
    """
    url = (f'https://earth.google.com/web/search/{lat},{lon}/'
           f'@{lat},{lon},{altitude}a,{tilt}d,{heading}y,{roll}h,{twist}t,{rotation}r')
    return url




coordinates_list = get_coordinates_list()
for coordinate in coordinates_list:
    url = generate_google_earth_url(coordinate[0], coordinate[1])
    webbrowser.open(url, new=0)
    time.sleep(5)
    capture = pyautogui.screenshot(f'screenshot_{coordinate[0]}_{coordinate[1]}.png', region=(0,0, 1920, 1080))

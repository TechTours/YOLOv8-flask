import os
import exifread

def extract_location_info(image_path):
    with open(image_path, 'rb') as file:
        tags = exifread.process_file(file)
        
        if 'GPS GPSLongitude' in tags and 'GPS GPSLatitude' in tags:
            latitude = tags['GPS GPSLatitude'].values
            longitude = tags['GPS GPSLongitude'].values

            latitude = float(latitude[0].num) / float(latitude[0].den)
            longitude = float(longitude[0].num) / float(longitude[0].den)

            if tags['GPS GPSLatitudeRef'].values == 'S':
                latitude = -latitude
            if tags['GPS GPSLongitudeRef'].values == 'W':
                longitude = -longitude

            return latitude, longitude
        else:
            return None

image_folder = 'static/images'

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg'):
        image_path = os.path.join(image_folder, filename)
        location = extract_location_info(image_path)
        
        if location:
            latitude, longitude = location
            print(f'Image: {filename}, Latitude: {latitude}, Longitude: {longitude}')
        else:
            print(f'Image: {filename}, No GPS information found.')

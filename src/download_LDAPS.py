import requests
import os

# URL과 저장 경로 변수를 지정합니다.
URL = 'https://apihub.kma.go.kr/api/typ06/url/nwp_vars_down.php'
API_KEY = 'aWryczNzTHaq8nMzcyx2aQ' #os.environ.get('KMA_API_KEY')
tmfc_utc = '2020010100'
data_type = 'GRIB'
file_name = f"LDAPS_{tmfc_utc}.gb2"

params = {
    'authKey': API_KEY,
    'nwp': 'l015',
    'sub': 'pres',
    'pres_levels': '975',
    'vars':'dzdt',
    'tmfc': tmfc_utc,
    'ef': 24,
    'dataType': data_type,
    'lat': '35.73088463',
    'lon': '129.3672852'
}

with requests.Session() as session:
    response = session.get(URL, params=params)

    if response.status_code != 200:
        raise requests.RequestException(
            response.text
        )
    else:
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print("done.")



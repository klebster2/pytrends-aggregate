import gspread
from oauth2client.service_account import ServiceAccountCredentials
from pathlib import Path

class GoogleCloudCredentialManager:
    GOOGLE_CLOUD_CLIENT_SCOPE = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/drive.file'
    ]

class GoogleSheetsCredentialManager(GoogleCloudCredentialManager):
    def __init__(self, google_key_path:Path):
        self._google_key_path = google_key_path

    def resolve_creds(self):
        return ServiceAccountCredentials.from_json_keyfile_name(
            self._google_key_path.resolve(),
            self.GOOGLE_CLOUD_CLIENT_SCOPE,
        )


class GoogleSheetsManager:
    def __init__(self, sheets_name:str):
        self._sheets_name = sheets_name

    @property
    def sheets_name(self):
        return self._sheets_name

    @sheets_name.setter
    def sheets_name(self, value):
        self._sheets_name = value

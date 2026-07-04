import os
import io
from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback

def get_drive_service():
    """Autentica al usuario y retorna el servicio de Google Drive."""
    auth.authenticate_user()
    return build('drive', 'v3')

def save_model_to_drive(model_obj: tf.keras.Model, filename: str, folder_id: str, drive_service) -> str:
    """Guarda un modelo de Keras localmente y lo sube a Google Drive."""
    local_path = f"/content/{filename}"
    model_obj.save(local_path)
    print(f"Modelo guardado localmente en {local_path}")

    file_metadata = {'name': filename, 'parents': [folder_id]}
    media = MediaFileUpload(local_path, mimetype='application/octet-stream')

    print("Subiendo a Google Drive...")
    file = drive_service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id'
    ).execute()
    
    file_id = file.get('id')
    print(f"Modelo subido con ID: {file_id}")
    return file_id

def load_model_from_drive(filename: str, folder_id: str, drive_service, custom_objects: dict = None):
    """Descarga un modelo desde Google Drive y lo carga en Keras."""
    print(f"Buscando '{filename}' en Drive...")
    
    query = f"name = '{filename}' and '{folder_id}' in parents and trashed = false"
    results = drive_service.files().list(q=query, fields="files(id)").execute()
    items = results.get('files', [])

    if not items:
        raise FileNotFoundError(f"No se encontró '{filename}' en la carpeta {folder_id}.")

    file_id = items[0]['id']
    local_path = f"/content/downloaded_{filename}"

    # Descarga
    request = drive_service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, 'wb') as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"Descarga {int(status.progress() * 100)}%.", end='\r')
    
    print("\nDescarga completa. Cargando modelo...")
    
    # Carga en memoria
    return load_model(local_path, custom_objects=custom_objects)

class DriveModelCheckpoint(Callback):
    """Guarda el modelo en Drive solo si la métrica monitoreada mejora."""
    def __init__(self, drive_service, folder_id: str, filename: str = 'best_model.keras', monitor: str = 'val_loss', verbose: int = 1):
        super().__init__()
        self.drive_service = drive_service
        self.folder_id = folder_id
        self.filename = filename
        self.monitor = monitor
        self.verbose = verbose
        self.best = float('inf')
        self.file_id = self._get_existing_file_id()

    def _get_existing_file_id(self):
        query = f"name = '{self.filename}' and '{self.folder_id}' in parents and trashed = false"
        results = self.drive_service.files().list(q=query, fields="files(id)").execute()
        items = results.get('files', [])
        return items[0]['id'] if items else None

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if current is None:
            return

        if current < self.best:
            if self.verbose > 0:
                print(f"\nÉpoca {epoch+1}: {self.monitor} mejoró de {self.best:.4f} a {current:.4f}. Actualizando Drive...")
            self.best = current
            self._save_and_overwrite()

    def _save_and_overwrite(self):
        local_path = f"/content/{self.filename}"
        self.model.save(local_path)

        media = MediaFileUpload(local_path, mimetype='application/octet-stream')
        try:
            if self.file_id:
                self.drive_service.files().update(fileId=self.file_id, media_body=media).execute()
            else:
                file_metadata = {'name': self.filename, 'parents': [self.folder_id]}
                file = self.drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
                self.file_id = file.get('id')
        except Exception as e:
            print(f"Error subiendo a Drive: {e}")

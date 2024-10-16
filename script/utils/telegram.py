import requests


class TelegramNotifier:
    def __init__(self, bot_token):
        self.bot_token = bot_token
        self.base_url = f"https://api.telegram.org/bot{self.bot_token}/"

    def send_text_message(self, chat_id, text):
        """
        Invia un messaggio di testo a una chat specifica.

        :param chat_id: ID della chat a cui inviare il messaggio
        :param text: Testo del messaggio da inviare
        :return: True se l'invio ha avuto successo, False altrimenti
        """
        endpoint = f"{self.base_url}sendMessage"
        params = {"chat_id": chat_id, "text": text}
        response = requests.get(endpoint, params=params)
        return response.status_code == 200

    def send_image_with_caption(self, chat_id, image_url, caption):
        """
        Invia un'immagine con una descrizione a una chat specifica.

        :param chat_id: ID della chat a cui inviare l'immagine
        :param image_url: URL dell'immagine da inviare
        :param caption: Descrizione dell'immagine
        :return: True se l'invio ha avuto successo, False altrimenti
        """
        endpoint = f"{self.base_url}sendPhoto"
        params = {"chat_id": chat_id, "photo": image_url, "caption": caption}
        response = requests.get(endpoint, params=params)
        return response.status_code == 200


# Esempio di utilizzo:
# notifier = TelegramNotifier("YOUR_BOT_TOKEN")
# notifier.send_text_message("CHAT_ID", "Ciao! Questo è un messaggio di test.")
# notifier.send_image_with_caption("CHAT_ID", "https://example.com/image.jpg", "Questa è un'immagine di esempio.")

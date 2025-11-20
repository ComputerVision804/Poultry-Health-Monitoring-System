# alerts.py
# Requirements: twilio (optional)
import smtplib
from email.mime.text import MIMEText
from typing import Tuple

# Email via SMTP
SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = "youremail@gmail.com"       # TODO: set
SMTP_PASSWORD = "your_app_password"     # TODO: set (use app password)

def send_email_alert(to: str, subject: str, body: str) -> Tuple[bool, str]:
    try:
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = to
        s = smtplib.SMTP(SMTP_HOST, SMTP_PORT)
        s.starttls()
        s.login(SMTP_USER, SMTP_PASSWORD)
        s.sendmail(SMTP_USER, [to], msg.as_string())
        s.quit()
        return True, "sent"
    except Exception as e:
        return False, str(e)

# Twilio SMS (optional)
try:
    import importlib
    _twilio_rest = importlib.import_module("twilio.rest")
    Client = getattr(_twilio_rest, "Client")
    TWILIO_SID = "TWILIO_SID"       # TODO
    TWILIO_TOKEN = "TWILIO_TOKEN"   # TODO
    TWILIO_FROM = "+1234567890"     # TODO (your Twilio number)
    _twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)
    _HAS_TW = True
except Exception:
    _HAS_TW = False
    _twilio_client = None

def send_sms_alert(to: str, message: str) -> Tuple[bool, str]:
    if not _HAS_TW:
        return False, "twilio not configured"
    try:
        m = _twilio_client.messages.create(body=message, from_=TWILIO_FROM, to=to)
        return True, m.sid
    except Exception as e:
        return False, str(e)

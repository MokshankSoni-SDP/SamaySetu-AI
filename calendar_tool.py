import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import pytz
from dotenv import load_dotenv

load_dotenv()

SERVICE_ACCOUNT_FILE = 'service_account.json' 
CALENDAR_ID = os.getenv('CALENDER_ID')

# UPDATED: Changed from .readonly to allow event creation
SCOPES = ['https://www.googleapis.com/auth/calendar']
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('calendar', 'v3', credentials=creds)

IST = pytz.timezone('Asia/Kolkata')

def check_calendar_availability(start_time_str: str):
    """Checks if a 30-minute slot is free in Google Calendar."""

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    end_dt = start_dt + datetime.timedelta(minutes=30)
    
    body = {
        "timeMin": start_dt.isoformat() ,
        "timeMax": end_dt.isoformat() ,
        "items": [{"id": CALENDAR_ID}]
    }
    
    query = service.freebusy().query(body=body).execute()
    busy_slots = query['calendars'][CALENDAR_ID]['busy']
    
    if not busy_slots:
        return f"Slot {start_time_str} is FREE."
    else:
        return f"Slot {start_time_str} is BUSY."

def book_appointment(start_time_str: str, summary: str = "AI Appointment"):
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    
    end_dt = naive_dt + datetime.timedelta(minutes=30)

    event = {
        'summary': summary,
        'description': 'Booked via SamaySetu Gujarati AI Bot',
        'start': {
            'dateTime': naive_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            'timeZone': 'Asia/Kolkata',
        },
        'end': {
            'dateTime': end_dt.strftime('%Y-%m-%dT%H:%M:%S'),
            'timeZone': 'Asia/Kolkata',
        },
    }

    created_event = service.events().insert(
        calendarId=CALENDAR_ID,
        body=event
    ).execute()

    return f"SUCCESS: Appointment booked."
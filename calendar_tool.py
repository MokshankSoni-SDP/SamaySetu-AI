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

def find_event_id(start_time_str: str):
    """Helper to find an event ID based on a start time."""
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    # Search window: 1 minute before/after to be safe
    time_min = (start_dt - datetime.timedelta(minutes=1)).isoformat()
    time_max = (start_dt + datetime.timedelta(minutes=1)).isoformat()

    events_result = service.events().list(
        calendarId=CALENDAR_ID,
        timeMin=time_min,
        timeMax=time_max,
        singleEvents=True
    ).execute()
    
    events = events_result.get('items', [])
    return events[0]['id'] if events else None

def cancel_appointment(start_time_str: str):
    """Deletes an appointment at the given start time."""
    event_id = find_event_id(start_time_str)
    if not event_id:
        return f"Error: No appointment found at {start_time_str} to cancel."
    
    service.events().delete(calendarId=CALENDAR_ID, eventId=event_id).execute()
    return f"SUCCESS: Appointment at {start_time_str} has been cancelled."

def reschedule_appointment(old_start_time_str: str, new_start_time_str: str):
    """Moves an appointment from an old time to a new time."""
    event_id = find_event_id(old_start_time_str)
    if not event_id:
        return f"Error: No appointment found at {old_start_time_str}."

    # First, check if the NEW slot is free
    availability = check_calendar_availability(new_start_time_str)
    if "BUSY" in availability:
        return f"Error: The new slot {new_start_time_str} is already occupied."

    # Get existing event to keep the summary/description
    event = service.events().get(calendarId=CALENDAR_ID, eventId=event_id).execute()
    
    new_naive_dt = datetime.datetime.fromisoformat(new_start_time_str)
    new_end_dt = new_naive_dt + datetime.timedelta(minutes=30)

    event['start']['dateTime'] = new_naive_dt.strftime('%Y-%m-%dT%H:%M:%S')
    event['end']['dateTime'] = new_end_dt.strftime('%Y-%m-%dT%H:%M:%S')

    service.events().update(calendarId=CALENDAR_ID, eventId=event_id, body=event).execute()
    return f"SUCCESS: Appointment moved from {old_start_time_str} to {new_start_time_str}."
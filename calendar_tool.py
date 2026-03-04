import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
import os
import pytz
from dotenv import load_dotenv
import config

load_dotenv()

SERVICE_ACCOUNT_FILE = 'service_account.json' 
CALENDAR_ID = os.getenv('CALENDER_ID')

# UPDATED: Changed from .readonly to allow event creation
SCOPES = ['https://www.googleapis.com/auth/calendar']
creds = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('calendar', 'v3', credentials=creds)

IST = pytz.timezone(config.CALENDAR_TIMEZONE)

def is_past_time(start_time_str: str):
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    now_dt = datetime.datetime.now(IST)
    return start_dt <= now_dt

def check_calendar_availability(start_time_str: str,duration_minutes: int = config.DEFAULT_APPOINTMENT_DURATION):
    """Checks if a 30-minute slot is free in Google Calendar."""

    if is_past_time(start_time_str):
        return "Error: Cannot book an appointment in the past."
    
    # NEW: Business Hours Check
    if not is_within_business_hours(start_time_str):
        return (f"Error: We only accept appointments between {config.BUSINESS_START_HOUR}:00 AM "
                f"and {config.BUSINESS_END_HOUR}:00 PM. Please choose a different time.")

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)
    end_dt = start_dt + datetime.timedelta(minutes=duration_minutes)
    
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


def suggest_next_available_slot(
    start_time_str: str,
    duration_minutes: int = config.DEFAULT_APPOINTMENT_DURATION,
    search_hours: int = 4,
    max_slots: int = 3
):

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    start_dt = IST.localize(naive_dt)

    end_search = start_dt + datetime.timedelta(hours=search_hours)

    current = start_dt + datetime.timedelta(minutes=duration_minutes)

    available_slots = []

    while current < end_search and len(available_slots) < max_slots:

        availability = check_calendar_availability(
            current.strftime("%Y-%m-%dT%H:%M:%S"),
            duration_minutes
        )

        if "FREE" in availability:
            available_slots.append(
                current.strftime("%Y-%m-%dT%H:%M:%S")
            )

        current += datetime.timedelta(minutes=duration_minutes)

    if available_slots:
        return f"AVAILABLE_SLOTS: {available_slots}"

    return "No available slots found in the next few hours."

def book_appointment(start_time_str: str, summary: str = "AI Appointment", duration_minutes: int = config.DEFAULT_APPOINTMENT_DURATION):
    
    if is_past_time(start_time_str):
        return "Error: Cannot book an appointment in the past."

    availability = check_calendar_availability(start_time_str)
    if "BUSY" in availability:
        return "Error: Slot already occupied."

    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    
    end_dt = naive_dt + datetime.timedelta(minutes=duration_minutes)

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

def reschedule_appointment(old_start_time_str: str, new_start_time_str: str, duration_minutes: int = config.DEFAULT_APPOINTMENT_DURATION):
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
    new_end_dt = new_naive_dt + datetime.timedelta(minutes=duration_minutes)

    event['start']['dateTime'] = new_naive_dt.strftime('%Y-%m-%dT%H:%M:%S')
    event['end']['dateTime'] = new_end_dt.strftime('%Y-%m-%dT%H:%M:%S')

    service.events().update(calendarId=CALENDAR_ID, eventId=event_id, body=event).execute()
    return f"SUCCESS: Appointment moved from {old_start_time_str} to {new_start_time_str}."

def is_within_business_hours(start_time_str: str):
    naive_dt = datetime.datetime.fromisoformat(start_time_str)
    hour = naive_dt.hour
    
    # Check if the time falls between 9 AM and 6 PM
    if config.BUSINESS_START_HOUR <= hour < config.BUSINESS_END_HOUR:
        return True
    return False
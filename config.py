#--------------calender_tool----------------
BUSINESS_START_HOUR = 9  # 9:00 AM
BUSINESS_END_HOUR = 18

DEFAULT_APPOINTMENT_DURATION = 30  # In minutes

CALENDAR_TIMEZONE = 'Asia/Kolkata'


#--------------main----------------
MAX_HISTORY = 4
MAX_TOOL_ITERATIONS = 2

MIN_CHUNK_CHARS = 20

#--------these are the frames sent to stt saaras model to warm it up before actual converstation
WARMUP_FRAMES = 10

#------------max times code tries to reconnect to saaras stt if it auto disconnects after some idle time where user speaks nothing
MAX_RECONNECTS = 15

#-----------this is the extra buffer timing taken after which user mic is enabled ,so that ai's last words are not echoed to user speaker 
AI_POST_TTS_BUFFER = 0.90
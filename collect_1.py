
# Collect TimeDateStamp
import pefile

try:
    # Load the PE file
    pe = pefile.PE("path/to/your/executable.exe")

    # Access the TimeDateStamp from the COFF File Header
    time_date_stamp = pe.FILE_HEADER.TimeDateStamp

    print(f"TimeDateStamp (raw): {hex(time_date_stamp)}")

    # Convert the timestamp to a human-readable format
    import datetime
    dt_object = datetime.datetime.fromtimestamp(time_date_stamp)
    print(f"TimeDateStamp (human-readable): {dt_object}")

except pefile.PEFormatError as e:
    print(f"Error parsing PE file: {e}")
except FileNotFoundError:
    print("File not found. Please provide a valid path to the executable.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
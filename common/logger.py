# Configure logging
import logging
import os
import colorlog


handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
))

logger = colorlog.getLogger()
logger.addHandler(handler)
logger.setLevel(logging.DEBUG if os.getenv('DEBUG_MODE', 'False').lower() == 'true' else logging.INFO)

# Some unncessary coloring
import pyfiglet
from termcolor import colored
import time
import sys

# Function to print text with a delay
def delayed_print(text, delay=0.1):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        if char :
            time.sleep(delay)
    sys.stdout.write('\n')  # Move to the next line after finishing
    sys.stdout.flush()

# Create big text using a larger font
big_text = pyfiglet.figlet_format("ECHO ECHO", font="big")
# Create small text
small_text = "by Ayoub Achak"
bottom_text = "2024 All rights reserved."

# Print big text in color with delay
for line in big_text.splitlines():
    delayed_print(colored(line, 'cyan'), delay=0.001)

# Print small text aligned to the bottom right with delay
delayed_print(" " * (80 - len(small_text)) + colored(small_text, 'yellow'), delay=0.001)

# Print bottom text with delay
delayed_print("\n" + colored(bottom_text, 'magenta'), delay=0.001)
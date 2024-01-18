import subprocess
import random
import time

# List of standard internet webpages
webpages = [
    "http://www.google.com",
    "http://www.facebook.com",
    "http://www.gmail.com",
    "http://www.amazon.com",
    "http://www.twitter.com",
    "http://www.linkedin.com",
    "http://www.reddit.com",
    "http://www.youtube.com",
    "http://www.instagram.com",
    "http://www.wikipedia.org",
    "http://www.microsoft.com",
    "http://www.apple.com",
    "http://www.yahoo.com",
    "http://www.netflix.com",
    "http://www.stackoverflow.com",
    "http://www.github.com",
    "http://www.dropbox.com"
]

while True:
    # Generate a random interval between 1 and 10 seconds
    random_interval = random.randint(1, 10)
    
    # Select a random webpage from the extended list
    random_webpage = random.choice(webpages)

    # Form the wget command
    wget_command = ["wget", random_webpage]

    try:
        # Execute the wget command
        subprocess.run(wget_command, check=True)
        print(f"Successfully fetched {random_webpage}")
    except subprocess.CalledProcessError as e:
        print(f"Error fetching {random_webpage}: {e}")

    # Wait for the next random interval
    time.sleep(random_interval)
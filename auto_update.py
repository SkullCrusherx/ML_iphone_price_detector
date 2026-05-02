import os
import time

while True:
    os.system("git add .")
    os.system('git commit -m "auto commit"')
    os.system("git push origin main")
    time.sleep(10)
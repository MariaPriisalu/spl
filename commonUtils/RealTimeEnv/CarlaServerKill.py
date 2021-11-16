# Kill the app running at port if needed
import os
cmd = f"sudo kill -9 $(sudo lsof -t -i:2000) > /dev/null 2>&1"
ret = os.system(cmd)

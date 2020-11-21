from __future__ import print_function
from builtins import range
import MalmoPython
import os
import sys
import time
import numpy as np
from PIL import Image
import pandas as pd

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)


HEIGHT = 360
WIDTH = 640
missionXML='''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
            <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
            
            <About>
                <Summary>Coordinate Data Mining</Summary>
            </About>
              
            <ServerSection>
                <ServerInitialConditions>
                    <Time>
                        <StartTime>1000</StartTime>
                        <AllowPassageOfTime>false</AllowPassageOfTime> <!-- Keep steady daylight to make image parsing simple -->
                    </Time>
                    <Weather>clear</Weather> <!-- Keep steady weather to make image parsing simple -->
                </ServerInitialConditions>
                <ServerHandlers>
                    <FileWorldGenerator src="C:\\Users\\alaister\\Desktop\\CS175\\malmo\\Minecraft\\run\\saves\\Washington" />
                    <ServerQuitWhenAnyAgentFinishes/>
                </ServerHandlers>
            </ServerSection>
              
            <AgentSection mode="Creative">
                <Name>SteveBot</Name>
                <AgentStart/>
                <AgentHandlers>
                    <ObservationFromFullStats/>
                    <AbsoluteMovementCommands/>
                    <DiscreteMovementCommands/>
                    <VideoProducer>
                        <Width>''' + str(WIDTH) + '''</Width>
                        <Height>''' + str(HEIGHT) + '''</Height>
                    </VideoProducer>
                    <MissionQuitCommands quitDescription="Finished"/>
                </AgentHandlers>
            </AgentSection>
            </Mission>'''



def saveImg(pix,h,w,name):
    pixels = np.array(pix)
    pixels = pixels.reshape((h,w,3))
    image = Image.fromarray(pixels,mode="RGB")
    image.save(name)


# Create default Malmo objects:
agent_host = MalmoPython.AgentHost()
agent_host.setVideoPolicy(MalmoPython.VideoPolicy.LATEST_FRAME_ONLY)
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

my_mission = MalmoPython.MissionSpec(missionXML, True)
my_mission_record = MalmoPython.MissionRecordSpec()
my_mission_record.recordMP4(20, 800000)

# Attempt to start a mission:
max_retries = 3
for retry in range(max_retries):
    try:
        agent_host.startMission( my_mission, my_mission_record )
        break
    except RuntimeError as e:
        if retry == max_retries - 1:
            print("Error starting mission:",e)
            exit(1)
        else:
            time.sleep(2)


# Loop until mission starts:
print("Waiting for the mission to start")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    world_state = agent_host.getWorldState()
    for error in world_state.errors:
        print("Error:",error.text)


# Mission Running
print("Mission is running")


coordinate_list = list()
time.sleep(1)
counter = 0

home_x = 30
home_z = 100

time.sleep(10)
for xcood in range(-10,11,1):
    for zcood in range(-10,11,1):
        
        # calculate the coordinates
        x = xcood*10 + home_x
        z = zcood*10 + home_z
        
        agent_host.sendCommand("setYaw 180")
        # teleport & let the scene load
        agent_host.sendCommand(f"tp {x} 120 {z}")
        time.sleep(5)

        # get current frame and export as jpg
        world_state = agent_host.getWorldState()
        img = world_state.video_frames[0].pixels
        saveImg(img,HEIGHT,WIDTH,f"{counter}.jpg")
        print(f"Picture Saved for {(x,z)}")
        
        # update necessary information 
        counter += 1
        coordinate_list.append((x,z))


        time.sleep(1)
        agent_host.sendCommand("setYaw 90")
        time.sleep(2)
        world_state = agent_host.getWorldState()
        img = world_state.video_frames[0].pixels
        saveImg(img,HEIGHT,WIDTH,f"{counter}.jpg")
        print(f"Picture Saved for {(x,z)}")
        counter += 1

        time.sleep(1)
        agent_host.sendCommand("setYaw 0")
        time.sleep(2)
        world_state = agent_host.getWorldState()
        img = world_state.video_frames[0].pixels
        saveImg(img,HEIGHT,WIDTH,f"{counter}.jpg")
        print(f"Picture Saved for {(x,z)}")
        counter += 1

        time.sleep(1)
        agent_host.sendCommand("setYaw -90")
        time.sleep(2)
        world_state = agent_host.getWorldState()
        img = world_state.video_frames[0].pixels
        saveImg(img,HEIGHT,WIDTH,f"{counter}.jpg")
        print(f"Picture Saved for {(x,z)}")
        counter += 1
        

        
        

        # debug purpose
        for error in world_state.errors:
            print("Error:",error.text)



agent_host.sendCommand("quit")

# Save to CSV
df = pd.DataFrame(coordinate_list, columns=["x","z"])
df.to_csv("Coordinates.csv",index=False)

# Indicate Mission Ended
print("Mission ended")

# kill the video recording thread
os._exit(1)

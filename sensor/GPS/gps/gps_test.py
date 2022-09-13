import time
import gps

class cansat(object):
    def __init__(self):
        self.gps = gps.GPS()
        self.goallat = 40.900545
        self.goallon = 119.079109
        self.state = 0

    def setup(self):
        self.gps.setupGps()
        
    def writeData(self):
        self.gps.gpsread()
                
        timer = 1000*(time.time() - start_time)
        timer = int(timer)
        datalog = str(timer) + ","\
                  + "Time:" + str(self.gps.Time) + ","\
                  + "緯度:" + str(self.gps.Lat) + ","\
                  + "経度:" + str(self.gps.Lon)
        print(datalog)
        distance = self.gps.vincenty_inverse(float(self.gps.Lat),float(self.gps.Lon),self.goallat,self.goallon) #距離:self.gps.gpsdis 方位角:self.gps.gpsdegrees
        print(f"distance:{distance} m")

start_time = time.time()
cansat = cansat() 
cansat.setup()
while True:
    cansat.writeData()
    time.sleep(1)     
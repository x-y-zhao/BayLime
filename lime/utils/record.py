import os
import time

class record: 

    def __init__(self,filename,startTime): 

        self.startTime = startTime

        directory = os.path.dirname(filename)
        try:
            os.stat(directory)
        except:
            os.mkdir(directory) 
        self.file = open(filename,"w+") 
        
    def write(self,text): 
        self.file.write(text) 
        
    def close(self): 
        self.file.close()


    def resetTime(self): 
        self.write("reset time at %s\n\n"%(time.time() - self.startTime))
        self.startTime = time.time()

def writeInfo(r, i, h1_del, h1_ins, h2_del, h2_ins, h3_del, h3_ins):
    r.write("time:%s\n" % (time.time() - r.startTime))
    r.write('--------------------------\n')
    r.write("No. of Image: %s\n" % (i))
    r.write("Lime Deletion: %.5f\n" % (h1_del))
    r.write("Lime Insertion: %.5f\n" % (h1_ins))
    r.write("Grad-CAM Deletion: %.5f\n" % (h2_del))
    r.write("Grad-CAM Insertion: %.5f\n" % (h2_ins))
    r.write("BayLime Deletion: %.5f\n" % (h3_del))
    r.write("BayLime Insertion: %.5f\n" % (h3_ins))
    r.write('--------------------------\n')
    r.write('--------------------------\n')
    r.close()




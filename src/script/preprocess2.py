import os
import fnmatch
import subprocess
outdir = 'video'

for root, dir, files in os.walk("."):
        #print root
        mfiles = fnmatch.filter(files, "*.png")
        filenum = len(mfiles)
        if filenum > 0:
            #print root+'/'+mfiles[0]
            suffix = mfiles[0].split('.')[-1]
            name = mfiles[0].split('.')[0]
            comname = name[0:-2]+'%02d'
            label = 'None'
            for r,d,f in os.walk('./emotion'+root[1:]):
                print f
                if len(f)>0:
                    with open(r+'/'+f[0]) as labelfile:
                        label = str(int(float(labelfile.readline().strip())))
                        
            if not label=='None':
                command = 'ffmpeg -i '+root+'/'+comname+'.'+suffix+' '+ outdir+'/'+name+'emotion'+label+'.mp4'
                os.system(command)
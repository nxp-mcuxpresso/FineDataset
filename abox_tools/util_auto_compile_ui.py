'''
监视ui文件，如果更新了，就自动调用pyuic5重新编译
待监测的ui文件位于 lstUIs 中
'''

import time
from os.path import getmtime
import os
lstUIs = ['pyqt5_ab_main.ui', 'pyqt5_ab_dlg_cfg_cluster.ui']

lstTSs = [0] * len(lstUIs)

for (i, uiFile) in enumerate(lstUIs):
    lstTSs[i] = getmtime(uiFile)

print('monitoring')
while True:
    time.sleep(1)
    for (i, uiFile) in enumerate(lstUIs):
        t = getmtime(uiFile)
        sOutFile = lstUIs[i][:-2] + 'py'
        if t > lstTSs[i] or not os.path.exists(sOutFile):
            lstTSs[i] = t
            sCmd = 'pyuic5 -o %s %s' % (lstUIs[i][:-2] + 'py', lstUIs[i])
            os.system(sCmd)

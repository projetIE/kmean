# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:11:52 2023

@author: yrolland
check si les packages sont installés
"""


import pip

def import_or_install(package):
    try:
        __import__(package)
        print("imported",package)
    except ImportError:
        pip.main(['install', package])  
        
        
        
        
import_or_install("cv2")
import_or_install("numpy")


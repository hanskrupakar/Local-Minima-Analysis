from setuptools import setup, find_packages
import os
import sys

dependencies = [
    "certifi",      
    "cffi",         
    "cycler",                   
    "Cython",                  
    "kiwisolver",                
    "matplotlib",                
    "mkl-fft",                  
    "mkl-random",                
    "numpy",                    
    "olefile",                    
    "opencv-python",          
    "Pillow",                    
    "pip",                      
    "pycparser",                  
    "pyparsing",                 
    "python-dateutil",           
    "rawpy",                
    "setuptools",           
    "six",                  
    "torch",                 
    "torchvision", 
    "wheel"
    ]
	
setup(name='minima_exploration',
      version='1.0',
      description='Minima-Exploration-Study',
      author='hanskrupakar',
      author_email='hansk@nyu.edu',
      license='Open-Source',
      url='https://www.github.com/hanskrupakar/Local-Minima-Analysis',
      packages=[],
      install_requires=dependencies,
)

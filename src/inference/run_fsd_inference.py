import tensorflow.compat.v1 as tf
import cv2
import numpy as np
import colorsys
from ultralytics import YOLO
from subprocess import call
from typing import List, Tuple
import concurrent.futures
from src.models import model
import time

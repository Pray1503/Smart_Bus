#!/usr/bin/env python3
"""
Smart Bus Management System - Data Generation Script
Generates synthetic bus ridership and GPS data for Bangalore routes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import random
import math
import uuid

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class BangaloreBusDataGenerator:
    """Generate synthetic bus data for Bangalore routes based on AMTS patterns"""
    
    def __init__(self):
        # Define Bangalore bus routes based on AMTS structure
        self.ahmedabad_routes = {
                1: {
                    "source": "Ratan Park",
                    "destination": "Lal Darwaja",
                    "distance_km": 9.5,
                    "fare": 20,
                    "stops": ["Ratan Park", "Lal Darwaja"],
                    "shift_trips": {"1st": 13, "2nd": 14}
                },
                4: {
                    "source": "Lal Darwaja",
                    "destination": "Lal Darwaja",
                    "distance_km": 22.8,
                    "fare": 30,
                    "stops": ["Lal Darwaja", "Lal Darwaja"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                5: {
                    "source": "Lal Darwaja",
                    "destination": "Lal Darwaja",
                    "distance_km": 22.8,
                    "fare": 30,
                    "stops": ["Lal Darwaja", "Lal Darwaja"],
                    "shift_trips": {"1st": 6, "2nd": 7}
                },
                14: {
                    "source": "Lal Darwaja",
                    "destination": "Chosar Gam",
                    "distance_km": 18.7,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "Chosar Gam"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                15: {
                    "source": "Vivekanand Nagar",
                    "destination": "Civil Hospital",
                    "distance_km": 22.15,
                    "fare": 30,
                    "stops": ["Vivekanand Nagar", "Civil Hospital"],
                    "shift_trips": {"1st": 7, "2nd": 8}
                },
                16: {
                    "source": "Nigam Society",
                    "destination": "Chiloda Octroi Naka",
                    "distance_km": 27.8,
                    "fare": 30,
                    "stops": ["Nigam Society", "Chiloda Octroi Naka"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                17: {
                    "source": "Nigam Society",
                    "destination": "Meghani Nagar",
                    "distance_km": 16.45,
                    "fare": 25,
                    "stops": ["Nigam Society", "Meghani Nagar"],
                    "shift_trips": {"1st": 8, "2nd": 7}
                },
                18: {
                    "source": "Kalupur",
                    "destination": "Punit Nagar",
                    "distance_km": 8.9,
                    "fare": 20,
                    "stops": ["Kalupur", "Punit Nagar"],
                    "shift_trips": {"1st": 14, "2nd": 14}
                },
                22: {
                    "source": "Tragad Gam",
                    "destination": "Lambha Gam",
                    "distance_km": 32.05,
                    "fare": 30,
                    "stops": ["Tragad Gam", "Lambha Gam"],
                    "shift_trips": {"1st": 5, "2nd": 5}
                },
                23: {
                    "source": "Isanpur",
                    "destination": "Jivandeep Circular",
                    "distance_km": 22.9,
                    "fare": 30,
                    "stops": ["Isanpur", "Jivandeep Circular"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                28: {
                    "source": "Meghani Nagar",
                    "destination": "Lambha Gam",
                    "distance_km": 19.85,
                    "fare": 25,
                    "stops": ["Meghani Nagar", "Lambha Gam"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                31: {
                    "source": "Sarkhej Gam",
                    "destination": "Meghaninagar",
                    "distance_km": 19.85,
                    "fare": 25,
                    "stops": ["Sarkhej Gam", "Meghaninagar"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                32: {
                    "source": "Butbhavani Mand",
                    "destination": "Shahiyadri Bung",
                    "distance_km": 18.6,
                    "fare": 25,
                    "stops": ["Butbhavani Mand", "Shahiyadri Bung"],
                    "shift_trips": {"1st": 7, "2nd": 8}
                },
                33: {
                    "source": "Narayan Nagar",
                    "destination": "Manmohan Park",
                    "distance_km": 19.65,
                    "fare": 25,
                    "stops": ["Narayan Nagar", "Manmohan Park"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                34: {
                    "source": "Butbhavani Mand",
                    "destination": "Kalapi Nagar",
                    "distance_km": 17.55,
                    "fare": 25,
                    "stops": ["Butbhavani Mand", "Kalapi Nagar"],
                    "shift_trips": {"1st": 10, "2nd": 8}
                },
                35: {
                    "source": "Lal Darwaja",
                    "destination": "Matoda Patia",
                    "distance_km": 25.95,
                    "fare": 30,
                    "stops": ["Lal Darwaja", "Matoda Patia"],
                    "shift_trips": {"1st": 5, "2nd": 6}
                },
                36: {
                    "source": "Sarangpur",
                    "destination": "Sarkhej Gam",
                    "distance_km": 14.0,
                    "fare": 25,
                    "stops": ["Sarangpur", "Sarkhej Gam"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                37: {
                    "source": "Vasna",
                    "destination": "Tejendra Nagar",
                    "distance_km": 17.1,
                    "fare": 25,
                    "stops": ["Vasna", "Tejendra Nagar"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                38: {
                    "source": "Juhapura",
                    "destination": "Meghani Nagar",
                    "distance_km": 16.65,
                    "fare": 25,
                    "stops": ["Juhapura", "Meghani Nagar"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                40: {
                    "source": "Vasna",
                    "destination": "Lapkaman",
                    "distance_km": 21.95,
                    "fare": 30,
                    "stops": ["Vasna", "Lapkaman"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                42: {
                    "source": "Ghodasar",
                    "destination": "Judges Bunglows",
                    "distance_km": 17.65,
                    "fare": 25,
                    "stops": ["Ghodasar", "Judges Bunglows"],
                    "shift_trips": {"1st": 7, "2nd": 6}
                },
                43: {
                    "source": "Lal Darwaja",
                    "destination": "Judges Bunglow",
                    "distance_km": 9.4,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Judges Bunglow"],
                    "shift_trips": {"1st": 14, "2nd": 12}
                },
                45: {
                    "source": "Lal Darwaja",
                    "destination": "Jodhpur Gam",
                    "distance_km": 8.4,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Jodhpur Gam"],
                    "shift_trips": {"1st": 15, "2nd": 14}
                },
                46: {
                    "source": "Kalupur",
                    "destination": "Kalupur",
                    "distance_km": 18.2,
                    "fare": 25,
                    "stops": ["Kalupur", "Kalupur"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                47: {
                    "source": "Kalupur",
                    "destination": "Kalupur",
                    "distance_km": 18.2,
                    "fare": 25,
                    "stops": ["Kalupur", "Kalupur"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                48: {
                    "source": "Kalupur",
                    "destination": "Prhalad Nagar",
                    "distance_km": 14.25,
                    "fare": 25,
                    "stops": ["Kalupur", "Prhalad Nagar"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                49: {
                    "source": "Adinath Nagar",
                    "destination": "Manipur Vad",
                    "distance_km": 29.15,
                    "fare": 30,
                    "stops": ["Adinath Nagar", "Manipur Vad"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                50: {
                    "source": "Ghuma Gam",
                    "destination": "Meghani Nagar",
                    "distance_km": 25.7,
                    "fare": 30,
                    "stops": ["Ghuma Gam", "Meghani Nagar"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                52: {
                    "source": "Punit Nagar",
                    "destination": "Thaltej",
                    "distance_km": 21.6,
                    "fare": 30,
                    "stops": ["Punit Nagar", "Thaltej"],
                    "shift_trips": {"1st": 7, "2nd": 6}
                },
                54: {
                    "source": "Vatva Rly Cross",
                    "destination": "Vaishnodevi Man",
                    "distance_km": 34.4,
                    "fare": 30,
                    "stops": ["Vatva Rly Cross", "Vaishnodevi Man"],
                    "shift_trips": {"1st": 5, "2nd": 5}
                },
                56: {
                    "source": "Sitaram Bapa Chowk",
                    "destination": "Judges Bunglows",
                    "distance_km": 24.95,
                    "fare": 30,
                    "stops": ["Sitaram Bapa Chowk", "Judges Bunglows"],
                    "shift_trips": {"1st": 5, "2nd": 6}
                },
                58: {
                    "source": "Thaltej Gam",
                    "destination": "Kush Society",
                    "distance_km": 30.15,
                    "fare": 30,
                    "stops": ["Thaltej Gam", "Kush Society"],
                    "shift_trips": {"1st": 5, "2nd": 5}
                },
                60: {
                    "source": "Maninagar",
                    "destination": "Judges Bunglows",
                    "distance_km": 18.65,
                    "fare": 25,
                    "stops": ["Maninagar", "Judges Bunglows"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                61: {
                    "source": "Maninagar",
                    "destination": "Gujarat High Court",
                    "distance_km": 19.6,
                    "fare": 25,
                    "stops": ["Maninagar", "Gujarat High Court"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                63: {
                    "source": "Maninagar",
                    "destination": "Gujarat High Court",
                    "distance_km": 19.45,
                    "fare": 25,
                    "stops": ["Maninagar", "Gujarat High Court"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                64: {
                    "source": "Lal Darwaja",
                    "destination": "Gujarat High Court",
                    "distance_km": 11.35,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Gujarat High Court"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                65: {
                    "source": "Lal Darwaja",
                    "destination": "Sola Bhagwat Vidhyapith",
                    "distance_km": 13.55,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Sola Bhagwat Vidhyapith"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                66: {
                    "source": "Kalupur Terminu",
                    "destination": "Shilaj Gam",
                    "distance_km": 16.5,
                    "fare": 25,
                    "stops": ["Kalupur Terminu", "Shilaj Gam"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                67: {
                    "source": "Kalupur",
                    "destination": "Satadhar Society",
                    "distance_km": 11.1,
                    "fare": 20,
                    "stops": ["Kalupur", "Satadhar Society"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                68: {
                    "source": "Kalupur",
                    "destination": "Sattadhar Society",
                    "distance_km": 17.2,
                    "fare": 25,
                    "stops": ["Kalupur", "Sattadhar Society"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                69: {
                    "source": "Kalupur",
                    "destination": "Chanakyapuri",
                    "distance_km": 10.35,
                    "fare": 20,
                    "stops": ["Kalupur", "Chanakyapuri"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                70: {
                    "source": "Vagheshwari Soc",
                    "destination": "Naroda Terminus",
                    "distance_km": 20.05,
                    "fare": 30,
                    "stops": ["Vagheshwari Soc", "Naroda Terminus"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                72: {
                    "source": "Maninagar",
                    "destination": "Nava Vadaj",
                    "distance_km": 18.2,
                    "fare": 25,
                    "stops": ["Maninagar", "Nava Vadaj"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                74: {
                    "source": "Nava Ranip",
                    "destination": "Nigam Society",
                    "distance_km": 21.55,
                    "fare": 30,
                    "stops": ["Nava Ranip", "Nigam Society"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                75: {
                    "source": "Maninagar",
                    "destination": "Chandkheda",
                    "distance_km": 21.2,
                    "fare": 30,
                    "stops": ["Maninagar", "Chandkheda"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                76: {
                    "source": "Vatva Ind.Towns",
                    "destination": "Gujarat High Court",
                    "distance_km": 25.7,
                    "fare": 30,
                    "stops": ["Vatva Ind.Towns", "Gujarat High Court"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                77: {
                    "source": "Vadaj Terminus",
                    "destination": "Hatkeshwar",
                    "distance_km": 12.35,
                    "fare": 20,
                    "stops": ["Vadaj Terminus", "Hatkeshwar"],
                    "shift_trips": {"1st": 9, "2nd": 9}
                },
                79: {
                    "source": "Thakkarbapa Nagar",
                    "destination": "Chenpur Gam",
                    "distance_km": 19.0,
                    "fare": 25,
                    "stops": ["Thakkarbapa Nagar", "Chenpur Gam"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                82: {
                    "source": "Lal Darwaja",
                    "destination": "Nirnay Nagar",
                    "distance_km": 10.45,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Nirnay Nagar"],
                    "shift_trips": {"1st": 11, "2nd": 11}
                },
                83: {
                    "source": "Lal Darwaja",
                    "destination": "Sabarmati D Cab",
                    "distance_km": 14.1,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "Sabarmati D Cab"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                84: {
                    "source": "Mani Nagar",
                    "destination": "Chandkheda Gam",
                    "distance_km": 23.6,
                    "fare": 30,
                    "stops": ["Mani Nagar", "Chandkheda Gam"],
                    "shift_trips": {"1st": 5, "2nd": 6}
                },
                85: {
                    "source": "Lal Darwaja",
                    "destination": "Chandkheda Gam",
                    "distance_km": 12.85,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Chandkheda Gam"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                87: {
                    "source": "Maninagar",
                    "destination": "Chandkheda Gam",
                    "distance_km": 24.95,
                    "fare": 30,
                    "stops": ["Maninagar", "Chandkheda Gam"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                88: {
                    "source": "Ranip",
                    "destination": "Nikol Gam",
                    "distance_km": 16.45,
                    "fare": 25,
                    "stops": ["Ranip", "Nikol Gam"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                90: {
                    "source": "Tragad Gam",
                    "destination": "Meghaninagar",
                    "distance_km": 21.15,
                    "fare": 30,
                    "stops": ["Tragad Gam", "Meghaninagar"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                96: {
                    "source": "Vatva Rly Cross",
                    "destination": "Circuit House",
                    "distance_km": 22.65,
                    "fare": 30,
                    "stops": ["Vatva Rly Cross", "Circuit House"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                101: {
                    "source": "Lal Darwaja",
                    "destination": "Sardar Nagar",
                    "distance_km": 9.45,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Sardar Nagar"],
                    "shift_trips": {"1st": 14, "2nd": 14}
                },
                102: {
                    "source": "Lal Darwaja",
                    "destination": "New Airport",
                    "distance_km": 10.8,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "New Airport"],
                    "shift_trips": {"1st": 12, "2nd": 12}
                },
                105: {
                    "source": "Lal Darwaja",
                    "destination": "Naroda Ind East",
                    "distance_km": 16.35,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "Naroda Ind East"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                112: {
                    "source": "Lal Darwaja",
                    "destination": "Kubernagar",
                    "distance_km": 11.05,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Kubernagar"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                116: {
                    "source": "Civil Hospital",
                    "destination": "Danilimda Gam",
                    "distance_km": 9.5,
                    "fare": 20,
                    "stops": ["Civil Hospital", "Danilimda Gam"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                117: {
                    "source": "Suedge Farm App",
                    "destination": "Kalapi Nagar",
                    "distance_km": 12.85,
                    "fare": 20,
                    "stops": ["Suedge Farm App", "Kalapi Nagar"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                122: {
                    "source": "Lal Darwaja",
                    "destination": "Ambawadi Police",
                    "distance_km": 15.25,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "Ambawadi Police"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                "123 SH": {
                    "source": "Lal Darwaja",
                    "destination": "Krushna Nagar",
                    "distance_km": 10.35,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Krushna Nagar"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                125: {
                    "source": "Lal Darwaja",
                    "destination": "Vahelal Gam",
                    "distance_km": 27.45,
                    "fare": 30,
                    "stops": ["Lal Darwaja", "Vahelal Gam"],
                    "shift_trips": {"1st": 8, "2nd": 7}
                },
                126: {
                    "source": "Sarangpur",
                    "destination": "Sardarnagar",
                    "distance_km": 14.4,
                    "fare": 25,
                    "stops": ["Sarangpur", "Sardarnagar"],
                    "shift_trips": {"1st": 9, "2nd": 9}
                },
                127: {
                    "source": "Sarangpur",
                    "destination": "Sukan Bunglow",
                    "distance_km": 12.4,
                    "fare": 20,
                    "stops": ["Sarangpur", "Sukan Bunglow"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                128: {
                    "source": "Mani Nagar",
                    "destination": "Naroda Ind Town",
                    "distance_km": 17.25,
                    "fare": 25,
                    "stops": ["Mani Nagar", "Naroda Ind Town"],
                    "shift_trips": {"1st": 7, "2nd": 7}
                },
                129: {
                    "source": "Haridarshan",
                    "destination": "Vasna",
                    "distance_km": 23.35,
                    "fare": 30,
                    "stops": ["Haridarshan", "Vasna"],
                    "shift_trips": {"1st": 7, "2nd": 6}
                },
                130: {
                    "source": "Naroda Terminus",
                    "destination": "Indira Nagar 2",
                    "distance_km": 24.35,
                    "fare": 30,
                    "stops": ["Naroda Terminus", "Indira Nagar 2"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                134: {
                    "source": "Lal Darwaja",
                    "destination": "Thakkarbapanagar",
                    "distance_km": 10.75,
                    "fare": 20,
                    "stops": ["Lal Darwaja", "Thakkarbapanagar"],
                    "shift_trips": {"1st": 10, "2nd": 10}
                },
                135: {
                    "source": "Lal Darwaja",
                    "destination": "New India Colony",
                    "distance_km": 14.55,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "New India Colony"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                136: {
                    "source": "New India Colon",
                    "destination": "Sattadhar Society",
                    "distance_km": 27.0,
                    "fare": 30,
                    "stops": ["New India Colon", "Sattadhar Society"],
                    "shift_trips": {"1st": 6, "2nd": 6}
                },
                138: {
                    "source": "Bapunagar",
                    "destination": "Ghuma Gam",
                    "distance_km": 23.7,
                    "fare": 30,
                    "stops": ["Bapunagar", "Ghuma Gam"],
                    "shift_trips": {"1st": 6, "2nd": 5}
                },
                141: {
                    "source": "Lal Darwaja",
                    "destination": "Rakhial Char Rasta",
                    "distance_km": 7.95,
                    "fare": 15,
                    "stops": ["Lal Darwaja", "Rakhial Char Rasta"],
                    "shift_trips": {"1st": 14, "2nd": 14}
                },
                142: {
                    "source": "Vastral Gam",
                    "destination": "Gujarat University",
                    "distance_km": 19.05,
                    "fare": 25,
                    "stops": ["Vastral Gam", "Gujarat University"],
                    "shift_trips": {"1st": 8, "2nd": 7}
                },
                143: {
                    "source": "Lal Darwaja",
                    "destination": "Bhuvaldi Gam",
                    "distance_km": 16.25,
                    "fare": 25,
                    "stops": ["Lal Darwaja", "Bhuvaldi Gam"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                144: {
                    "source": "Arbuda Nagar",
                    "destination": "Gujarat University",
                    "distance_km": 15.95,
                    "fare": 25,
                    "stops": ["Arbuda Nagar", "Gujarat University"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                145: {
                    "source": "Arbudanagar",
                    "destination": "Civil Hospital",
                    "distance_km": 10.2,
                    "fare": 20,
                    "stops": ["Arbudanagar", "Civil Hospital"],
                    "shift_trips": {"1st": 13, "2nd": 12}
                },
                147: {
                    "source": "Surbhi Society",
                    "destination": "Vagheshwari Society",
                    "distance_km": 18.85,
                    "fare": 25,
                    "stops": ["Surbhi Society", "Vagheshwari Society"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                148: {
                    "source": "Sarangpur",
                    "destination": "Kathwada Gam",
                    "distance_km": 13.85,
                    "fare": 20,
                    "stops": ["Sarangpur", "Kathwada Gam"],
                    "shift_trips": {"1st": 8, "2nd": 8}
                },
                150: {
                    "source": "Sarkhej Gam",
                    "destination": "Chinubhai Nagar",
                    "distance_km": 24.05,
                    "fare": 30,
                    "stops": ["Sarkhej Gam", "Chinubhai Nagar"],
                    "shift_trips": {"1st": 5, "2nd": 5}
                }



            }

        
        # Bangalore coordinates (approximate)
        self.city_center = {'lat': 12.9716, 'lng': 77.5946}
        
        # Time periods for ridership patterns
        self.start_date = datetime(2024, 1, 1)
        self.end_date = datetime(2024, 1, 8)  # 1 week of data
        
    # def generate_gps_coordinates(self, route_id, stop_index, total_stops):
    #     """Generate realistic GPS coordinates for bus stops"""
    #     base_lat = self.city_center['lat']
    #     base_lng = self.city_center['lng']
        
    #     # Create variation based on route and stop
    #     route_offset = float(route_id) * 0.01
    #     stop_offset = (stop_index / total_stops) * 0.02
        
    #     # Add some randomness for realistic coordinates
    #     lat_variation = np.random.normal(0, 0.005)
    #     lng_variation = np.random.normal(0, 0.005)
        
    #     lat = base_lat + route_offset + stop_offset + lat_variation
    #     lng = base_lng + route_offset + stop_offset + lng_variation
        
    #     return round(lat, 6), round(lng, 6)
    def generate_gps_coordinates(self, route_id, stop_index, total_stops):
        """Generate realistic GPS coordinates for bus stops"""
        base_lat = self.city_center['lat']
        base_lng = self.city_center['lng']
        
        # Ensure route_id is numeric
        try:
            route_num = float(route_id)
        except ValueError:
            route_num = 0  # fallback if route_id is not a number
        
        # Create variation based on route and stop
        route_offset = route_num * 0.01
        stop_offset = (stop_index / total_stops) * 0.02
        
        # Add some randomness for realistic coordinates
        lat_variation = np.random.normal(0, 0.005)
        lng_variation = np.random.normal(0, 0.005)
        
        lat = base_lat + route_offset + stop_offset + lat_variation
        lng = base_lng + route_offset + stop_offset + lng_variation
        
        return round(lat, 6), round(lng, 6)

    
    def generate_ridership_pattern(self, hour, day_of_week, weather_factor=1.0):
        """Generate realistic ridership patterns based on time and conditions"""
        base_ridership = 50
        
        # Hour-based patterns (rush hours have higher ridership)
        if 7 <= hour <= 9:  # Morning rush
            hourly_multiplier = 3.5
        elif 17 <= hour <= 19:  # Evening rush
            hourly_multiplier = 3.2
        elif 10 <= hour <= 16:  # Daytime
            hourly_multiplier = 2.0
        elif 20 <= hour <= 22:  # Evening
            hourly_multiplier = 1.5
        else:  # Night/early morning
            hourly_multiplier = 0.5
        
        # Day of week patterns (weekdays vs weekends)
        if day_of_week < 5:  # Weekdays
            day_multiplier = 1.2
        else:  # Weekends
            day_multiplier = 0.7
        
        # Calculate base ridership
        ridership = base_ridership * hourly_multiplier * day_multiplier * weather_factor
        
        # Add random variation
        ridership += np.random.normal(0, ridership * 0.15)
        
        return max(0, int(ridership))
    
    def generate_weather_factor(self, date):
        """Generate weather impact factor"""
        # Simulate seasonal and random weather effects
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal pattern (monsoon season has lower ridership)
        seasonal_factor = 1.0
        if 150 < day_of_year < 250:  # Monsoon season
            seasonal_factor = 0.8
        
        # Random weather events
        if random.random() < 0.1:  # 10% chance of bad weather
            weather_factor = random.uniform(0.5, 0.8)
        else:
            weather_factor = random.uniform(0.9, 1.1)
        
        return seasonal_factor * weather_factor
    
    def generate_gps_logs(self):
        """Generate GPS tracking logs for buses"""
        gps_logs = []
        
        current_time = self.start_date
        
        while current_time < self.end_date:
            for route_id, route_info in self.ahmedabad_routes.items():
                stops = route_info['stops']
                
                # Generate multiple bus trips per hour for each route
                trips_per_hour = random.randint(2, 6)
                
                for trip in range(trips_per_hour):
                    bus_id = f"KA-01-{route_id}-{random.randint(1000, 9999)}"
                    trip_start_time = current_time + timedelta(minutes=random.randint(0, 59))
                    
                    # Generate GPS points for each stop
                    for stop_idx, stop_name in enumerate(stops):
                        lat, lng = self.generate_gps_coordinates(route_id, stop_idx, len(stops))
                        
                        # Add travel time between stops
                        stop_time = trip_start_time + timedelta(minutes=stop_idx * 5)
                        
                        # Add some GPS noise and speed variation
                        speed = random.randint(15, 45)  # km/h
                        
                        gps_logs.append({
                            'timestamp': stop_time.strftime('%Y-%m-%d %H:%M:%S'),
                            'bus_id': bus_id,
                            'route_id': route_id,
                            'lat': lat,
                            'lng': lng,
                            'speed': speed,
                            'stop_name': stop_name,
                            'stop_sequence': stop_idx + 1
                        })
            
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(gps_logs)
    
    def generate_ridership_data(self):
        """Generate hourly ridership data"""
        ridership_data = []
        
        current_time = self.start_date
        
        while current_time < self.end_date:
            weather_factor = self.generate_weather_factor(current_time.date())
            
            for route_id, route_info in self.ahmedabad_routes.items():
                # Generate ridership for each hour
                base_ridership = self.generate_ridership_pattern(
                    current_time.hour,
                    current_time.weekday(),
                    weather_factor
                )
                
                # Add route-specific multipliers based on route popularity
                route_multipliers = {500: 1.5, 501: 1.3, 502: 1.8, 503: 1.1, 504: 1.4}
                ridership = int(base_ridership * route_multipliers.get(route_id, 1.0))
                
                # Calculate occupancy percentage
                bus_capacity = 60  # Typical city bus capacity
                occupancy = min(100, (ridership / bus_capacity) * 100)
                
                ridership_data.append({
                    'timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'route_id': route_id,
                    'ridership': ridership,
                    'occupancy_percent': round(occupancy, 1),
                    'weather_factor': round(weather_factor, 2),
                    'hour': current_time.hour,
                    'day_of_week': current_time.weekday(),
                    'is_weekend': current_time.weekday() >= 5,
                    'is_rush_hour': current_time.hour in [7, 8, 9, 17, 18, 19]
                })
            
            current_time += timedelta(hours=1)
        
        return pd.DataFrame(ridership_data)
    
    def generate_route_master_data(self):
        """Generate route master data"""
        route_data = []
        
        for route_id, route_info in self.ahmedabad_routes.items():
            route_data.append({
                'route_id': route_id,
                'route_name': f"Route {route_id}",
                'source': route_info['source'],
                'destination': route_info['destination'],
                'distance_km': route_info['distance_km'],
                'fare': route_info['fare'],
                'stops': json.dumps(route_info['stops']),
                'total_stops': len(route_info['stops']),
                'estimated_travel_time': len(route_info['stops']) * 5,  # 5 minutes per stop
                'operational_status': 'active'
            })
        
        return pd.DataFrame(route_data)
    
    def save_generated_data(self):
        """Generate and save all datasets"""
        print("Generating GPS logs...")
        gps_df = self.generate_gps_logs()
        gps_df.to_csv('/app/backend/data/gps_logs.csv', index=False)
        print(f"Generated {len(gps_df)} GPS log entries")
        
        print("Generating ridership data...")
        ridership_df = self.generate_ridership_data()
        ridership_df.to_json('/app/backend/data/ridership.json', orient='records', date_format='iso')
        print(f"Generated {len(ridership_df)} ridership entries")
        
        print("Generating route master data...")
        routes_df = self.generate_route_master_data()
        routes_df.to_csv('/app/backend/data/routes_master.csv', index=False)
        print(f"Generated {len(routes_df)} route entries")
        
        # Generate summary statistics
        summary = {
            'generation_date': datetime.now().isoformat(),
            'data_period': {
                'start_date': self.start_date.isoformat(),
                'end_date': self.end_date.isoformat(),
                'duration_days': (self.end_date - self.start_date).days
            },
            'statistics': {
                'total_gps_logs': len(gps_df),
                'total_ridership_entries': len(ridership_df),
                'total_routes': len(routes_df),
                'avg_daily_ridership': ridership_df.groupby(ridership_df['timestamp'].str[:10])['ridership'].sum().mean(),
                'peak_hour_ridership': ridership_df[ridership_df['is_rush_hour']]['ridership'].mean(),
                'off_peak_ridership': ridership_df[~ridership_df['is_rush_hour']]['ridership'].mean()
            }
        }
        
        with open('/app/backend/data/data_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\nData generation completed successfully!")
        print(f"Files saved in /app/backend/data/")
        print(f"Summary: {summary['statistics']}")

if __name__ == "__main__":
    # Create data directory
    import os
    os.makedirs('/app/backend/data', exist_ok=True)
    
    # Generate data
    generator = BangaloreBusDataGenerator()
    generator.save_generated_data()
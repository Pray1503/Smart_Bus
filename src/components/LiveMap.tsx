// import React, { useEffect, useRef, useState } from 'react';
// import Map from 'ol/Map.js';
// import View from 'ol/View.js';
// import TileLayer from 'ol/layer/Tile.js';
// import OSM from 'ol/source/OSM.js';
// import VectorLayer from 'ol/layer/Vector.js';
// import VectorSource from 'ol/source/Vector.js';
// import Feature from 'ol/Feature.js';
// import Point from 'ol/geom/Point.js';
// import { fromLonLat } from 'ol/proj.js';
// import { Style, Circle, Fill, Stroke, Text } from 'ol/style.js';
// import Overlay from 'ol/Overlay.js';
// import 'ol/ol.css';
// import { Card } from '@/components/ui/card';
// import { Bus } from 'lucide-react';

// interface BusData {
//   id: string;
//   route: string;
//   lat: number;
//   lng: number;
//   occupancy: number;
//   capacity: number;
//   speed: number;
//   nextStop: string;
// }

// const BusMap = () => {
//   const mapContainer = useRef<HTMLDivElement>(null);
//   const map = useRef<Map | null>(null);
//   const vectorSource = useRef<VectorSource>(new VectorSource());
//   const [buses, setBuses] = useState<BusData[]>([]);
//   const overlaysRef = useRef<Overlay[]>([]);

//   // Generate synthetic bus data
//   const generateBusData = (): BusData[] => {
//     const routes = ['Route 1A', 'Route 41C', 'Route 500D', 'Route G4', 'Route 414E'];
//     const stops = ['Majestic', 'Whitefield', 'Electronic City', 'Koramangala', 'Indiranagar', 'Banashankari'];

//     return Array.from({ length: 12 }, (_, i) => ({
//       id: `bus-${i + 1}`,
//       route: routes[Math.floor(Math.random() * routes.length)],
//       lat: 12.9716 + (Math.random() - 0.5) * 0.1,
//       lng: 77.5946 + (Math.random() - 0.5) * 0.1,
//       occupancy: Math.floor(Math.random() * 60) + 10,
//       capacity: 60,
//       speed: Math.floor(Math.random() * 30) + 15,
//       nextStop: stops[Math.floor(Math.random() * stops.length)],
//     }));
//   };

//   const getOccupancyColor = (occupancy: number, capacity: number) => {
//     const ratio = occupancy / capacity;
//     if (ratio < 0.5) return '#22c55e'; // Low - green
//     if (ratio < 0.8) return '#f59e0b'; // Medium - yellow
//     return '#ef4444'; // High - red
//   };

//   useEffect(() => {
//     if (!mapContainer.current) return;

//     // Initialize OpenLayers map
//     const vectorLayer = new VectorLayer({
//       source: vectorSource.current,
//     });

//     map.current = new Map({
//       target: mapContainer.current,
//       layers: [
//         new TileLayer({
//           source: new OSM(),
//         }),
//         vectorLayer,
//       ],
//       view: new View({
//         center: fromLonLat([77.5946, 12.9716]), // Bangalore
//         zoom: 12,
//       }),
//     });

//     // Initialize bus data
//     const initialBuses = generateBusData();
//     setBuses(initialBuses);
//     updateBusMarkers(initialBuses);

//     // Simulate bus movement
//     const moveInterval = setInterval(() => {
//       setBuses(prevBuses => {
//         const updatedBuses = prevBuses.map(bus => ({
//           ...bus,
//           lat: bus.lat + (Math.random() - 0.5) * 0.001,
//           lng: bus.lng + (Math.random() - 0.5) * 0.001,
//           occupancy: Math.max(5, Math.min(bus.capacity, bus.occupancy + Math.floor((Math.random() - 0.5) * 5))),
//           speed: Math.max(10, Math.min(50, bus.speed + Math.floor((Math.random() - 0.5) * 10))),
//         }));
//         updateBusMarkers(updatedBuses);
//         return updatedBuses;
//       });
//     }, 3000);

//     return () => {
//       clearInterval(moveInterval);
//       overlaysRef.current.forEach(overlay => map.current?.removeOverlay(overlay));
//       map.current?.dispose();
//     };
//   }, []);

//   const updateBusMarkers = (busData: BusData[]) => {
//     if (!map.current) return;

//     // Clear existing overlays
//     overlaysRef.current.forEach(overlay => map.current?.removeOverlay(overlay));
//     overlaysRef.current = [];
//     vectorSource.current.clear();

//     // Add new markers
//     busData.forEach(bus => {
//       // Create feature for the bus
//       const feature = new Feature({
//         geometry: new Point(fromLonLat([bus.lng, bus.lat])),
//         bus: bus,
//       });

//       // Style the bus marker
//       feature.setStyle(new Style({
//         image: new Circle({
//           radius: 12,
//           fill: new Fill({
//             color: getOccupancyColor(bus.occupancy, bus.capacity),
//           }),
//           stroke: new Stroke({
//             color: '#1e293b',
//             width: 2,
//           }),
//         }),
//         text: new Text({
//           text: '🚌',
//           font: '12px sans-serif',
//           fill: new Fill({ color: 'white' }),
//         }),
//       }));

//       vectorSource.current.addFeature(feature);

//       // Create popup overlay
//       const popupElement = document.createElement('div');
//       popupElement.className = 'bus-popup';
//       popupElement.style.cssText = `
//         background: white;
//         border-radius: 8px;
//         padding: 12px;
//         box-shadow: 0 4px 12px rgba(0,0,0,0.2);
//         border: 1px solid #e2e8f0;
//         font-family: system-ui;
//         color: #1e293b;
//         min-width: 200px;
//       `;
//       popupElement.innerHTML = `
//         <h3 style="margin: 0 0 8px 0; font-weight: bold; color: #1e40af;">${bus.route}</h3>
//         <p style="margin: 2px 0; font-size: 14px;"><strong>Bus ID:</strong> ${bus.id}</p>
//         <p style="margin: 2px 0; font-size: 14px;"><strong>Occupancy:</strong> ${bus.occupancy}/${bus.capacity}</p>
//         <p style="margin: 2px 0; font-size: 14px;"><strong>Speed:</strong> ${bus.speed} km/h</p>
//         <p style="margin: 2px 0; font-size: 14px;"><strong>Next Stop:</strong> ${bus.nextStop}</p>
//       `;

//       const overlay = new Overlay({
//         element: popupElement,
//         positioning: 'bottom-center',
//         stopEvent: false,
//         offset: [0, -20],
//       });

//       map.current?.addOverlay(overlay);
//       overlaysRef.current.push(overlay);

//       // Hide popup initially
//       popupElement.style.display = 'none';
//     });

//     // Add click handler for popups
//     map.current?.on('click', (evt) => {
//       const feature = map.current?.forEachFeatureAtPixel(evt.pixel, (feature) => feature);

//       // Hide all popups first
//       overlaysRef.current.forEach(overlay => {
//         const element = overlay.getElement();
//         if (element) element.style.display = 'none';
//       });

//       if (feature) {
//         const busData = feature.get('bus') as BusData;
//         const busIndex = buses.findIndex(b => b.id === busData.id);
//         if (busIndex >= 0 && overlaysRef.current[busIndex]) {
//           const overlay = overlaysRef.current[busIndex];
//           overlay.setPosition(fromLonLat([busData.lng, busData.lat]));
//           const element = overlay.getElement();
//           if (element) element.style.display = 'block';
//         }
//       }
//     });
//   };

//   return (
//     <Card className="p-0 overflow-hidden bg-card border-border shadow-card-enhanced">
//       <div className="p-4 border-b border-border bg-gradient-control">
//         <div className="flex items-center justify-between">
//           <div className="flex items-center gap-2">
//             <Bus className="h-5 w-5 text-primary" />
//             <h3 className="text-lg font-semibold text-foreground">Live Bus Tracking</h3>
//           </div>
//           <div className="flex items-center gap-4 text-sm text-muted-foreground">
//             <div className="flex items-center gap-2">
//               <div className="w-3 h-3 rounded-full bg-success"></div>
//               <span>Low Occupancy</span>
//             </div>
//             <div className="flex items-center gap-2">
//               <div className="w-3 h-3 rounded-full bg-warning"></div>
//               <span>Medium</span>
//             </div>
//             <div className="flex items-center gap-2">
//               <div className="w-3 h-3 rounded-full bg-destructive"></div>
//               <span>High</span>
//             </div>
//           </div>
//         </div>
//       </div>
//       <div className="relative">
//         <div ref={mapContainer} className="h-[500px] w-full" />
//         <div className="absolute top-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border border-border">
//           <div className="text-sm text-foreground">
//             <p className="font-semibold">Active Buses: {buses.length}</p>
//             <p className="text-muted-foreground">Real-time tracking enabled</p>
//           </div>
//         </div>
//       </div>
//     </Card>
//   );
// };

// export default BusMap;

import React, { useEffect, useRef, useState } from "react";
import Map from "ol/Map.js";
import View from "ol/View.js";
import TileLayer from "ol/layer/Tile.js";
import OSM from "ol/source/OSM.js";
import VectorLayer from "ol/layer/Vector.js";
import VectorSource from "ol/source/Vector.js";
import Feature from "ol/Feature.js";
import Point from "ol/geom/Point.js";
import { fromLonLat } from "ol/proj.js";
import { Style, Circle, Fill, Stroke, Text } from "ol/style.js";
import Overlay from "ol/Overlay.js";
import "ol/ol.css";
import { Card } from "./card";
import { Bus } from "lucide-react";

interface BusData {
  id: string;
  route: string;
  source: string;
  destination: string;
  distance: number;
  firstShift: number;
  secondShift: number;
  fare: string;
  lat: number;
  lng: number;
  occupancy: number;
  capacity: number;
  speed: number;
  nextStop: string;
}

const busRoutes = [
  {
    "Route No.": "1",
    Source: "Ratan Park",
    Destination: "Lal Darwaja",
    "Distance Covered": 9.5,
    "1st Shift Trip": 13,
    "2nd Shift Trip": 14,
    Fare: "Rs 20",
  },
  {
    "Route No.": "4",
    Source: "Lal Darwaja",
    Destination: "Lal Darwaja",
    "Distance Covered": 22.8,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "5",
    Source: "Lal Darwaja",
    Destination: "Lal Darwaja",
    "Distance Covered": 22.8,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "14",
    Source: "Lal Darwaja",
    Destination: "Chosar Gam",
    "Distance Covered": 18.7,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "15",
    Source: "Vivekanand Nagar",
    Destination: "Civil Hospital",
    "Distance Covered": 22.15,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 8,
    Fare: "Rs 30",
  },
  {
    "Route No.": "16",
    Source: "Nigam Society",
    Destination: "Chiloda Octroi Naka",
    "Distance Covered": 27.8,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "17",
    Source: "Nigam Society",
    Destination: "Meghani Nagar",
    "Distance Covered": 16.45,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "18",
    Source: "Kalupur",
    Destination: "Punit Nagar",
    "Distance Covered": 8.9,
    "1st Shift Trip": 14,
    "2nd Shift Trip": 14,
    Fare: "Rs 20",
  },
  {
    "Route No.": "22",
    Source: "Tragad Gam",
    Destination: "Lambha Gam",
    "Distance Covered": 32.05,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "23",
    Source: "Isanpur",
    Destination: "Jivandeep Circular",
    "Distance Covered": 22.9,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "28",
    Source: "Meghani Nagar",
    Destination: "Lambha Gam",
    "Distance Covered": 19.85,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "31",
    Source: "Sarkhej Gam",
    Destination: "Meghaninagar",
    "Distance Covered": 19.85,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "32",
    Source: "Butbhavani Mand",
    Destination: "Shahiyadri Bung",
    "Distance Covered": 18.6,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "33",
    Source: "Narayan Nagar",
    Destination: "Manmohan Park",
    "Distance Covered": 19.65,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "34",
    Source: "Butbhavani Mand",
    Destination: "Kalapi Nagar",
    "Distance Covered": 17.55,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "35",
    Source: "Lal Darwaja",
    Destination: "Matoda Patia",
    "Distance Covered": 25.95,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "36",
    Source: "Sarangpur",
    Destination: "Sarkhej Gam",
    "Distance Covered": 14.0,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "37",
    Source: "Vasna",
    Destination: "Tejendra Nagar",
    "Distance Covered": 17.1,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "38",
    Source: "Juhapura",
    Destination: "Meghani Nagar",
    "Distance Covered": 16.65,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "40",
    Source: "Vasna",
    Destination: "Lapkaman",
    "Distance Covered": 21.95,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "42",
    Source: "Ghodasar",
    Destination: "Judges Bunglows",
    "Distance Covered": 17.65,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "43",
    Source: "Lal Darwaja",
    Destination: "Judges Bunglow",
    "Distance Covered": 9.4,
    "1st Shift Trip": 14,
    "2nd Shift Trip": 12,
    Fare: "Rs 20",
  },
  {
    "Route No.": "45",
    Source: "Lal Darwaja",
    Destination: "Jodhpur Gam",
    "Distance Covered": 8.4,
    "1st Shift Trip": 15,
    "2nd Shift Trip": 14,
    Fare: "Rs 20",
  },
  {
    "Route No.": "46",
    Source: "Kalupur",
    Destination: "Kalupur",
    "Distance Covered": 18.2,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "47",
    Source: "Kalupur",
    Destination: "Kalupur",
    "Distance Covered": 18.2,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "48",
    Source: "Kalupur",
    Destination: "Prhalad Nagar",
    "Distance Covered": 14.25,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 25",
  },
  {
    "Route No.": "49",
    Source: "Adinath Nagar",
    Destination: "Manipur Vad",
    "Distance Covered": 29.15,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "50",
    Source: "Ghuma Gam",
    Destination: "Meghani Nagar",
    "Distance Covered": 25.7,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "52",
    Source: "Punit Nagar",
    Destination: "Thaltej",
    "Distance Covered": 21.6,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "54",
    Source: "Vatva Rly Cross",
    Destination: "Vaishnodevi Man",
    "Distance Covered": 34.4,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "56",
    Source: "Sitaram Bapa Chowk",
    Destination: "Judges Bunglows",
    "Distance Covered": 24.95,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "58",
    Source: "Thaltej Gam",
    Destination: "Kush Society",
    "Distance Covered": 30.15,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "60",
    Source: "Maninagar",
    Destination: "Judges Bunglows",
    "Distance Covered": 18.65,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "61",
    Source: "Maninagar",
    Destination: "Gujarat High Court",
    "Distance Covered": 19.6,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "63",
    Source: "Maninagar",
    Destination: "Gujarat High Court",
    "Distance Covered": 19.45,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "64",
    Source: "Lal Darwaja",
    Destination: "Gujarat High Court",
    "Distance Covered": 11.35,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "65",
    Source: "Lal Darwaja",
    Destination: "Sola Bhagwat Vidhyapith",
    "Distance Covered": 13.55,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 20",
  },
  {
    "Route No.": "65",
    Source: "Lal Darwaja",
    Destination: "Sola Bhagwat",
    "Distance Covered": 13.55,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 20",
  },
  {
    "Route No.": "66",
    Source: "Kalupur Terminu",
    Destination: "Shilaj Gam",
    "Distance Covered": 16.5,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "67",
    Source: "Kalupur",
    Destination: "Satadhar Society",
    "Distance Covered": 11.1,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "68",
    Source: "Kalupur",
    Destination: "Sattadhar Society",
    "Distance Covered": 17.2,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 25",
  },
  {
    "Route No.": "69",
    Source: "Kalupur",
    Destination: "Chanakyapuri",
    "Distance Covered": 10.35,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "70",
    Source: "Vagheshwari Soc",
    Destination: "Naroda Terminus",
    "Distance Covered": 20.05,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "72",
    Source: "Maninagar",
    Destination: "Nava Vadaj",
    "Distance Covered": 18.2,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 25",
  },
  {
    "Route No.": "74",
    Source: "Nava Ranip",
    Destination: "Nigam Society",
    "Distance Covered": 21.55,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "75",
    Source: "Maninagar",
    Destination: "Chandkheda",
    "Distance Covered": 21.2,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "76",
    Source: "Vatva Ind.Towns",
    Destination: "Gujarat High Court",
    "Distance Covered": 25.7,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "77",
    Source: "Vadaj Terminus",
    Destination: "Hatkeshwar",
    "Distance Covered": 12.35,
    "1st Shift Trip": 9,
    "2nd Shift Trip": 9,
    Fare: "Rs 20",
  },
  {
    "Route No.": "79",
    Source: "Thakkarbapa Nagar",
    Destination: "Chenpur Gam",
    "Distance Covered": 19.0,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "82",
    Source: "Lal Darwaja",
    Destination: "Nirnay Nagar",
    "Distance Covered": 10.45,
    "1st Shift Trip": 11,
    "2nd Shift Trip": 11,
    Fare: "Rs 20",
  },
  {
    "Route No.": "83",
    Source: "Lal Darwaja",
    Destination: "Sabarmati D Cab",
    "Distance Covered": 14.1,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "84",
    Source: "Mani Nagar",
    Destination: "Chandkheda Gam",
    "Distance Covered": 23.6,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "85",
    Source: "Lal Darwaja",
    Destination: "Chandkheda Gam",
    "Distance Covered": 12.85,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 20",
  },
  {
    "Route No.": "87",
    Source: "Maninagar",
    Destination: "Chandkheda Gam",
    "Distance Covered": 24.95,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "88",
    Source: "Ranip",
    Destination: "Nikol Gam",
    "Distance Covered": 16.45,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "90",
    Source: "Tragad Gam",
    Destination: "Meghaninagar",
    "Distance Covered": 21.15,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "96",
    Source: "Vatva Rly Cross",
    Destination: "Circuit House",
    "Distance Covered": 22.65,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "101",
    Source: "Lal Darwaja",
    Destination: "Sardar Nagar",
    "Distance Covered": 9.45,
    "1st Shift Trip": 14,
    "2nd Shift Trip": 14,
    Fare: "Rs 20",
  },
  {
    "Route No.": "102",
    Source: "Lal Darwaja",
    Destination: "New Airport",
    "Distance Covered": 10.8,
    "1st Shift Trip": 12,
    "2nd Shift Trip": 12,
    Fare: "Rs 20",
  },
  {
    "Route No.": "105",
    Source: "Lal Darwaja",
    Destination: "Naroda Ind East",
    "Distance Covered": 16.35,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "112",
    Source: "Lal Darwaja",
    Destination: "Kubernagar",
    "Distance Covered": 11.05,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "116",
    Source: "Civil Hospital",
    Destination: "Danilimda Gam",
    "Distance Covered": 9.5,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "117",
    Source: "Suedge Farm App",
    Destination: "Kalapi Nagar",
    "Distance Covered": 12.85,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "122",
    Source: "Lal Darwaja",
    Destination: "Ambawadi Police",
    "Distance Covered": 15.25,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "123 SH",
    Source: "Lal Darwaja",
    Destination: "Krushna Nagar",
    "Distance Covered": 10.35,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "125",
    Source: "Lal Darwaja",
    Destination: "Vahelal Gam",
    "Distance Covered": 27.45,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "126",
    Source: "Sarangpur",
    Destination: "Sardarnagar",
    "Distance Covered": 14.4,
    "1st Shift Trip": 9,
    "2nd Shift Trip": 9,
    Fare: "Rs 25",
  },
  {
    "Route No.": "127",
    Source: "Sarangpur",
    Destination: "Sukan Bunglow",
    "Distance Covered": 12.4,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "128",
    Source: "Mani Nagar",
    Destination: "Naroda Ind Town",
    "Distance Covered": 17.25,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "129",
    Source: "Haridarshan",
    Destination: "Vasna",
    "Distance Covered": 23.35,
    "1st Shift Trip": 7,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "130",
    Source: "Naroda Terminus",
    Destination: "Indira Nagar 2",
    "Distance Covered": 24.35,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "134",
    Source: "Lal Darwaja",
    Destination: "Thakkarbapanagar",
    "Distance Covered": 10.75,
    "1st Shift Trip": 10,
    "2nd Shift Trip": 10,
    Fare: "Rs 20",
  },
  {
    "Route No.": "135",
    Source: "Lal Darwaja",
    Destination: "New India Colony",
    "Distance Covered": 14.55,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "136",
    Source: "New India Colon",
    Destination: "Sattadhar Society",
    "Distance Covered": 27.0,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "138",
    Source: "Bapunagar",
    Destination: "Ghuma Gam",
    "Distance Covered": 23.7,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "141",
    Source: "Lal Darwaja",
    Destination: "Rakhial Char Rasta",
    "Distance Covered": 7.95,
    "1st Shift Trip": 14,
    "2nd Shift Trip": 14,
    Fare: "Rs 15",
  },
  {
    "Route No.": "142",
    Source: "Vastral Gam",
    Destination: "Gujarat University",
    "Distance Covered": 19.05,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 7,
    Fare: "Rs 25",
  },
  {
    "Route No.": "143",
    Source: "Lal Darwaja",
    Destination: "Bhuvaldi Gam",
    "Distance Covered": 16.25,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "144",
    Source: "Arbuda Nagar",
    Destination: "Gujarat University",
    "Distance Covered": 15.95,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "145",
    Source: "Arbudanagar",
    Destination: "Civil Hospital",
    "Distance Covered": 10.2,
    "1st Shift Trip": 13,
    "2nd Shift Trip": 12,
    Fare: "Rs 20",
  },
  {
    "Route No.": "147",
    Source: "Surbhi Society",
    Destination: "Vagheshwari Society",
    "Distance Covered": 18.85,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "148",
    Source: "Sarangpur",
    Destination: "Kathwada Gam",
    "Distance Covered": 13.85,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 20",
  },
  {
    "Route No.": "150",
    Source: "Sarkhej Gam",
    Destination: "Chinubhai Nagar",
    "Distance Covered": 24.05,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "152",
    Source: "Lal Darwaja",
    Destination: "Vanch",
    "Distance Covered": 17.4,
    "1st Shift Trip": 8,
    "2nd Shift Trip": 8,
    Fare: "Rs 25",
  },
  {
    "Route No.": "153",
    Source: "Lal Darwaja",
    Destination: "Shyamaprasad Vasavad Community Hall",
    "Distance Covered": 7.9,
    "1st Shift Trip": 14,
    "2nd Shift Trip": 14,
    Fare: "Rs 15",
  },
  {
    "Route No.": "160",
    Source: "Hatkeshwer",
    Destination: "Gujarat High Court",
    "Distance Covered": 26.75,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "200",
    Source: "Maninagar",
    Destination: "Maninagar",
    "Distance Covered": 42.85,
    "1st Shift Trip": 3,
    "2nd Shift Trip": 3,
    Fare: "Rs 30",
  },
  {
    "Route No.": "201",
    Source: "Naroda Terminus",
    Destination: "Vasna Terminus",
    "Distance Covered": 26.3,
    "1st Shift Trip": 5,
    "2nd Shift Trip": 5,
    Fare: "Rs 30",
  },
  {
    "Route No.": "202",
    Source: "Vasna",
    Destination: "Naroda Terminus",
    "Distance Covered": 32.6,
    "1st Shift Trip": 4,
    "2nd Shift Trip": 4,
    Fare: "Rs 30",
  },
  {
    "Route No.": "203",
    Source: "Paldi",
    Destination: "Vaishno Devi",
    "Distance Covered": 28.35,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 7,
    Fare: "Rs 30",
  },
  {
    "Route No.": "300",
    Source: "Maninagar",
    Destination: "Maninagar",
    "Distance Covered": 42.85,
    "1st Shift Trip": 3,
    "2nd Shift Trip": 3,
    Fare: "Rs 30",
  },
  {
    "Route No.": "301",
    Source: "Naroda Terminus",
    Destination: "Naroda Terminus",
    "Distance Covered": 61.65,
    "1st Shift Trip": 3,
    "2nd Shift Trip": 3,
    Fare: "Rs 30",
  },
  {
    "Route No.": "400",
    Source: "Lal Darwaja",
    Destination: "Lal Darwaja",
    "Distance Covered": 20.05,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "401",
    Source: "Vasna",
    Destination: "Chandkheda",
    "Distance Covered": 52.75,
    "1st Shift Trip": 3,
    "2nd Shift Trip": 3,
    Fare: "Rs 30",
  },
  {
    "Route No.": "500",
    Source: "Lal Darwaja",
    Destination: "Lal Darwaja",
    "Distance Covered": 20.05,
    "1st Shift Trip": 6,
    "2nd Shift Trip": 6,
    Fare: "Rs 30",
  },
  {
    "Route No.": "501",
    Source: "Vasna",
    Destination: "Chandkheda",
    "Distance Covered": 52.75,
    "1st Shift Trip": 3,
    "2nd Shift Trip": 2,
    Fare: "Rs 30",
  },
  {
    "Route No.": "800",
    Source: "Nava Vadaj",
    Destination: "Nava Vadaj",
    "Distance Covered": 38.3,
    "1st Shift Trip": 4,
    "2nd Shift Trip": 4,
    Fare: "Rs 30",
  },
  {
    "Route No.": "900",
    Source: "Nava Vadaj",
    Destination: "Nava Vadaj",
    "Distance Covered": 38.3,
    "1st Shift Trip": 4,
    "2nd Shift Trip": 4,
    Fare: "Rs 30",
  },
];

export const BusMap = () => {
  const mapContainer = useRef<HTMLDivElement>(null);
  const map = useRef<Map | null>(null);
  const vectorSource = useRef<VectorSource>(new VectorSource());
  const [buses, setBuses] = useState<BusData[]>([]);
  const overlaysRef = useRef<Overlay[]>([]);

  // Generate bus data from provided routes
  const generateBusData = (): BusData[] => {
    const allStops = [
      ...new Set(busRoutes.flatMap((r) => [r.Source, r.Destination])),
    ];

    return busRoutes.map((route, i) => ({
      id: `bus-${route["Route No."]}`,
      route: route["Route No."],
      source: route.Source,
      destination: route.Destination,
      distance: route["Distance Covered"],
      firstShift: route["1st Shift Trip"],
      secondShift: route["2nd Shift Trip"],
      fare: route.Fare,
      lat: 23.0225 + (Math.random() - 0.5) * 0.1,
      lng: 72.5714 + (Math.random() - 0.5) * 0.1,
      occupancy: Math.floor(Math.random() * 60) + 10,
      capacity: 60,
      speed: Math.floor(Math.random() * 30) + 15,
      nextStop: allStops[Math.floor(Math.random() * allStops.length)],
    }));
  };

  const getOccupancyColor = (occupancy: number, capacity: number) => {
    const ratio = occupancy / capacity;
    if (ratio < 0.5) return "#22c55e"; // Low - green
    if (ratio < 0.8) return "#f59e0b"; // Medium - yellow
    return "#ef4444"; // High - red
  };

  useEffect(() => {
    if (!mapContainer.current) return;

    // Initialize OpenLayers map
    const vectorLayer = new VectorLayer({
      source: vectorSource.current,
    });

    map.current = new Map({
      target: mapContainer.current,
      layers: [
        new TileLayer({
          source: new OSM(),
        }),
        vectorLayer,
      ],
      view: new View({
        center: fromLonLat([72.5714, 23.0225]), // Ahmedabad
        zoom: 12,
      }),
    });

    // Initialize bus data
    const initialBuses = generateBusData();
    setBuses(initialBuses);
    updateBusMarkers(initialBuses);

    // Simulate bus movement
    const moveInterval = setInterval(() => {
      setBuses((prevBuses) => {
        const updatedBuses = prevBuses.map((bus) => ({
          ...bus,
          lat: bus.lat + (Math.random() - 0.5) * 0.001,
          lng: bus.lng + (Math.random() - 0.5) * 0.001,
          occupancy: Math.max(
            5,
            Math.min(
              bus.capacity,
              bus.occupancy + Math.floor((Math.random() - 0.5) * 5)
            )
          ),
          speed: Math.max(
            10,
            Math.min(50, bus.speed + Math.floor((Math.random() - 0.5) * 10))
          ),
        }));
        updateBusMarkers(updatedBuses);
        return updatedBuses;
      });
    }, 3000);

    return () => {
      clearInterval(moveInterval);
      overlaysRef.current.forEach((overlay) =>
        map.current?.removeOverlay(overlay)
      );
      map.current?.dispose();
    };
  }, []);

  const updateBusMarkers = (busData: BusData[]) => {
    if (!map.current) return;

    // Clear existing overlays
    overlaysRef.current.forEach((overlay) =>
      map.current?.removeOverlay(overlay)
    );
    overlaysRef.current = [];
    vectorSource.current.clear();

    // Add new markers
    busData.forEach((bus) => {
      // Create feature for the bus
      const feature = new Feature({
        geometry: new Point(fromLonLat([bus.lng, bus.lat])),
        bus: bus,
      });

      // Style the bus marker
      feature.setStyle(
        new Style({
          image: new Circle({
            radius: 12,
            fill: new Fill({
              color: getOccupancyColor(bus.occupancy, bus.capacity),
            }),
            stroke: new Stroke({
              color: "#1e293b",
              width: 2,
            }),
          }),
          text: new Text({
            text: "🚌",
            font: "12px sans-serif",
            fill: new Fill({ color: "white" }),
          }),
        })
      );

      vectorSource.current.addFeature(feature);

      // Create popup overlay
      const popupElement = document.createElement("div");
      popupElement.className = "bus-popup";
      popupElement.style.cssText = `
        background: white;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        border: 1px solid #e2e8f0;
        font-family: system-ui;
        color: #1e293b;
        min-width: 200px;
      `;
      popupElement.innerHTML = `
        <h3 style="margin: 0 0 8px 0; font-weight: bold; color: #1e40af;">Route ${bus.route}</h3>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Source:</strong> ${bus.source}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Destination:</strong> ${bus.destination}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Distance:</strong> ${bus.distance} km</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Fare:</strong> ${bus.fare}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>1st Shift Trips:</strong> ${bus.firstShift}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>2nd Shift Trips:</strong> ${bus.secondShift}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Bus ID:</strong> ${bus.id}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Occupancy:</strong> ${bus.occupancy}/${bus.capacity}</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Speed:</strong> ${bus.speed} km/h</p>
        <p style="margin: 2px 0; font-size: 14px;"><strong>Next Stop:</strong> ${bus.nextStop}</p>
      `;

      const overlay = new Overlay({
        element: popupElement,
        positioning: "bottom-center",
        stopEvent: false,
        offset: [0, -20],
      });

      map.current?.addOverlay(overlay);
      overlaysRef.current.push(overlay);

      // Hide popup initially
      popupElement.style.display = "none";
    });

    // Add click handler for popups
    map.current?.on("click", (evt) => {
      const feature = map.current?.forEachFeatureAtPixel(
        evt.pixel,
        (feature) => feature
      );

      // Hide all popups first
      overlaysRef.current.forEach((overlay) => {
        const element = overlay.getElement();
        if (element) element.style.display = "none";
      });

      if (feature) {
        const busData = feature.get("bus") as BusData;
        const busIndex = buses.findIndex((b) => b.id === busData.id);
        if (busIndex >= 0 && overlaysRef.current[busIndex]) {
          const overlay = overlaysRef.current[busIndex];
          overlay.setPosition(fromLonLat([busData.lng, busData.lat]));
          const element = overlay.getElement();
          if (element) element.style.display = "block";
        }
      }
    });
  };

  return (
    <Card className="p-0 overflow-hidden bg-card border-border shadow-card-enhanced">
      <div className="p-4 border-b border-border bg-gradient-control">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Bus className="h-5 w-5 text-primary" />
            <h3 className="text-lg font-semibold text-foreground">
              AMTS Live Bus Tracking
            </h3>
          </div>
          <div className="flex items-center gap-4 text-sm text-muted-foreground">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-success"></div>
              <span>Low Occupancy</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-warning"></div>
              <span>Medium</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-destructive"></div>
              <span>High</span>
            </div>
          </div>
        </div>
      </div>
      <div className="relative">
        <div ref={mapContainer} className="h-[500px] w-full" />
        <div className="absolute top-4 left-4 bg-card/90 backdrop-blur-sm rounded-lg p-3 border border-border">
          <div className="text-sm text-foreground">
            <p className="font-semibold">Active Buses: {buses.length}</p>
            <p className="text-muted-foreground">Real-time tracking enabled</p>
          </div>
        </div>
      </div>
    </Card>
  );
};

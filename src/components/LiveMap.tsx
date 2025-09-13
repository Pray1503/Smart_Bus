import React from 'react';
import { motion } from 'framer-motion';
import { Bus as BusIcon, MapPin } from 'lucide-react';
import { Bus, RouteStop } from '../types';

interface LiveMapProps {
  buses: Bus[];
  stops: RouteStop[];
}

export const LiveMap: React.FC<LiveMapProps> = ({ buses, stops }) => {
  const getStatusColor = (status: Bus['status']) => {
    switch (status) {
      case 'normal': return 'text-emerald-400';
      case 'delayed': return 'text-amber-400';
      case 'overcrowded': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getStatusBg = (status: Bus['status']) => {
    switch (status) {
      case 'normal': return 'bg-emerald-500/20 border-emerald-400';
      case 'delayed': return 'bg-amber-500/20 border-amber-400';
      case 'overcrowded': return 'bg-red-500/20 border-red-400';
      default: return 'bg-gray-500/20 border-gray-400';
    }
  };

  return (
    <div className="bg-gray-900/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Live Bus Tracking</h3>
        <div className="flex items-center space-x-4 text-xs">
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-emerald-400 rounded-full"></div>
            <span className="text-gray-300">Normal</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-amber-400 rounded-full"></div>
            <span className="text-gray-300">Delayed</span>
          </div>
          <div className="flex items-center space-x-1">
            <div className="w-2 h-2 bg-red-400 rounded-full"></div>
            <span className="text-gray-300">Overcrowded</span>
          </div>
        </div>
      </div>
      
      <div className="relative bg-gray-800/50 rounded-lg h-96 overflow-hidden border border-gray-600">
        {/* Route Lines */}
        <svg className="absolute inset-0 w-full h-full" style={{ zIndex: 1 }}>
          {stops.map((stop, index) => 
            stops.slice(index + 1).map((nextStop, nextIndex) => (
              <line
                key={`${stop.id}-${nextStop.id}`}
                x1={stop.x}
                y1={stop.y}
                x2={nextStop.x}
                y2={nextStop.y}
                stroke="rgb(59, 130, 246)"
                strokeWidth="2"
                strokeOpacity="0.3"
                strokeDasharray="5,5"
              />
            ))
          )}
        </svg>

        {/* Bus Stops */}
        {stops.map((stop) => (
          <motion.div
            key={stop.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            style={{ left: stop.x, top: stop.y, zIndex: 2 }}
            whileHover={{ scale: 1.1 }}
          >
            <div className="flex flex-col items-center">
              <MapPin className="w-4 h-4 text-blue-400" />
              <span className="text-xs text-gray-300 mt-1 bg-gray-900/80 px-2 py-1 rounded">
                {stop.name}
              </span>
            </div>
          </motion.div>
        ))}

        {/* Buses */}
        {buses.map((bus) => (
          <motion.div
            key={bus.id}
            className="absolute transform -translate-x-1/2 -translate-y-1/2"
            animate={{ 
              x: bus.x, 
              y: bus.y,
            }}
            transition={{ 
              duration: 2,
              ease: "linear"
            }}
            style={{ zIndex: 3 }}
          >
            <motion.div
              className={`relative p-2 rounded-lg border-2 ${getStatusBg(bus.status)}`}
              whileHover={{ scale: 1.1 }}
              animate={{ 
                boxShadow: bus.status === 'normal' ? '0 0 20px rgba(16, 185, 129, 0.3)' :
                          bus.status === 'delayed' ? '0 0 20px rgba(245, 158, 11, 0.3)' :
                          '0 0 20px rgba(239, 68, 68, 0.3)'
              }}
            >
              <BusIcon className={`w-6 h-6 ${getStatusColor(bus.status)}`} />
              
              {/* Bus Info */}
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2 bg-gray-900/90 px-2 py-1 rounded text-xs text-white whitespace-nowrap">
                <div className="font-semibold">{bus.id}</div>
                <div className="text-gray-300">{bus.occupancy}/{bus.maxCapacity}</div>
              </div>

              {/* Route Info */}
              <div className="absolute -bottom-8 left-1/2 transform -translate-x-1/2 bg-gray-900/90 px-2 py-1 rounded text-xs text-white whitespace-nowrap">
                <div className="text-blue-400">{bus.route}</div>
              </div>
            </motion.div>
          </motion.div>
        ))}
      </div>
    </div>
  );
};
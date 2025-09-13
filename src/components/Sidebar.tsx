import React from 'react';
import { motion } from 'framer-motion';
import { Bus, Filter, Search } from 'lucide-react';
import { Route } from '../types';

interface SidebarProps {
  routes: Route[];
  selectedRoute: string | null;
  onRouteSelect: (routeId: string) => void;
  searchTerm: string;
  onSearchChange: (term: string) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ 
  routes, 
  selectedRoute, 
  onRouteSelect, 
  searchTerm, 
  onSearchChange 
}) => {
  const getStatusColor = (status: Route['status']) => {
    switch (status) {
      case 'optimal': return 'bg-emerald-500';
      case 'delayed': return 'bg-amber-500';
      case 'active': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const filteredRoutes = routes.filter(route =>
    route.name.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="w-80 bg-gray-900 border-r border-gray-700 flex flex-col h-full">
      {/* Header */}
      <div className="p-6 border-b border-gray-700">
        <div className="flex items-center space-x-3 mb-6">
          <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center">
            <Bus className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Smart Bus Dashboard</h1>
            <p className="text-sm text-gray-400">View Project</p>
          </div>
        </div>
      </div>

      {/* Routes Section */}
      <div className="flex-1 overflow-hidden">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Routes</h2>
          </div>

          {/* Search */}
          <div className="relative mb-4">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              placeholder="Search routes..."
              value={searchTerm}
              onChange={(e) => onSearchChange(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:border-emerald-500"
            />
          </div>

          {/* Filters */}
          <div className="flex items-center space-x-2 mb-6">
            <Filter className="w-4 h-4 text-gray-400" />
            <span className="text-sm text-gray-400">Filters</span>
          </div>
        </div>

        {/* Route List */}
        <div className="px-6 pb-6 overflow-y-auto flex-1">
          <div className="space-y-3">
            {filteredRoutes.map((route) => (
              <motion.div
                key={route.id}
                className={`p-4 rounded-lg border cursor-pointer transition-all ${
                  selectedRoute === route.id
                    ? 'bg-gray-700 border-emerald-500'
                    : 'bg-gray-800 border-gray-600 hover:bg-gray-750 hover:border-gray-500'
                }`}
                onClick={() => onRouteSelect(route.id)}
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${getStatusColor(route.status)}`} />
                    <h3 className="font-medium text-white">{route.name}</h3>
                  </div>
                  {route.status === 'delayed' && (
                    <div className="w-2 h-2 bg-amber-500 rounded-full animate-pulse" />
                  )}
                </div>
                
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-gray-400">{route.buses} buses</p>
                    <p className="text-gray-400">{route.avgWait} wait</p>
                  </div>
                  <div>
                    <p className="text-gray-400">{route.stops} stops</p>
                    <p className="text-gray-400">{route.onTime} on-time</p>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
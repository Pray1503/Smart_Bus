import React from 'react';
import { motion } from 'framer-motion';
import { Bus, Clock, TrendingUp, AlertTriangle, Settings, Bell } from 'lucide-react';

interface HeaderProps {
  stats: {
    activeBuses: number;
    avgWaitTime: number;
    onTimePercentage: number;
    alerts: number;
  };
  notificationCount: number;
}

export const Header: React.FC<HeaderProps> = ({ stats, notificationCount }) => {
  return (
    <div className="bg-gray-900 border-b border-gray-700 px-6 py-4">
      <div className="flex items-center justify-between">
        {/* Stats */}
        <div className="flex items-center space-x-8">
          <div className="flex items-center space-x-2">
            <Bus className="w-5 h-5 text-emerald-400" />
            <div>
              <p className="text-sm text-gray-400">Active Buses</p>
              <p className="text-xl font-bold text-white">{stats.activeBuses}</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <Clock className="w-5 h-5 text-blue-400" />
            <div>
              <p className="text-sm text-gray-400">Avg Wait Time</p>
              <p className="text-xl font-bold text-white">{stats.avgWaitTime}m</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <TrendingUp className="w-5 h-5 text-emerald-400" />
            <div>
              <p className="text-sm text-gray-400">On-Time %</p>
              <p className="text-xl font-bold text-white">{stats.onTimePercentage}%</p>
            </div>
          </div>

          <div className="flex items-center space-x-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <div>
              <p className="text-sm text-gray-400">Alerts</p>
              <p className="text-xl font-bold text-white">{stats.alerts}</p>
            </div>
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center space-x-4">
          <motion.button
            className="relative p-2 text-gray-400 hover:text-white transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Bell className="w-5 h-5" />
            {notificationCount > 0 && (
              <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 text-white text-xs rounded-full flex items-center justify-center">
                {notificationCount}
              </span>
            )}
          </motion.button>

          <motion.button
            className="p-2 text-gray-400 hover:text-white transition-colors"
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            <Settings className="w-5 h-5" />
          </motion.button>

          <div className="w-8 h-8 bg-gradient-to-r from-emerald-400 to-blue-500 rounded-full" />
        </div>
      </div>
    </div>
  );
};